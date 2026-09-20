from runtime import configure_runtime
configure_runtime()

import csv
import json
import re
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor

from eval import (
    eval_output_dir,
    load_local_policy_model,
    local_dataset_root,
    resolve_eval_dataset,
    write_eval_results,
)
from utils import NpyMemmapDataset


CSV_PROBE_TARGET = re.compile(
    r"^(gripper|moving_\d+|static_\d+|background_\d+)_(xy|color_rgb|category)$"
)
FOREGROUND_CATEGORY_IDS = {
    "food": 0,
    "cookware": 1,
    "towel": 2,
    "small object": 3,
}
BACKGROUND_CATEGORY_IDS = {
    "table": 0,
    "countertop": 1,
    "sink": 2
}
OBJECT_SLOT_TARGET = re.compile(r"^(moving|static|background)(\d+)_centroid$")


def _slice_from_spec(spec):
    if spec is None:
        return slice(None)
    if isinstance(spec, (int, np.integer)):
        return [int(spec)]
    if isinstance(spec, str):
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) > 3:
            raise ValueError(f"Invalid slice spec {spec!r}")
        values = [int(part) if part else None for part in parts]
        return slice(*values)
    if isinstance(spec, (list, tuple)):
        return [int(idx) for idx in spec]
    raise TypeError(f"Unsupported slice spec: {spec!r}")


def _values_from_spec(values, index):
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        values = values[None]
    if isinstance(index, slice):
        return values[index]
    if values.ndim == 1 and values.size == len(index):
        return values
    return values[np.asarray(index, dtype=np.int64)]


def default_probe_targets(dataset_name):
    name = str(dataset_name).replace("\\", "/").rstrip("/").split("/")[-1].lower()
    if name.endswith(".npy") or name.endswith(".h5"):
        name = name.rsplit(".", 1)[0]

    if "pusht" in name:
        return {
            "agent_location": {
                "column": "state",
                "slice": [0, 1],
            },
            "block_location": {
                "column": "state",
                "slice": [2, 3],
            },
            "block_angle": {
                "column": "state",
                "slice": [4],
            },
        }
    if "tworoom" in name:
        return {
            "agent_position": {"column": "pos_agent"},
        }
    if "cube" in name:
        return {
            "joint_position": {"column": "qpos", "slice": "0:6"},
            "joint_velocity": {"column": "qvel", "slice": "0:6"},
            "end_effector_position": {
                "derive": "cube_end_effector_position",
                "columns": ["qpos", "qvel"],
            },
            "end_effector_yaw": {
                "derive": "cube_end_effector_yaw",
                "columns": ["qpos", "qvel"],
            },
            "gripper": {"derive": "cube_gripper", "columns": ["qpos"]},
            "block_position": {"column": "privileged_block_0_pos"},
            "block_quaternion": {"column": "privileged_block_0_quat"},
            "block_yaw": {
                "derive": "wxyz_quat_yaw",
                "columns": ["privileged_block_0_quat"],
            },
        }
    if "reacher" in name:
        return {
            "joint_position": {"column": "qpos"},
            "joint_velocity": {"column": "qvel"},
            "finger_position": {"column": "finger_pos"},
            "target_position": {"column": "target_pos"},
        }
    raise ValueError(
        "No default probe targets for dataset "
        f"{dataset_name!r}; set eval.probe_targets explicitly."
    )


def probe_target_specs(cfg):
    targets = cfg.eval.get("probe_targets")
    if targets is None:
        targets = default_probe_targets(cfg.eval.dataset_name)
    if OmegaConf.is_config(targets):
        targets = OmegaConf.to_container(targets, resolve=True)

    specs = {}
    for name, spec in targets.items():
        if isinstance(spec, str):
            spec = {"column": spec}
        if "column" not in spec and "derive" not in spec:
            raise ValueError(f"Probe target {name!r} must define a column or derive")
        target_slice = _slice_from_spec(spec.get("slice"))
        resolved = {
            "slice": target_slice,
            "derive": spec.get("derive"),
            "columns": [str(col) for col in spec.get("columns", [])],
            "metric_mean": _values_from_spec(spec.get("metric_mean"), target_slice),
            "metric_std": _values_from_spec(spec.get("metric_std"), target_slice),
        }
        if "column" in spec:
            resolved["column"] = str(spec["column"])
        specs[name] = resolved
    return specs


def probe_keys_to_load(cfg, available_columns):
    keys = ["pixels"]
    for spec in probe_target_specs(cfg).values():
        if "column" in spec:
            keys.append(spec["column"])
        keys.extend(spec.get("columns", []))
    seen = set()
    return [
        key
        for key in keys
        if key not in seen and not seen.add(key) and key in available_columns
    ]


def probe_target_columns(cfg):
    columns = []
    for spec in probe_target_specs(cfg).values():
        if "column" in spec:
            columns.append(spec["column"])
        columns.extend(spec.get("columns", []))
    return set(columns)


def get_probe_dataset(cfg, dataset_name):
    dataset_root = local_dataset_root(cfg)
    dataset_path = resolve_eval_dataset(dataset_name, dataset_root)

    if dataset_path and dataset_path.is_dir() and (dataset_path / NpyMemmapDataset.META_NAME).exists():
        with open(dataset_path / NpyMemmapDataset.META_NAME, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        keys = probe_keys_to_load(cfg, metadata["columns"])
        missing = sorted(probe_target_columns(cfg) - set(keys))
        if missing:
            raise KeyError(f"Missing probe columns in {dataset_path}: {missing}")
        return NpyMemmapDataset(dataset_path, keys_to_load=keys)

    h5_kwargs = {"keys_to_cache": [], "cache_dir": dataset_root}
    if dataset_path and dataset_path.suffix == ".h5":
        import h5py

        with h5py.File(dataset_path, "r") as f:
            available_columns = [k for k in f.keys() if k not in ("ep_len", "ep_offset")]
        keys = probe_keys_to_load(cfg, available_columns)
        missing = sorted(probe_target_columns(cfg) - set(keys))
        if missing:
            raise KeyError(f"Missing probe columns in {dataset_path}: {missing}")
        h5_kwargs["path"] = dataset_path
        h5_kwargs["keys_to_load"] = keys
        return swm.data.HDF5Dataset(**h5_kwargs)

    return swm.data.HDF5Dataset(dataset_name, **h5_kwargs)


def _num_rows(dataset, target_column):
    if isinstance(dataset, NpyMemmapDataset):
        return int(dataset.metadata["num_rows"])
    return int(len(dataset.get_col_data(target_column)))


def _imagenet_normalized_pixels(pixels, img_size, device):
    pixels = torch.as_tensor(pixels)
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    if pixels.shape[-1] in (1, 3):
        pixels = pixels.permute(0, 3, 1, 2)
    pixels = pixels.float()
    if pixels.max() > 2:
        pixels = pixels / 255.0

    if pixels.shape[-2:] != (img_size, img_size):
        pixels = torch.nn.functional.interpolate(
            pixels,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

    stats = spt.data.dataset_stats.ImageNet
    mean = torch.as_tensor(stats["mean"], dtype=pixels.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=pixels.dtype).view(1, -1, 1, 1)
    return ((pixels - mean) / std).to(device)


class CubeKinematics:
    def __init__(self):
        import mujoco
        from ogbench.manipspace import lie
        from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

        self.mujoco = mujoco
        self.lie = lie
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.env = CubeEnv(
                env_type="single",
                ob_type="states",
                terminate_at_goal=False,
            )
            self.env.reset()

    def batch(self, qpos, qvel=None):
        qpos = np.asarray(qpos)
        if qvel is None:
            qvel = np.zeros((len(qpos), self.env._model.nv), dtype=qpos.dtype)
        qvel = np.asarray(qvel)

        positions = np.empty((len(qpos), 3), dtype=np.float32)
        yaws = np.empty((len(qpos), 1), dtype=np.float32)
        for idx, (qpos_row, qvel_row) in enumerate(zip(qpos, qvel)):
            self.env.set_state(qpos_row, qvel_row)
            self.mujoco.mj_forward(self.env._model, self.env._data)
            positions[idx] = self.env._data.site_xpos[self.env._pinch_site_id]
            yaws[idx, 0] = self.lie.SO3.from_matrix(
                self.env._data.site_xmat[self.env._pinch_site_id].reshape(3, 3)
            ).compute_yaw_radians()
        return {
            "cube_end_effector_position": positions,
            "cube_end_effector_yaw": yaws,
        }


def _wxyz_quat_yaw(quat):
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = [quat[:, idx] for idx in range(4)]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw[:, None].astype(np.float32)


def _derived_target_array(rows, spec, derived_cache):
    derive = spec["derive"]
    if derive in ("cube_end_effector_position", "cube_end_effector_yaw"):
        if "cube_kinematics" not in derived_cache:
            derived_cache["cube_kinematics"] = CubeKinematics()
        cache_key = ("cube_kinematics_values", id(rows["qpos"]))
        if cache_key not in derived_cache:
            derived_cache[cache_key] = derived_cache["cube_kinematics"].batch(
                rows["qpos"],
                rows.get("qvel"),
            )
        return derived_cache[cache_key][derive]
    if derive == "cube_gripper":
        return np.clip(np.asarray(rows["qpos"])[:, 6:7] / 0.8, 0.0, 1.0).astype(
            np.float32
        )
    if derive == "wxyz_quat_yaw":
        return _wxyz_quat_yaw(rows["privileged_block_0_quat"])
    raise ValueError(f"Unknown derived probe target: {derive}")


def _target_array(rows, spec, derived_cache=None):
    if spec.get("derive"):
        if derived_cache is None:
            derived_cache = {}
        values = _derived_target_array(rows, spec, derived_cache)
        return values.astype(np.float32, copy=False)

    values = np.asarray(rows[spec["column"]])
    if values.ndim == 1:
        values = values[:, None]
    else:
        values = values[:, spec["slice"]]
        if values.ndim == 1:
            values = values[:, None]
    return values.astype(np.float32, copy=False)


def _pearsonr_per_dim(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    corr = []
    for dim in range(target.shape[1]):
        x = pred[:, dim]
        y = target[:, dim]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corr.append(np.nan)
        else:
            corr.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.nanmean(corr))


def _fit_probe(kind, cfg):
    if kind == "linear":
        alpha = float(cfg.eval.get("probe_ridge_alpha", 1e-3))
        return Ridge(alpha=alpha)
    if kind == "mlp":
        hidden = tuple(int(v) for v in cfg.eval.get("probe_mlp_hidden", [256, 256]))
        return MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=str(cfg.eval.get("probe_mlp_activation", "relu")),
            alpha=float(cfg.eval.get("probe_mlp_alpha", 1e-4)),
            batch_size=int(cfg.eval.get("probe_mlp_batch_size", 256)),
            learning_rate_init=float(cfg.eval.get("probe_mlp_lr", 1e-3)),
            max_iter=int(cfg.eval.get("probe_mlp_max_iter", 200)),
            early_stopping=bool(cfg.eval.get("probe_mlp_early_stopping", True)),
            random_state=int(cfg.seed),
            verbose=bool(cfg.eval.get("probe_mlp_verbose", False)),
        )
    raise ValueError(f"Unknown probe kind: {kind}")


def _metric_normalize(values, mean, std):
    return (values - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-12)


def _metric_stats(spec):
    mean = spec.get("metric_mean")
    std = spec.get("metric_std")
    if mean is None or std is None:
        raise ValueError("Probe metric stats were not resolved")
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _resolve_metric_stats(targets, labels, train_idx, mode):
    mode = str(mode).lower()
    if mode not in ("sample", "train"):
        raise ValueError("eval.probe_metric_stats must be 'sample' or 'train'")

    resolved = {}
    details = {}
    for name, spec in targets.items():
        spec = dict(spec)
        mean = spec.get("metric_mean")
        std = spec.get("metric_std")
        source = "target_spec"
        if mean is None or std is None:
            values = labels[name] if mode == "sample" else labels[name][train_idx]
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            source = mode

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        spec["metric_mean"] = mean
        spec["metric_std"] = std
        spec["metric_stats_source"] = source
        resolved[name] = spec
        details[name] = {
            "source": source,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return resolved, details


def _evaluate_probe(kind, x_train, y_train, x_test, y_test, spec, cfg):
    x_scaler = preprocessing.StandardScaler()
    y_scaler = preprocessing.StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train)
    x_test_s = x_scaler.transform(x_test)
    y_train_s = y_scaler.fit_transform(y_train)

    model = _fit_probe(kind, cfg)
    model.fit(x_train_s, y_train_s)
    pred = np.asarray(model.predict(x_test_s))
    if pred.ndim == 1:
        pred = pred[:, None]
    pred_fit_s = pred
    pred = y_scaler.inverse_transform(pred_fit_s)
    y_test_fit_s = y_scaler.transform(y_test)
    metric_mean, metric_std = _metric_stats(spec)
    pred_metric = _metric_normalize(pred, metric_mean, metric_std)
    y_test_metric = _metric_normalize(y_test, metric_mean, metric_std)
    return {
        "mse": float(mean_squared_error(y_test_metric, pred_metric)),
        "mse_normalized": float(mean_squared_error(y_test_metric, pred_metric)),
        "mse_raw": float(mean_squared_error(y_test, pred)),
        "mse_fit_standardized": float(mean_squared_error(y_test_fit_s, pred_fit_s)),
        "pearson_r": _pearsonr_per_dim(pred, y_test),
        "mse_space": "target_normalized",
    }


def _cfg_path(value):
    if value is None:
        return None
    return Path(str(value))


def _external_probe_labels_path(cfg):
    path = cfg.eval.get("probe_labels", None)
    if path is None:
        path = cfg.eval.get("labels", None)
    return _cfg_path(path)


def _external_probe_specs_path(cfg, labels_path):
    path = cfg.eval.get("probe_target_specs", None)
    if path is None:
        path = cfg.eval.get("target_specs", None)
    if path is not None:
        return _cfg_path(path)
    return labels_path.with_name("target_specs.json")


def _load_external_labels(cfg):
    labels_path = _external_probe_labels_path(cfg)
    if labels_path is None:
        raise ValueError("External probe labels path is not configured")
    if labels_path.is_dir():
        labels_path = labels_path / "labels.npz"
    if labels_path.suffix.lower() == ".csv":
        return labels_path, _load_csv_probe_labels(labels_path)
    data = np.load(labels_path)
    return labels_path, {key: data[key] for key in data.files}


def _load_external_target_specs(cfg, labels_path, labels):
    if labels_path.suffix.lower() == ".csv":
        return _csv_probe_target_specs(labels)

    specs_path = _external_probe_specs_path(cfg, labels_path)
    if specs_path.exists():
        with specs_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("targets", payload)

    specs = {}
    for key, value in labels.items():
        if key in {"row_idx", "episode_idx", "frame_local_idx"}:
            continue
        kind = "classification" if np.issubdtype(value.dtype, np.integer) else "regression"
        specs[key] = {"kind": kind, "group": "inferred"}
    return specs


def _load_csv_probe_labels(path):
    """Convert the human-readable Bridge preview CSV into probe target arrays."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Probe label CSV is empty: {path}")

    required_columns = ("row_idx", "episode_idx", "frame_local_idx", "gripper_xy")
    missing = [column for column in required_columns if column not in rows[0]]
    if missing:
        raise KeyError(f"Probe label CSV is missing required columns: {missing}")

    labels = {
        "row_idx": csv_int_column(rows, "row_idx"),
        "episode_idx": csv_int_column(rows, "episode_idx"),
        "frame_local_idx": csv_int_column(rows, "frame_local_idx"),
    }
    for column in rows[0]:
        match = CSV_PROBE_TARGET.match(column)
        if not match:
            continue
        object_name, field = match.groups()
        target_name, values = csv_probe_target_column(rows, object_name, field)
        labels[target_name] = values
    selected = select_csv_probe_objects(labels)
    print(
        f"[probe] retained {len(selected['row_idx'])}/{len(rows)} CSV rows with "
        "one valid moving, static, and background object"
    )
    return selected


def select_csv_probe_objects(labels):
    """Select the first valid object slot in each group and discard incomplete rows."""
    metadata_keys = ("row_idx", "episode_idx", "frame_local_idx")
    selected = {key: np.asarray(labels[key]) for key in metadata_keys}
    selected["gripper_centroid"] = np.asarray(labels["gripper_centroid"])
    row_valid = np.ones(len(selected["row_idx"]), dtype=bool)

    for group in ("moving", "static", "background"):
        slot_prefixes = csv_object_slot_prefixes(labels, group)
        if not slot_prefixes:
            raise KeyError(f"CSV probe labels contain no {group} object slots")
        group_values, group_valid = select_first_valid_slot(labels, slot_prefixes)
        selected.update({f"{group}_{field}": value for field, value in group_values.items()})
        row_valid &= group_valid

    return {name: values[row_valid] for name, values in selected.items()}


def csv_object_slot_prefixes(labels, group):
    slots = []
    for name in labels:
        match = OBJECT_SLOT_TARGET.match(name)
        if match and match.group(1) == group:
            slots.append((int(match.group(2)), name.removesuffix("_centroid")))
    return [prefix for _, prefix in sorted(slots)]


def select_first_valid_slot(labels, slot_prefixes):
    num_rows = len(labels["row_idx"])
    selected = {
        "centroid": np.full((num_rows, 2), np.nan, dtype=np.float32),
        "color_rgb": np.full((num_rows, 3), np.nan, dtype=np.float32),
        "category": np.full(num_rows, -1, dtype=np.int64),
    }
    found = np.zeros(num_rows, dtype=bool)
    for prefix in slot_prefixes:
        centroid = np.asarray(labels[f"{prefix}_centroid"], dtype=np.float32)
        color_rgb = np.asarray(labels[f"{prefix}_color_rgb"], dtype=np.float32)
        category = np.asarray(labels[f"{prefix}_category"], dtype=np.int64)
        valid = (
            np.isfinite(centroid).all(axis=1)
            & np.isfinite(color_rgb).all(axis=1)
            & (category >= 0)
        )
        take = valid & ~found
        selected["centroid"][take] = centroid[take]
        selected["color_rgb"][take] = color_rgb[take]
        selected["category"][take] = category[take]
        found |= valid
    return selected, found


def csv_probe_target_column(rows, object_name, field):
    prefix = object_name.replace("_", "")
    group = "background" if object_name.startswith("background_") else "foreground"
    if object_name == "gripper":
        group = "robot"

    if field == "xy":
        return f"{prefix}_centroid", csv_vector_column(rows, f"{object_name}_{field}", 2)
    if field == "color_rgb":
        return f"{prefix}_color_rgb", csv_vector_column(rows, f"{object_name}_{field}", 3)
    return f"{prefix}_category", csv_category_column(
        rows,
        f"{object_name}_{field}",
        BACKGROUND_CATEGORY_IDS if group == "background" else FOREGROUND_CATEGORY_IDS,
    )


def csv_int_column(rows, column):
    return np.asarray([int(row[column]) for row in rows], dtype=np.int64)


def csv_vector_column(rows, column, width):
    values = np.full((len(rows), width), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        raw = row.get(column, "").strip()
        if not raw:
            continue
        vector = np.asarray(json.loads(raw), dtype=np.float32).reshape(-1)
        if vector.size != width:
            raise ValueError(
                f"Expected {width} values in {column!r} at CSV row {index}, got {vector.size}"
            )
        values[index] = vector
    return values


def csv_category_column(rows, column, category_to_id):
    values = np.full(len(rows), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        category = row.get(column, "").strip().lower()
        if not category:
            continue
        if category not in category_to_id:
            raise ValueError(f"Unsupported category {category!r} in {column!r}")
        values[index] = category_to_id[category]
    return values


def _csv_probe_target_specs(labels):
    specs = {}
    for name, values in labels.items():
        if name in {"row_idx", "episode_idx", "frame_local_idx"}:
            continue
        if name.startswith("gripper"):
            group = "robot"
        elif name.startswith("moving"):
            group = "dynamic"
        elif name.startswith("static"):
            group = "static"
        elif name.startswith("background"):
            group = "background"
        else:
            group = "inferred"
        specs[name] = {
            "kind": "classification" if name.endswith("_category") else "regression",
            "group": group,
        }
    return specs


def _external_probe_dataset(cfg):
    dataset_name = cfg.eval.get("probe_dataset", None)
    if dataset_name is None:
        dataset_name = cfg.dataset.get("name", None)
    if dataset_name is None:
        dataset_name = cfg.eval.get("dataset_name", None)
    dataset_root = local_dataset_root(cfg)
    dataset_path = resolve_eval_dataset(str(dataset_name), dataset_root)
    if dataset_path is None:
        dataset_path = Path(str(dataset_name))
    return NpyMemmapDataset(dataset_path, keys_to_load=["pixels"])


def _extract_feature_sets(cfg, model, dataset, labels, device):
    row_idx = np.asarray(labels["row_idx"], dtype=np.int64)
    batch_size = int(cfg.eval.get("probe_batch_size", cfg.eval.get("batch_size", 128)))
    img_size = int(cfg.eval.img_size)
    z_parts = []
    u_parts = []

    for start in range(0, len(row_idx), batch_size):
        batch_rows = row_idx[start : start + batch_size]
        rows = dataset.get_row_data(batch_rows)
        pixels = _imagenet_normalized_pixels(rows["pixels"], img_size, device)
        with torch.no_grad():
            output = model.encode({"pixels": pixels.unsqueeze(1)}, return_u=True)
        z_parts.append(output["z"][:, 0].detach().cpu().float().numpy())
        if "u" in output:
            u = output["u"][:, 0]
            if str(cfg.eval.get("probe_u_reduction", cfg.eval.get("u_reduction", "flatten"))) == "mean":
                u = u.mean(dim=1)
            else:
                u = u.reshape(u.size(0), -1)
            u_parts.append(u.detach().cpu().float().numpy())

        print(
            f"[probe] encoded {min(start + batch_size, len(row_idx))}/"
            f"{len(row_idx)} externally labeled Bridge frames"
        )

    features = {"z": np.concatenate(z_parts, axis=0)}
    if u_parts:
        features["u"] = np.concatenate(u_parts, axis=0)
        features["zu"] = np.concatenate([features["z"], features["u"]], axis=1)
    return features


def _episode_split(episode_idx, train_fraction, seed):
    unique = np.unique(episode_idx)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(unique)
    train_count = max(1, min(len(unique) - 1, int(round(len(unique) * train_fraction))))
    train_eps = set(int(v) for v in perm[:train_count])
    train = np.array([int(ep) in train_eps for ep in episode_idx], dtype=bool)
    return train, ~train


def _target_kind(spec, values):
    kind = str(spec.get("kind", "")).lower()
    if kind in {"classification", "class"}:
        return "classification"
    if kind in {"regression", "continuous"}:
        return "regression"
    return "classification" if np.issubdtype(values.dtype, np.integer) else "regression"


def _external_target_values(labels, name, kind):
    values = np.asarray(labels[name])
    if kind == "classification":
        return values.reshape(-1).astype(np.int64)
    if values.ndim == 1:
        values = values[:, None]
    return values.astype(np.float32, copy=False)


def _finite_external_target_mask(y, kind):
    if kind == "classification":
        return np.asarray(y).reshape(-1) >= 0
    if y.ndim == 1:
        y = y[:, None]
    return np.isfinite(y).all(axis=1)


def _fit_external_classifier(kind, cfg):
    if kind == "linear":
        return LogisticRegression(
            max_iter=int(cfg.eval.get("probe_logreg_max_iter", 1000)),
            class_weight="balanced",
            random_state=int(cfg.seed),
        )
    if kind == "mlp":
        hidden = tuple(int(v) for v in cfg.eval.get("probe_mlp_hidden", [256, 256]))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            batch_size=int(cfg.eval.get("probe_mlp_batch_size", 256)),
            learning_rate_init=float(cfg.eval.get("probe_mlp_lr", 1e-3)),
            max_iter=int(cfg.eval.get("probe_mlp_max_iter", 200)),
            early_stopping=bool(cfg.eval.get("probe_mlp_early_stopping", True)),
            random_state=int(cfg.seed),
        )
    raise ValueError(f"Unknown probe kind: {kind}")


def _evaluate_external_classification(kind, x_train, y_train, x_test, y_test, cfg):
    x_scaler = preprocessing.StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train)
    x_test_s = x_scaler.transform(x_test)
    model = _fit_external_classifier(kind, cfg)
    model.fit(x_train_s, y_train)
    pred = model.predict(x_test_s)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "num_classes_train": int(len(np.unique(y_train))),
        "num_classes_test": int(len(np.unique(y_test))),
    }


def _evaluate_external_regression(kind, x_train, y_train, x_test, y_test, cfg):
    spec = {
        "metric_mean": y_train.mean(axis=0),
        "metric_std": y_train.std(axis=0),
    }
    return _evaluate_probe(kind, x_train, y_train, x_test, y_test, spec, cfg)


def run_external_label_probe_eval(cfg):
    policy_path = Path(cfg.policy)
    if cfg.policy == "random" or not policy_path.exists():
        raise ValueError(
            "External-label probe evaluation requires a local checkpoint path "
            f"in cfg.policy; got {cfg.policy!r}"
        )

    labels_path, labels = _load_external_labels(cfg)
    specs = _load_external_target_specs(cfg, labels_path, labels)
    model, results_path, _ = load_local_policy_model(policy_path)
    device = torch.device(cfg.solver.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model.requires_grad_(False)

    dataset = _external_probe_dataset(cfg)
    start_time = time.time()
    features = _extract_feature_sets(cfg, model, dataset, labels, device)
    train_base, test_base = _episode_split(
        labels["episode_idx"],
        float(cfg.eval.get("probe_train_fraction", cfg.eval.get("train_fraction", 0.8))),
        int(cfg.seed),
    )

    feature_sets = list(cfg.eval.get("probe_feature_sets", cfg.eval.get("feature_sets", ["z"])))
    probe_kinds = list(cfg.eval.get("probe_kinds", ["linear", "mlp"]))
    min_train = int(cfg.eval.get("probe_min_train", cfg.eval.get("min_train", 20)))
    min_test = int(cfg.eval.get("probe_min_test", cfg.eval.get("min_test", 10)))

    metrics = {
        "probe_mode": "external_labels",
        "labels": str(labels_path),
        "num_frames": int(len(labels["row_idx"])),
        "num_episodes": int(len(np.unique(labels["episode_idx"]))),
        "feature_dims": {name: int(value.shape[1]) for name, value in features.items()},
        "targets": {},
    }

    for target_name, spec in specs.items():
        if target_name not in labels:
            continue
        kind = _target_kind(spec, np.asarray(labels[target_name]))
        y = _external_target_values(labels, target_name, kind)
        valid = _finite_external_target_mask(y, kind)
        train_mask = valid & train_base
        test_mask = valid & test_base
        target_metrics = {
            "kind": kind,
            "group": spec.get("group", "unknown"),
            "num_valid": int(valid.sum()),
            "num_train": int(train_mask.sum()),
            "num_test": int(test_mask.sum()),
            "features": {},
        }
        if train_mask.sum() < min_train or test_mask.sum() < min_test:
            target_metrics["skipped"] = "not enough valid train/test examples"
            metrics["targets"][target_name] = target_metrics
            continue
        if kind == "classification" and len(np.unique(y[train_mask])) < 2:
            target_metrics["skipped"] = "fewer than two training classes"
            metrics["targets"][target_name] = target_metrics
            continue

        for feature_name in feature_sets:
            if feature_name not in features:
                continue
            x = features[feature_name]
            feature_metrics = {}
            for probe_kind in probe_kinds:
                if kind == "classification":
                    feature_metrics[probe_kind] = _evaluate_external_classification(
                        probe_kind,
                        x[train_mask],
                        y[train_mask],
                        x[test_mask],
                        y[test_mask],
                        cfg,
                    )
                    score = feature_metrics[probe_kind]["accuracy"]
                    print(
                        f"[probe] {target_name}/{feature_name}/{probe_kind}: "
                        f"acc={score:.4f}"
                    )
                else:
                    feature_metrics[probe_kind] = _evaluate_external_regression(
                        probe_kind,
                        x[train_mask],
                        y[train_mask],
                        x[test_mask],
                        y[test_mask],
                        cfg,
                    )
                    score = feature_metrics[probe_kind]["mse"]
                    corr = feature_metrics[probe_kind]["pearson_r"]
                    print(
                        f"[probe] {target_name}/{feature_name}/{probe_kind}: "
                        f"mse={score:.6g} r={corr:.4f}"
                    )
            target_metrics["features"][feature_name] = feature_metrics
        metrics["targets"][target_name] = target_metrics

    metrics["evaluation_time"] = time.time() - start_time
    print(json.dumps(metrics, indent=2))

    output_path, json_path = write_eval_results(
        eval_output_dir(results_path, cfg),
        cfg.eval.get("probe_output_filename", cfg.output.filename),
        cfg,
        metrics,
        metrics["evaluation_time"],
    )
    print(f"Saved external-label probe metrics to {output_path}")
    print(f"Saved external-label probe details to {json_path}")
    return metrics


def run_probe_eval(cfg):
    if _external_probe_labels_path(cfg) is not None:
        return run_external_label_probe_eval(cfg)

    policy_path = Path(cfg.policy)
    if cfg.policy == "random" or not policy_path.exists():
        raise ValueError(
            "Probe evaluation requires a local LRC-JEPA checkpoint path in cfg.policy; "
            f"got {cfg.policy!r}"
        )

    model, results_path, _ = load_local_policy_model(policy_path)
    device = torch.device(cfg.solver.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model.requires_grad_(False)

    dataset = get_probe_dataset(cfg, cfg.eval.dataset_name)
    targets = probe_target_specs(cfg)
    first_target = next(iter(probe_target_columns(cfg)))
    num_rows = _num_rows(dataset, first_target)
    num_samples = min(int(cfg.eval.get("probe_samples", 10000)), num_rows)
    batch_size = int(cfg.eval.get("probe_batch_size", 128))
    rng = np.random.default_rng(int(cfg.seed))
    sample_indices = np.sort(rng.choice(num_rows, size=num_samples, replace=False))

    feature_parts = {"z": [], "u": []}
    labels = {name: [] for name in targets}
    start_time = time.time()

    for start in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[start : start + batch_size]
        rows = dataset.get_row_data(batch_indices)
        pixels = _imagenet_normalized_pixels(rows["pixels"], int(cfg.eval.img_size), device)
        with torch.no_grad():
            encoded = model.encode({"pixels": pixels.unsqueeze(1)}, return_u=True)
        feature_parts["z"].append(encoded["z"][:, 0].detach().cpu().float().numpy())
        if "u" in encoded:
            u = encoded["u"][:, 0]
            if str(cfg.eval.get("probe_u_reduction", "flatten")) == "mean":
                u = u.mean(dim=1)
            else:
                u = u.reshape(u.size(0), -1)
            feature_parts["u"].append(u.detach().cpu().float().numpy())
        derived_cache = {}
        for name, spec in targets.items():
            labels[name].append(_target_array(rows, spec, derived_cache))

        print(
            f"[probe] encoded {min(start + batch_size, len(sample_indices))}/"
            f"{len(sample_indices)} frames"
        )

    features = {"z": np.concatenate(feature_parts["z"], axis=0)}
    if feature_parts["u"]:
        features["u"] = np.concatenate(feature_parts["u"], axis=0)
        features["zu"] = np.concatenate([features["z"], features["u"]], axis=1)
    feature_sets = list(cfg.eval.get("probe_feature_sets", ["z"]))
    unavailable = [name for name in feature_sets if name not in features]
    if unavailable:
        raise ValueError(
            f"Requested unavailable probe feature sets: {unavailable}; "
            f"available: {list(features)}"
        )
    labels = {name: np.concatenate(parts, axis=0) for name, parts in labels.items()}

    all_labels = np.concatenate(list(labels.values()), axis=1)
    valid = np.isfinite(all_labels).all(axis=1)
    for name in feature_sets:
        valid &= np.isfinite(features[name]).all(axis=1)
    features = {name: value[valid] for name, value in features.items()}
    labels = {name: value[valid] for name, value in labels.items()}
    if len(features["z"]) < 2:
        raise ValueError("Probe sample produced fewer than two finite examples")

    train_fraction = float(cfg.eval.get("probe_train_fraction", 0.8))
    train_count = max(
        1,
        min(len(features["z"]) - 1, int(round(len(features["z"]) * train_fraction))),
    )
    perm = rng.permutation(len(features["z"]))
    train_idx = perm[:train_count]
    test_idx = perm[train_count:]
    targets, metric_stats = _resolve_metric_stats(
        targets,
        labels,
        train_idx,
        cfg.eval.get("probe_metric_stats", "sample"),
    )

    metrics = {
        "num_samples": int(len(features["z"])),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "feature_dims": {name: int(value.shape[1]) for name, value in features.items()},
        "metric_stats_mode": str(cfg.eval.get("probe_metric_stats", "sample")),
        "metric_stats": metric_stats,
        "targets": {},
    }

    kinds = list(cfg.eval.get("probe_kinds", ["linear", "mlp"]))
    for target_name, y in labels.items():
        target_metrics = {"features": {}}
        for feature_name in feature_sets:
            feature_metrics = {}
            for kind in kinds:
                feature_metrics[kind] = _evaluate_probe(
                    kind,
                    features[feature_name][train_idx],
                    y[train_idx],
                    features[feature_name][test_idx],
                    y[test_idx],
                    targets[target_name],
                    cfg,
                )
                print(
                    f"[probe] {target_name}/{feature_name}/{kind}: "
                    f"mse={feature_metrics[kind]['mse']:.6g} "
                    f"raw_mse={feature_metrics[kind]['mse_raw']:.6g} "
                    f"r={feature_metrics[kind]['pearson_r']:.4f}"
                )
            target_metrics["features"][feature_name] = feature_metrics
        metrics["targets"][target_name] = target_metrics

    metrics["evaluation_time"] = time.time() - start_time
    print(json.dumps(metrics, indent=2))

    output_path, json_path = write_eval_results(
        eval_output_dir(results_path, cfg),
        cfg.output.filename,
        cfg,
        metrics,
        metrics["evaluation_time"],
    )
    print(f"Saved probe metrics to {output_path}")
    print(f"Saved probe details to {json_path}")
    return metrics


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    if cfg.eval.dataset_name == "bridge_v2_clean" and not cfg.eval.get("probe_labels"):
        raise ValueError("Bridge-v2 probing requires eval.probe_labels=/path/to/labels.csv (or labels.npz).")
    run_probe_eval(cfg)


if __name__ == "__main__":
    run()
