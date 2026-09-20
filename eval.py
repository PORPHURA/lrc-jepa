from runtime import configure_runtime
configure_runtime()

import os
import json
import time
import re
from pathlib import Path
import hydra
import numpy as np

import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from compat import patch_dm_control_missing_mujoco_fields
from utils import NpyMemmapDataset, prepare_training_dataset


patch_dm_control_missing_mujoco_fields()


def local_dataset_root(cfg):
    return Path(os.environ.get("LOCAL_DATASET_DIR") or cfg.get("cache_dir") or "data")


def resolve_eval_dataset(dataset_name, dataset_root):
    candidates = [
        dataset_root / dataset_name,
        dataset_root / f"{dataset_name}.npy",
        dataset_root / f"{dataset_name}.h5",
        dataset_root / "datasets" / dataset_name,
        dataset_root / "datasets" / f"{dataset_name}.npy",
        dataset_root / "datasets" / f"{dataset_name}.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def callable_dataset_keys(callables):
    keys = []
    for spec in callables or []:
        for data in spec.get("args", {}).values():
            if not data.get("in_dataset", True):
                continue
            key = data.get("value")
            if not isinstance(key, str):
                continue
            if key.startswith("goal_"):
                key = key[len("goal_") :]
            keys.append(key)
    return keys


def eval_keys_to_load(cfg, available_columns):
    requested = ["pixels", "action", *cfg.dataset.keys_to_cache]
    requested.extend(callable_dataset_keys(cfg.eval.get("callables")))

    if "episode_idx" in available_columns:
        requested.append("episode_idx")
    if "ep_idx" in available_columns:
        requested.append("ep_idx")
    if "step_idx" in available_columns:
        requested.append("step_idx")

    seen = set()
    keys = []
    for key in requested:
        if key in seen or key not in available_columns:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def episode_column_name(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def episode_values_for_rows(dataset, episode_ids, values):
    row_episode_ids = dataset.get_col_data(episode_column_name(dataset))
    episode_ids = np.asarray(episode_ids)
    values = np.asarray(values)
    if episode_ids.size == 0:
        raise ValueError("Cannot map row episode ids from an empty episode list")

    if (
        np.issubdtype(row_episode_ids.dtype, np.integer)
        and episode_ids.size == values.size
        and episode_ids.size > 0
        and episode_ids[0] == 0
        and episode_ids[-1] == episode_ids.size - 1
    ):
        return values[row_episode_ids]

    order = np.argsort(episode_ids)
    sorted_ids = episode_ids[order]
    positions = np.searchsorted(sorted_ids, row_episode_ids)
    missing = positions >= len(sorted_ids)
    safe_positions = np.minimum(positions, len(sorted_ids) - 1)
    if np.any(missing) or np.any(sorted_ids[safe_positions] != row_episode_ids):
        raise ValueError("Dataset rows reference episode ids that were not sampled")
    return values[order][positions]


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    episodes = np.asarray(episodes)
    lengths = np.asarray(dataset.lengths)
    if episodes.size == 0:
        return np.array([], dtype=lengths.dtype)
    if (
        not np.issubdtype(episodes.dtype, np.integer)
        or episodes.min() < 0
        or episodes.max() >= len(lengths)
    ):
        episode_idx = dataset.get_col_data(episode_column_name(dataset))
        step_idx = dataset.get_col_data("step_idx")
        order = np.argsort(episode_idx)
        sorted_episode_idx = episode_idx[order]
        sorted_step_idx = step_idx[order]
        unique_episodes, starts = np.unique(sorted_episode_idx, return_index=True)
        max_steps = np.maximum.reduceat(sorted_step_idx, starts) + 1
        positions = np.searchsorted(unique_episodes, episodes)
        missing = positions >= len(unique_episodes)
        safe_positions = np.minimum(positions, len(unique_episodes) - 1)
        if np.any(missing) or np.any(unique_episodes[safe_positions] != episodes):
            raise ValueError("Requested episode id not found in dataset")
        return max_steps[positions]

    return lengths[episodes]


def get_dataset(cfg, dataset_name):
    dataset_root = local_dataset_root(cfg)
    dataset_path = resolve_eval_dataset(dataset_name, dataset_root)

    if dataset_path and dataset_path.is_dir() and (dataset_path / NpyMemmapDataset.META_NAME).exists():
        import json

        with open(dataset_path / NpyMemmapDataset.META_NAME, "r", encoding="utf-8") as f:
            available_columns = json.load(f)["columns"]
        cache_keys = [
            key
            for key in dict.fromkeys([*cfg.dataset.keys_to_cache, "episode_idx", "ep_idx", "step_idx"])
            if key in available_columns
        ]
        return NpyMemmapDataset(
            dataset_path,
            keys_to_load=eval_keys_to_load(cfg, available_columns),
            keys_to_cache=cache_keys,
        )

    h5_kwargs = {
        "keys_to_cache": cfg.dataset.keys_to_cache,
        "cache_dir": dataset_root,
    }
    if dataset_path and dataset_path.suffix == ".h5":
        import h5py

        with h5py.File(dataset_path, "r") as f:
            available_columns = [k for k in f.keys() if k not in ("ep_len", "ep_offset")]
        h5_kwargs["path"] = dataset_path
        h5_kwargs["keys_to_load"] = eval_keys_to_load(cfg, available_columns)
        return swm.data.HDF5Dataset(**h5_kwargs)

    return swm.data.HDF5Dataset(dataset_name, **h5_kwargs)


def _epoch_from_weights_path(path):
    match = re.search(r"weights_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _find_run_config(path):
    candidates = []
    if path.is_dir():
        candidates.append(path / "config.yaml")
        candidates.append(path / "config.json")
    else:
        candidates.extend(
            [
                path.parent / "config.yaml",
                path.parent / "config.json",
                path.parent.parent / "config.yaml",
                path.parent.parent / "config.json",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find config.yaml or config.json near {path}")


def _find_weights_path(path):
    if path.is_file():
        return path

    weights = sorted(path.glob("weights_epoch_*.pt"), key=_epoch_from_weights_path)
    if weights:
        return weights[-1]

    ckpt_path = path / "checkpoints" / "last.ckpt"
    if ckpt_path.exists():
        return ckpt_path

    weights_path = path / "weights.pt"
    if weights_path.exists():
        return weights_path

    raise FileNotFoundError(
        f"Could not find weights_epoch_*.pt, checkpoints/last.ckpt, or weights.pt in {path}"
    )


def _model_config(train_cfg):
    if "model" in train_cfg:
        return train_cfg.model
    if "_target_" in train_cfg:
        return train_cfg
    raise KeyError("Run config must contain either a model section or a top-level _target_")


def _model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise TypeError("Checkpoint did not contain a state dict")

    if state and all(key.startswith("model.") for key in state):
        state = {key[len("model.") :]: value for key, value in state.items()}
    return state


def _compatible_state_dict(state, model):
    model_keys = model.state_dict().keys()
    if (
        any(key.startswith("encoder.encoder.layer.") for key in state)
        and any(key.startswith("encoder.layers.") for key in model_keys)
    ):
        replacements = (
            ("attention.attention.query.", "attention.q_proj."),
            ("attention.attention.key.", "attention.k_proj."),
            ("attention.attention.value.", "attention.v_proj."),
            ("attention.output.dense.", "attention.o_proj."),
            ("intermediate.dense.", "mlp.fc1."),
            ("output.dense.", "mlp.fc2."),
        )
        translated = {}
        for key, value in state.items():
            new_key = key.replace("encoder.encoder.layer.", "encoder.layers.")
            for old, new in replacements:
                new_key = new_key.replace(old, new)
            translated[new_key] = value
        return translated

    return state


def load_policy_model(policy_path):
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model, results_path, _ = load_local_policy_model(path)
    return model, results_path


def load_local_policy_model(path):
    path = Path(path)
    config_path = _find_run_config(path)
    weights_path = _find_weights_path(path)
    train_cfg = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(_model_config(train_cfg))
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = _compatible_state_dict(_model_state_dict(checkpoint), model)
    model.load_state_dict(state, strict=True)
    print(f"Loaded local policy model from {weights_path}")
    return model, weights_path.parent, train_cfg


def tensor_batch(item, device):
    return {k: v.unsqueeze(0).to(device) for k, v in item.items() if torch.is_tensor(v)}


def cost_actions(model, info, action_candidates):
    cached = {k: info[k] for k in ("goal_emb", "init_emb") if k in info}
    model_info = {
        k: v.clone() if torch.is_tensor(v) and k not in cached else v
        for k, v in info.items()
    }
    device = next(model.parameters()).device
    action_candidates = action_candidates.to(device)
    with torch.inference_mode():
        costs = model.get_cost(model_info, action_candidates).detach().cpu()

    for key in ("goal_emb", "init_emb"):
        if key not in info and key in model_info:
            info[key] = model_info[key].detach()
    return costs


def latent_distance(model, pixels, goal):
    device = next(model.parameters()).device
    with torch.no_grad():
        current = model.encode({"pixels": pixels[:, -1:].to(device)})["emb"][:, -1]
        target = model.encode({"pixels": goal[:, -1:].to(device)})["emb"][:, -1]
        return (current - target).pow(2).sum(dim=-1).detach().cpu()


def teacher_forced_one_step_loss(model, pixels, actions, history_size):
    device = next(model.parameters()).device
    with torch.no_grad():
        info = model.encode(
            {
                "pixels": pixels.unsqueeze(0).to(device),
                "action": actions.unsqueeze(0).to(device),
            }
        )
        pred = model.predict(
            info["z"][:, :history_size],
            info["act_emb"][:, :history_size],
        )
        target = info["z"][:, 1 : 1 + pred.size(1)]
        mse = (pred - target).pow(2)
        return {
            "teacher_forced_pred_mse": float(mse.mean().detach().cpu()),
            "teacher_forced_pred_l2": float(mse.sum(dim=-1).mean().detach().cpu()),
        }


def prepend_history_actions(history_actions, future_actions):
    """Attach recorded history actions to every candidate future plan.

    ``future_actions`` has shape ``(B, S, T_future, A)`` and the returned
    sequence is suitable for ``JEPA.rollout``.
    """
    if history_actions.ndim != 2 or future_actions.ndim != 4:
        raise ValueError("Expected history actions (T, A) and candidates (B, S, T, A)")
    if history_actions.shape[-1] != future_actions.shape[-1]:
        raise ValueError("History and candidate action dimensions must match")

    history = history_actions.to(
        device=future_actions.device,
        dtype=future_actions.dtype,
    )
    history = history.unsqueeze(0).unsqueeze(0).expand(
        future_actions.shape[0], future_actions.shape[1], -1, -1
    )
    return torch.cat([history, future_actions], dim=2)


def shuffle_future_actions(actions):
    """Shuffle future action order while guaranteeing a non-identity order."""
    if actions.size(0) < 2:
        return actions.clone()
    permutation = torch.randperm(actions.size(0), device=actions.device)
    if torch.equal(permutation, torch.arange(actions.size(0), device=actions.device)):
        permutation = torch.roll(permutation, shifts=1)
    return actions[permutation]


def cem_latent_plan(model, info, action_shape, cfg, history_actions=None):
    device = next(model.parameters()).device
    num_samples = int(cfg.solver.num_samples)
    topk = int(cfg.solver.topk)
    n_steps = int(cfg.solver.n_steps)
    var_scale = float(cfg.solver.var_scale)
    std_floor = float(cfg.eval.get("cem_std_floor", 0.05))

    if topk > num_samples:
        raise ValueError(f"solver.topk={topk} must be <= solver.num_samples={num_samples}")

    mean = torch.zeros(action_shape, device=device)
    std = torch.full(action_shape, var_scale, device=device)
    best_cost = None
    best_action = None

    for _ in range(n_steps):
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            num_samples, *action_shape, device=device
        )
        candidates = samples.unsqueeze(0)
        if history_actions is not None:
            candidates = prepend_history_actions(history_actions, candidates)
        costs = cost_actions(model, info, candidates)[0].to(device)
        elite_costs, elite_idx = torch.topk(costs, k=topk, largest=False)
        elites = samples[elite_idx]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0, unbiased=False).clamp_min(std_floor)

        if best_cost is None or elite_costs[0] < best_cost:
            best_cost = elite_costs[0].detach()
            best_action = elites[0].detach()

    return best_cost.cpu(), best_action.cpu()


def mean_metrics(rows):
    skip = {"eval_idx", "episode", "start"}
    keys = rows[0].keys()
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
        if key not in skip
        if isinstance(rows[0][key], (int, float, np.floating))
    }


def eval_output_dir(results_path, cfg):
    dataset_label = str(cfg.eval.get("dataset_name", "eval")).replace("\\", "/").rstrip("/")
    dataset_label = dataset_label.split("/")[-1]
    return Path(cfg.output.get("directory", "results")) / dataset_label


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def success_stats(metrics):
    successes = metrics.get("episode_successes")
    if successes is None:
        return {}
    successes = np.asarray(successes, dtype=np.float32)
    if successes.size == 0:
        return {}
    return {
        "success_rate_std": float(successes.std(ddof=0) * 100.0),
        "num_eval": int(successes.size),
    }


def write_eval_results(output_dir, filename, cfg, metrics, elapsed, episodes=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    json_path = output_path.with_suffix(".json")
    clean_metrics = to_jsonable(metrics)
    clean_metrics["evaluation_time"] = float(elapsed)

    with output_path.open("a", encoding="utf-8") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        for key, value in clean_metrics.items():
            f.write(f"{key}: {value}\n")

    payload = {"metrics": clean_metrics}
    if episodes is not None:
        payload["episodes"] = to_jsonable(episodes)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path, json_path


def run_bridge_offline_eval(cfg):
    policy_path = Path(cfg.policy)
    if cfg.policy == "random" or not policy_path.exists():
        raise ValueError(
            "Bridge offline latent eval requires a local checkpoint path in cfg.policy; "
            f"got {cfg.policy!r}"
        )

    model, results_path, train_cfg = load_local_policy_model(policy_path)
    device = torch.device(cfg.solver.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    dataset_cfg = OmegaConf.to_container(train_cfg.data.dataset, resolve=True)
    dataset, _ = prepare_training_dataset(
        dataset_cfg,
        img_size=train_cfg.img_size,
        cache_dir=str(local_dataset_root(cfg)),
        resize_images=train_cfg.preprocess.resize_images,
    )

    history_size = int(train_cfg.history_size)
    action_block = int(train_cfg.data.dataset.frameskip)
    if cfg.eval.goal_offset_steps % action_block != 0:
        raise ValueError("eval.goal_offset_steps must be divisible by the training frameskip")

    future_tokens = int(cfg.eval.goal_offset_steps // action_block)
    total_action_tokens = history_size + future_tokens - 1
    future_action_tokens = total_action_tokens - history_size
    span = (total_action_tokens + 1) * action_block
    action_dim = int(train_cfg.model.action_encoder.input_dim)
    recall_ks = list(cfg.eval.get("recall_ks", [1, 5, 10]))

    valid_episodes = np.flatnonzero(np.asarray(dataset.lengths) >= span)
    if len(valid_episodes) < cfg.eval.num_eval:
        raise ValueError(
            f"Only {len(valid_episodes)} Bridge episodes have span >= {span}, "
            f"but eval.num_eval={cfg.eval.num_eval}"
        )

    rng = np.random.default_rng(cfg.seed)
    eval_episodes = rng.choice(valid_episodes, size=cfg.eval.num_eval, replace=False)
    rows = []
    start_time = time.time()

    for eval_idx, ep in enumerate(eval_episodes):
        max_start = int(dataset.lengths[ep]) - span
        start = int(rng.integers(0, max_start + 1))
        chunk = dataset.load_chunk(
            np.asarray([ep]), np.asarray([start]), np.asarray([start + span])
        )[0]

        pixels = chunk["pixels"][:history_size]
        goal_idx = history_size - 1 + future_tokens
        goal = chunk["pixels"][goal_idx : goal_idx + 1]
        expert = chunk["action"][:total_action_tokens]
        history_actions = expert[:history_size]
        expert_future_actions = expert[history_size:]

        item = {
            "pixels": pixels,
            "goal": goal,
            "action": expert,
        }
        info = tensor_batch(item, device)
        info = {k: v.unsqueeze(1) for k, v in info.items()}

        expert_actions = expert_future_actions.unsqueeze(0).unsqueeze(0).to(device)
        random_actions = torch.randn(
            1,
            int(cfg.eval.num_random_actions),
            future_action_tokens,
            action_dim,
            device=device,
        )
        ranking_actions = torch.cat([expert_actions, random_actions], dim=1)

        costs = cost_actions(
            model, info, prepend_history_actions(history_actions, ranking_actions)
        )[0]
        expert_cost = float(costs[0])
        random_costs = costs[1:]
        rank = int((random_costs < expert_cost).sum().item()) + 1
        zero_cost = float(
            cost_actions(
                model,
                info,
                prepend_history_actions(history_actions, torch.zeros_like(expert_actions)),
            )[0, 0]
        )
        shuffled_expert = shuffle_future_actions(expert_future_actions)
        shuffled_cost = float(
            cost_actions(
                model,
                info,
                prepend_history_actions(
                    history_actions, shuffled_expert.unsqueeze(0).unsqueeze(0)
                ),
            )[0, 0]
        )
        tf_metrics = teacher_forced_one_step_loss(
            model,
            chunk["pixels"][: history_size + 1],
            chunk["action"][: history_size],
            history_size,
        )

        cem_cost, _ = cem_latent_plan(
            model,
            info,
            (future_action_tokens, action_dim),
            cfg,
            history_actions=history_actions,
        )
        start_cost = float(latent_distance(model, info["pixels"][:, 0], info["goal"][:, 0])[0])
        random_mean = float(random_costs.mean())
        random_best = float(random_costs.min())
        cem_cost = float(cem_cost)

        row = {
            "eval_idx": eval_idx,
            "episode": int(ep),
            "start": start,
            "start_cost": start_cost,
            "expert_cost": expert_cost,
            "random_mean_cost": random_mean,
            "random_best_cost": random_best,
            "cem_cost": cem_cost,
            "zero_action_cost": zero_cost,
            "shuffled_expert_cost": shuffled_cost,
            "expert_rank": rank,
            "expert_rank_percentile": (rank - 1) / max(1, int(cfg.eval.num_random_actions)),
            "cem_vs_start_improvement": (start_cost - cem_cost) / max(start_cost, 1e-8),
            "cem_vs_random_best_improvement": (random_best - cem_cost) / max(random_best, 1e-8),
            "cem_vs_random_mean_improvement": (random_mean - cem_cost) / max(random_mean, 1e-8),
            "cem_expert_cost_ratio": cem_cost / max(expert_cost, 1e-8),
            "expert_vs_random_best_improvement": (random_best - expert_cost) / max(random_best, 1e-8),
            "expert_vs_zero_improvement": (zero_cost - expert_cost) / max(zero_cost, 1e-8),
            "expert_vs_shuffled_improvement": (shuffled_cost - expert_cost) / max(shuffled_cost, 1e-8),
        }
        row.update(tf_metrics)
        for k in recall_ks:
            row[f"expert_recall@{k}"] = float(rank <= int(k))
        rows.append(row)
        print(
            f"[{eval_idx + 1}/{cfg.eval.num_eval}] ep={ep} start={start} "
            f"rank={rank} expert={expert_cost:.4f} zero={zero_cost:.4f} "
            f"shuf={shuffled_cost:.4f} random_best={random_best:.4f} cem={cem_cost:.4f}"
        )

    metrics = mean_metrics(rows)
    metrics["num_eval"] = int(cfg.eval.num_eval)
    metrics["num_random_actions"] = int(cfg.eval.num_random_actions)
    metrics["history_size"] = history_size
    metrics["future_tokens"] = future_tokens
    metrics["total_action_tokens"] = total_action_tokens
    metrics["history_action_tokens"] = history_size
    metrics["optimized_action_tokens"] = future_action_tokens
    metrics["action_dim"] = action_dim
    metrics["evaluation_time"] = time.time() - start_time

    print(json.dumps(metrics, indent=2))

    output_path, json_path = write_eval_results(
        eval_output_dir(results_path, cfg),
        cfg.output.filename,
        cfg,
        metrics,
        metrics["evaluation_time"],
        episodes=rows,
    )

    print(f"Saved Bridge offline metrics to {output_path}")
    print(f"Saved per-example details to {json_path}")
    return metrics


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Evaluate LRC-JEPA with MPC or offline Bridge-v2 planning."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    if cfg.eval.get("offline_latent", False):
        run_bridge_offline_eval(cfg)
        return

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = episode_column_name(dataset)
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model, results_path = load_policy_model(cfg.policy)
        model = model.to(cfg.solver.device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()
        results_path = Path(__file__).parent

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_per_row = episode_values_for_rows(dataset, ep_indices, max_start_idx)
    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    output_dir = eval_output_dir(results_path, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=None,
    )
    end_time = time.time()
    metrics.update(success_stats(metrics))
    print(to_jsonable(metrics))

    output_path, json_path = write_eval_results(
        output_dir,
        cfg.output.filename,
        cfg,
        metrics,
        end_time - start_time,
    )
    print(f"Saved eval metrics to {output_path}")
    print(f"Saved eval details to {json_path}")


if __name__ == "__main__":
    run()
