import numpy as np
import torch
from stable_pretraining import data as dt
import stable_worldmodel as swm
from stable_worldmodel.data.dataset import Dataset
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from pathlib import Path
import json
import logging
import re


def get_img_preprocessor(
    source: str,
    target: str,
    img_size: int = 224,
    resize_images: bool = True,
):
    imagenet_stats = dt.dataset_stats.ImageNet
    transforms = [dt.transforms.ToImage(**imagenet_stats, source=source, target=target)]
    if resize_images:
        transforms.append(dt.transforms.Resize(img_size, source=source, target=target))
    return dt.transforms.Compose(*transforms)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


class NpyMemmapDataset(Dataset):
    """Dataset backed by a directory of row-aligned .npy arrays."""

    META_NAME = "metadata.json"

    def __init__(
        self,
        path,
        frameskip=1,
        num_steps=1,
        transform=None,
        keys_to_load=None,
        keys_to_cache=None,
        keys_to_merge=None,
        **_,
    ):
        self.path = Path(path)
        with open(self.path / self.META_NAME, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self._keys = keys_to_load or list(self.metadata["columns"])
        self._arrays = {}
        self._shards = {}
        self._cache = {}
        lengths = np.load(self.path / "ep_len.npy", mmap_mode="r")
        offsets = np.load(self.path / "ep_offset.npy", mmap_mode="r")

        for key in keys_to_cache or []:
            self._cache[key] = self._load_full_column(key)
            logging.info("Cached '%s' from '%s'", key, self.path)

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self):
        return self._keys

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}
        state["_shards"] = {}
        return state

    def _array(self, col):
        if col in self._cache:
            return self._cache[col]
        if self.metadata.get("storage") == "npy_sharded":
            raise RuntimeError(f"column {col!r} is sharded; use _load_rows")
        if col not in self._arrays:
            self._arrays[col] = np.load(self.path / f"{col}.npy", mmap_mode="r")
        return self._arrays[col]

    def _shard(self, col, shard_idx):
        key = (col, shard_idx)
        if key not in self._shards:
            shard_path = self.path / col / f"{shard_idx:06d}.npy"
            self._shards[key] = np.load(shard_path, mmap_mode="r")
        return self._shards[key]

    def _load_rows(self, col, start, end):
        if col in self._cache:
            return self._cache[col][start:end]
        if self.metadata.get("storage") != "npy_sharded":
            return self._array(col)[start:end]

        shard_rows = int(self.metadata["shard_rows"])
        first = start // shard_rows
        last = (end - 1) // shard_rows
        if first == last:
            shard = self._shard(col, first)
            return shard[start - first * shard_rows : end - first * shard_rows]

        pieces = []
        for shard_idx in range(first, last + 1):
            shard = self._shard(col, shard_idx)
            shard_start = shard_idx * shard_rows
            local_start = max(start - shard_start, 0)
            local_end = min(end - shard_start, len(shard))
            pieces.append(shard[local_start:local_end])
        return np.concatenate(pieces, axis=0)

    def _load_full_column(self, col):
        if self.metadata.get("storage") != "npy_sharded":
            return np.load(self.path / f"{col}.npy", mmap_mode=None)
        num_shards = int(self.metadata["arrays"][col]["num_shards"])
        return np.concatenate(
            [np.asarray(self._shard(col, shard_idx)) for shard_idx in range(num_shards)],
            axis=0,
        )

    def _load_slice(self, ep_idx, start, end):
        g_start, g_end = self.offsets[ep_idx] + start, self.offsets[ep_idx] + end
        steps = {}
        for col in self._keys:
            data = self._load_rows(col, g_start, g_end)
            if col != "action":
                data = data[:: self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                val = data[0] if len(data) > 0 else b""
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                array = np.asarray(data)
                if not array.flags.writeable:
                    array = array.copy()
                steps[col] = torch.from_numpy(array)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col):
        return np.asarray(self._load_full_column(col))

    def get_row_data(self, row_idx):
        if isinstance(row_idx, (int, np.integer)):
            return {
                col: self._load_rows(col, int(row_idx), int(row_idx) + 1)[0]
                for col in self._keys
            }

        indices = np.asarray(row_idx)
        return {
            col: np.asarray(
                [self._load_rows(col, int(idx), int(idx) + 1)[0] for idx in indices.flat]
            ).reshape(indices.shape + tuple(self.metadata["arrays"][col]["shape"][1:]))
            for col in self._keys
        }

    def merge_col(self, source, target, dim=-1):
        if isinstance(source, str):
            source = [k for k in self.metadata["columns"] if re.match(source, k)]
        merged = np.concatenate([self.get_col_data(s) for s in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info("Merged columns %s into '%s' and cached it", source, target)

    def get_dim(self, col):
        shape = self.metadata["arrays"][col]["shape"]
        return int(np.prod(shape[1:])) if len(shape) > 1 else 1


def prepare_training_dataset(dataset_cfg, img_size, cache_dir=None, resize_images=True):
    """Load one HDF5 or NPY dataset using the training action normalization."""
    cfg = dict(dataset_cfg)
    name = cfg.pop("name")
    path = Path(name).expanduser()
    if not path.exists():
        path = Path(cache_dir or "data") / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Set LOCAL_DATASET_DIR.")
    if path.is_dir() and (path / NpyMemmapDataset.META_NAME).exists():
        dataset = NpyMemmapDataset(path, transform=None, **cfg)
    elif path.suffix == ".h5":
        dataset = swm.data.HDF5Dataset(path=path, transform=None, **cfg)
    else:
        raise ValueError(f"Expected HDF5 file or NPY dataset directory: {path}")
    transforms = [get_img_preprocessor("pixels", "pixels", img_size, resize_images)]
    for column in cfg["keys_to_load"]:
        if not column.startswith("pixels"):
            transforms.append(get_column_normalizer(dataset, column, column))
    dataset.transform = dt.transforms.Compose(*transforms)
    return dataset, [dataset.get_dim("action")]


def prepare_resume_checkpoint(ckpt_path):
    if not ckpt_path.exists():
        return None

    return ckpt_path


class LRCModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that avoids resuming with stale per-epoch batch state."""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        fit_loop = checkpoint.get("loops", {}).get("fit_loop", {})
        batch_progress = fit_loop.get("epoch_loop.batch_progress")
        epoch_progress = fit_loop.get("epoch_progress")
        if not isinstance(batch_progress, dict):
            return

        current = batch_progress.get("current")
        if not isinstance(current, dict):
            return

        if not batch_progress.get("is_last_batch", False):
            return

        if isinstance(epoch_progress, dict):
            for scope in ("total", "current"):
                tracker = epoch_progress.get(scope)
                if not isinstance(tracker, dict):
                    continue
                ready = tracker.get("ready")
                started = tracker.get("started")
                processed = tracker.get("processed")
                completed = tracker.get("completed")
                if (
                    isinstance(processed, int)
                    and ready == started == processed
                    and completed == processed - 1
                ):
                    tracker["completed"] = processed
                    checkpoint["epoch"] = processed

        for key in ("ready", "started", "processed", "completed"):
            if key in current:
                current[key] = 0
        batch_progress["is_last_batch"] = False
        logging.info("Closed epoch-end progress and reset batch_progress.current before saving checkpoint.")


class SaveCkptCallback(Callback):
    """Save model checkpoints into the local run output directory."""

    def __init__(self, save_dir, epoch_interval: int = 1):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.save_dir / f"weights_epoch_{epoch}.pt")
