import errno
import json
import os
import tempfile
import time
from pathlib import Path


def patch_stable_pretraining_sidecar_atomic_write(max_retries=50, base_delay=0.02):
    """Retry sidecar atomic replace when Windows briefly locks run metadata files."""

    from stable_pretraining.registry import _sidecar

    if getattr(_sidecar, "_lrc_jepa_atomic_json_retry_patch", False):
        return _sidecar.atomic_json_write

    def atomic_json_write_with_windows_retry(dest, data):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=False, default=_sidecar._json_default)
                f.flush()
                os.fsync(f.fileno())

            for attempt in range(max_retries + 1):
                try:
                    os.replace(tmp, dest)
                    return dest
                except PermissionError:
                    if attempt >= max_retries:
                        raise
                    time.sleep(base_delay * min(10, attempt + 1))
                except OSError as exc:
                    if exc.errno != errno.EACCES or attempt >= max_retries:
                        raise
                    time.sleep(base_delay * min(10, attempt + 1))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        return dest

    _sidecar.atomic_json_write = atomic_json_write_with_windows_retry
    _sidecar._lrc_jepa_atomic_json_retry_patch = True
    return atomic_json_write_with_windows_retry


def patch_dm_control_missing_mujoco_fields():
    """Skip dm_control MuJoCo fields absent from the installed mujoco runtime."""

    try:
        from dm_control.mujoco import index
    except ImportError:
        return

    if getattr(index.struct_indexer, "_lrc_jepa_missing_fields_patch", False):
        return

    original = index.struct_indexer

    def patched_struct_indexer(struct, struct_name, size_to_axis_indexer):
        struct_name = struct_name.lower()
        array_sizes = index.sizes.array_sizes.get(struct_name)
        if not array_sizes:
            return original(struct, struct_name, size_to_axis_indexer)

        missing = [field for field in array_sizes if not hasattr(struct, field)]
        if not missing:
            return original(struct, struct_name, size_to_axis_indexer)

        filtered = {
            field: size_names
            for field, size_names in array_sizes.items()
            if field not in missing
        }
        index.sizes.array_sizes[struct_name] = filtered
        try:
            return original(struct, struct_name, size_to_axis_indexer)
        finally:
            index.sizes.array_sizes[struct_name] = array_sizes

    patched_struct_indexer._lrc_jepa_missing_fields_patch = True
    index.struct_indexer = patched_struct_indexer
