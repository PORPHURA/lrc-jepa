"""Shared runtime setup for training and evaluation."""
import os
import signal
import sys
import types
from pathlib import Path


def configure_runtime():
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    root = Path(os.environ.get("LOCAL_DATASET_DIR") or os.environ.get("STABLEWM_HOME") or "data").expanduser().resolve()
    os.environ.setdefault("LOCAL_DATASET_DIR", str(root))
    os.environ.setdefault("STABLEWM_HOME", str(root))
    if os.name == "nt" and os.environ.get("MUJOCO_GL", "").lower() == "egl":
        os.environ["MUJOCO_GL"] = "glfw"
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    for name in ("SIGUSR1", "SIGUSR2", "SIGCONT"):
        if not hasattr(signal, name):
            setattr(signal, name, signal.SIGTERM)
    # Compatibility with the datasets/pyarrow versions used by the original runs.
    import pyarrow as pa
    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType
    import datasets
    if not hasattr(datasets, "config"):
        config = types.ModuleType("datasets.config")
        config.DATASET_STATE_JSON_FILENAME = "dataset_state.json"
        datasets.config = config
        sys.modules["datasets.config"] = config
