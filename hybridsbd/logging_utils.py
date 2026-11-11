"""Shared logging / persistence helpers so no run information is lost.

* ``tee_to_file`` mirrors everything printed (stdout + stderr, incl. tracebacks)
  into an append-only log file, flushed on every write.
* ``save_predictions`` stores the per-video probability arrays so results can be
  re-analysed (thresholds, Tables 8/9) without re-running the GPU model.
* ``result_metadata`` stamps every result file with checkpoint hash, git commit
  and environment.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import numpy as np


class _Tee:
    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle

    def write(self, data):
        self.stream.write(data)
        try:
            self.handle.write(data)
            self.handle.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        self.stream.flush()
        try:
            self.handle.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def tee_to_file(path):
    """Mirror stdout/stderr to ``path`` (append mode). Returns the file handle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", buffering=1, encoding="utf-8")
    handle.write(f"\n===== {datetime.datetime.now().isoformat()} :: {' '.join(sys.argv)} =====\n")
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    return handle


def save_predictions(path, video_y, video_z=None):
    """Save full-video probabilities (compressed npz) for offline re-analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"y::{k}": v for k, v in video_y.items()}
    if video_z is not None:
        arrays.update({f"z::{k}": v for k, v in video_z.items()})
    np.savez_compressed(path, **arrays)
    return str(path)


def load_predictions(path):
    data = np.load(path, allow_pickle=False)
    video_y, video_z = {}, {}
    for key in data.files:
        kind, name = key.split("::", 1)
        (video_y if kind == "y" else video_z)[name] = data[key]
    return video_y, (video_z or None)


def result_metadata(checkpoint, single_output, config=None):
    from hybridsbd.configs.ablation_config import environment_snapshot
    from hybridsbd.utils import file_sha256

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": os.fspath(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "single_output_rule": bool(single_output),
        "inference_rule": "b = p_y > tau_c" if single_output else "b = (p_z > tau_g) OR (p_y > tau_c)",
        "evaluation_protocol": "A_test_selected",
        "environment": environment_snapshot(),
    }
    if config is not None:
        meta["args"] = {k: v for k, v in config.items() if not k.startswith("_")}
    return meta
