from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import mne
import pandas as pd

from .config import ChannelConfig, PipelineConfig


def find_edf_files(root: Path) -> List[Path]:
    """Recursively list EDF files under the given directory."""
    return sorted(root.rglob("*.edf"))


def _has_channels(raw: mne.io.BaseRaw, channels: Iterable[str]) -> bool:
    names = set(raw.ch_names)
    return all(ch in names for ch in channels)


def validate_dataset(files: List[Path], channel_cfg: ChannelConfig, min_duration: int = 10) -> pd.DataFrame:
    """
    Validate that each EDF file contains the expected channels and duration.
    Returns a dataframe summarizing the checks.
    """
    checks = []
    for f in files:
        raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
        duration = raw.times[-1]
        checks.append(
            {
                "file": f.as_posix(),
                "has_eeg_channels": _has_channels(raw, channel_cfg.eeg_channels),
                "has_marker_channels": _has_channels(raw, channel_cfg.marker_channels),
                "duration_sec": round(duration, 2),
                "duration_ok": duration >= min_duration,
            }
        )
    return pd.DataFrame(checks)


def segment_times(raw: mne.io.BaseRaw, channel_cfg: ChannelConfig) -> List[Tuple[float, float]]:
    """
    Use marker channels to derive segment start/end timestamps.
    Mirrors the marker logic used in the exploration notebook.
    """
    marker_data = raw.copy().pick(channel_cfg.marker_channels)
    marker_values, marker_times = marker_data[:]
    non_zero_indices = (marker_values != 0).any(axis=0).nonzero()[0]
    event_times = marker_times[non_zero_indices]
    if len(event_times) < 2:
        return []
    return list(zip(event_times[:-1], event_times[1:]))


def load_raw(path: Path, preload: bool = True) -> mne.io.BaseRaw:
    """Wrapper around mne.read_raw_edf with sane defaults."""
    return mne.io.read_raw_edf(path, preload=preload, verbose=False)


def describe_config(cfg: PipelineConfig) -> str:
    """Pretty-print the active configuration."""
    return pd.Series(asdict(cfg)).to_string()
