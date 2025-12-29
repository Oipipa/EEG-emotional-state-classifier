from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch

from .config import PipelineConfig
from .data_ingestion import load_raw, segment_times


def _bandpower(psd: np.ndarray, freqs: np.ndarray, bands: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    return {
        band: float(psd[(freqs >= low) & (freqs <= high)].mean())
        for band, (low, high) in bands.items()
    }


def extract_features_from_raw(raw, cfg: PipelineConfig, participant: str, scenario: str) -> pd.DataFrame:
    """
    Generate per-segment bandpower, quality, and battery statistics
    for a single EDF recording.
    """
    channels = cfg.channels
    feature_rows: List[Dict[str, float]] = []

    raw_eeg = raw.copy().pick(channels.eeg_channels)
    raw_eeg.filter(1.0, 40.0, verbose=False)

    seg_pairs = segment_times(raw, channels)
    if not seg_pairs:
        return pd.DataFrame()

    quality_data = raw.copy().pick(channels.quality_channels)
    quality_vals, quality_times = quality_data[:]
    battery_data = raw.copy().pick(channels.battery_channels)
    battery_vals, battery_times = battery_data[:]

    sfreq = raw.info.get("sfreq", cfg.features.sampling_rate)
    for seg_idx, (start, end) in enumerate(seg_pairs, start=1):
        segment = raw_eeg.copy().crop(tmin=start, tmax=end)
        data, _ = segment[:]
        data_uv = data * 1e6  # convert to µV

        quality_idx = (quality_times >= start) & (quality_times <= end)
        battery_idx = (battery_times >= start) & (battery_times <= end)

        for ch_idx, ch_name in enumerate(channels.eeg_channels):
            ch_data = data_uv[ch_idx]
            freqs, psd = welch(ch_data, fs=sfreq, nperseg=256)
            band_powers = _bandpower(psd, freqs, cfg.features.bands)

            row = {
                "Participant": participant,
                "Scenario": scenario,
                "Segment": f"Segment_{seg_idx}",
                "Channel": ch_name,
                **band_powers,
            }

            for q_idx, q_name in enumerate(channels.quality_channels):
                seg_q = quality_vals[q_idx, quality_idx]
                row[f"{q_name}_mean"] = float(np.mean(seg_q)) if seg_q.size else np.nan
                row[f"{q_name}_std"] = float(np.std(seg_q)) if seg_q.size else np.nan

            for b_idx, b_name in enumerate(channels.battery_channels):
                seg_b = battery_vals[b_idx, battery_idx]
                row[f"{b_name}_mean"] = float(np.mean(seg_b)) if seg_b.size else np.nan
                row[f"{b_name}_std"] = float(np.std(seg_b)) if seg_b.size else np.nan

            feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def extract_features_from_dataset(data_dir: Path, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Walk the dataset directory (organized as scenario/edf files) and
    concatenate all extracted features.
    """
    all_rows = []
    for scenario_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        scenario_name = scenario_dir.name
        for edf_path in sorted(scenario_dir.glob("*.edf")):
            participant = edf_path.stem.replace(" ", "_")
            raw = load_raw(edf_path, preload=True)
            df = extract_features_from_raw(raw, cfg, participant, scenario_name)
            if not df.empty:
                all_rows.append(df)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def save_features(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
