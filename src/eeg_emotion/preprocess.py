from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig


def winsorize_band_columns(df: pd.DataFrame, band_cols: Iterable[str]) -> pd.DataFrame:
    """
    Clip bandpower columns using the IQR rule, matching the notebook logic.
    """
    df_out = df.copy()

    def _wins(series: pd.Series) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return series.clip(lower=lower, upper=upper)

    for col in band_cols:
        if col in df_out.columns:
            df_out[col] = _wins(df_out[col])
    return df_out


def build_pca_dataset(df: pd.DataFrame, n_components: int, cfg: PipelineConfig):
    """
    Standardize bandpower + quality features, apply PCA, and append battery stats.
    Returns (final_df, scaler, pca).
    """
    band_cols = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    quality_cols = [c for c in df.columns if c.startswith("CQ_") or c.startswith("EQ_")]
    brain_cols = band_cols + quality_cols
    battery_cols = ["BATTERY_mean", "BATTERY_std"]
    meta_cols = ["Participant", "Scenario", "Segment", "Channel"]

    required_cols = set(brain_cols + battery_cols + meta_cols)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    X = df[brain_cols].astype(float).values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=n_components, random_state=cfg.features.random_seed).fit(X_scaled)
    pcs = pca.transform(X_scaled)

    pc_df = pd.DataFrame(
        pcs, columns=[f"Brain_PC{i+1}" for i in range(pcs.shape[1])]
    )

    final_df = pd.concat(
        [
            df[meta_cols].reset_index(drop=True),
            pc_df.reset_index(drop=True),
            df[battery_cols].reset_index(drop=True).astype(float),
        ],
        axis=1,
    )
    return final_df, scaler, pca


def preprocess_features(raw_features: pd.DataFrame, cfg: PipelineConfig):
    """
    Run winsorization and PCA to produce both 10 and N-component datasets.
    """
    band_cols = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    wins_df = winsorize_band_columns(raw_features, band_cols) if cfg.features.winsorize else raw_features

    final_10, scaler_10, pca_10 = build_pca_dataset(wins_df, 10, cfg)
    final_n, scaler_n, pca_n = build_pca_dataset(wins_df, cfg.features.pca_components, cfg)

    return {
        "winsorized": wins_df,
        "final_10": (final_10, scaler_10, pca_10),
        "final_n": (final_n, scaler_n, pca_n),
    }


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
