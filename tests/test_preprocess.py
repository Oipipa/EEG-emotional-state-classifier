import pandas as pd
import numpy as np

from eeg_emotion.config import PipelineConfig
from eeg_emotion.preprocess import winsorize_band_columns, build_pca_dataset


def _dummy_row(seed: int = 0):
    np.random.seed(seed)
    row = {
        "Participant": f"p{seed}",
        "Scenario": "TestScenario",
        "Segment": "Segment_1",
        "Channel": "AF3",
        "Delta": np.random.rand(),
        "Theta": np.random.rand(),
        "Alpha": np.random.rand(),
        "Beta": np.random.rand(),
        "Gamma": np.random.rand(),
        "BATTERY_mean": 0.5,
        "BATTERY_std": 0.1,
    }
    cq = ["CQ_AF3", "CQ_T7", "CQ_Pz", "CQ_T8", "CQ_AF4", "CQ_OVERALL"]
    eq = ["EQ_AF3", "EQ_T7", "EQ_Pz", "EQ_T8", "EQ_AF4", "EQ_OVERALL"]
    for name in cq + eq:
        row[f"{name}_mean"] = np.random.rand()
        row[f"{name}_std"] = np.random.rand()
    return row


def make_df(n: int = 20):
    return pd.DataFrame([_dummy_row(i) for i in range(n)])


def test_build_pca_dataset_produces_expected_columns():
    cfg = PipelineConfig()
    df = make_df()
    wins = winsorize_band_columns(df, ["Delta", "Theta", "Alpha", "Beta", "Gamma"])
    final, scaler, pca = build_pca_dataset(wins, n_components=5, cfg=cfg)

    assert list(final.columns[:4]) == ["Participant", "Scenario", "Segment", "Channel"]
    assert final.shape[1] == 4 + 5 + 2  # meta + PCs + battery stats
    assert scaler.mean_.shape[0] == len([c for c in df.columns if c.startswith("CQ_") or c.startswith("EQ_")] + ["Delta", "Theta", "Alpha", "Beta", "Gamma"])
