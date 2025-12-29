from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from .config import PipelineConfig
from .models import SimpleNN


@dataclass
class LoadedModel:
    model: torch.nn.Module
    scaler: object
    label_encoder: object
    feature_order: List[str]


def _default_feature_order(n_pcs: int) -> List[str]:
    return [f"Brain_PC{i}" for i in range(1, n_pcs + 1)] + ["BATTERY_mean", "BATTERY_std"]


def load_model(model_dir: Path, cfg: PipelineConfig | None = None) -> LoadedModel:
    """
    Load saved artifacts (model weights, scaler, label encoder, feature order).
    """
    model_dir = Path(model_dir)
    scaler = joblib.load(model_dir / "scaler.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    feature_order_path = model_dir / "feature_order.joblib"
    if feature_order_path.exists():
        feature_order = joblib.load(feature_order_path)
    else:
        n_features = getattr(scaler, "n_features_in_", len(scaler.mean_))
        feature_order = _default_feature_order(n_features - 2)

    hidden_dims = cfg.model.hidden_dims if cfg else (256, 128, 64, 128, 256)
    input_dim = len(feature_order)
    num_classes = len(label_encoder.classes_)

    model = SimpleNN(input_dim, hidden_dims, num_classes)
    state_dict = torch.load(model_dir / "scenario_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedModel(model=model, scaler=scaler, label_encoder=label_encoder, feature_order=feature_order)


def predict(df: pd.DataFrame, loaded: LoadedModel) -> pd.DataFrame:
    """
    Run inference on a dataframe containing preprocessed features.
    Returns dataframe with predictions and probabilities.
    """
    X = df[loaded.feature_order].astype(float)
    X_scaled = loaded.scaler.transform(X)
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = loaded.model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
    labels = loaded.label_encoder.inverse_transform(preds)
    return pd.DataFrame({
        "prediction": labels,
        "confidence": probs.max(axis=1),
    })


def predict_file(csv_path: Path, model_dir: Path, cfg: PipelineConfig | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    loaded = load_model(model_dir, cfg)
    return predict(df, loaded)
