from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from .config import PipelineConfig


def _feature_columns(df: pd.DataFrame, include_battery: bool = True) -> List[str]:
    cols = [c for c in df.columns if c.startswith("Brain_PC")]
    if include_battery:
        cols += ["BATTERY_mean", "BATTERY_std"]
    return cols


class ScenarioDataset(Dataset):
    """Torch dataset that standardizes features and encodes labels."""

    def __init__(self, df: pd.DataFrame, include_battery: bool = True):
        feat_cols = _feature_columns(df, include_battery)
        data = df[feat_cols].dropna()
        self.feature_names = feat_cols
        self.X = data.values.astype(np.float32)

        y = df.loc[data.index, "Scenario"]
        self.le = LabelEncoder().fit(y)
        self.y = self.le.transform(y).astype(np.int64)

        self.scaler = StandardScaler().fit(self.X)
        self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)
        for in_d, out_d in zip(dims, dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainingArtifacts:
    model: torch.nn.Module
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_names: List[str]
    accuracy: float


def train_neural_net(df: pd.DataFrame, cfg: PipelineConfig, include_battery: bool = True) -> TrainingArtifacts:
    ds = ScenarioDataset(df, include_battery=include_battery)
    input_dim = ds.X.shape[1]
    num_classes = len(ds.le.classes_)

    test_size = int(len(ds) * cfg.model.test_frac)
    train_size = len(ds) - test_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.model.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.model.batch_size, shuffle=False)

    model = SimpleNN(input_dim, cfg.model.hidden_dims, num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr)

    model.train()
    for _ in range(cfg.model.epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = float(accuracy_score(y_true, y_pred))

    return TrainingArtifacts(
        model=model.cpu(),
        scaler=ds.scaler,
        label_encoder=ds.le,
        feature_names=ds.feature_names,
        accuracy=acc,
    )


def save_artifacts(artifacts: TrainingArtifacts, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts.model.state_dict(), out_dir / "scenario_model.pth")
    joblib.dump(artifacts.scaler, out_dir / "scaler.joblib")
    joblib.dump(artifacts.label_encoder, out_dir / "label_encoder.joblib")
    joblib.dump(artifacts.feature_names, out_dir / "feature_order.joblib")


def train_random_forest(df: pd.DataFrame) -> Dict[str, object]:
    feat_cols = _feature_columns(df)
    X = df[feat_cols].dropna()
    y = df.loc[X.index, "Scenario"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return {"model": model, "accuracy": acc, "report": report, "feature_names": feat_cols}
