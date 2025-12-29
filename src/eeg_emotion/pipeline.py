from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig
from .data_ingestion import find_edf_files, validate_dataset
from .features import extract_features_from_dataset, save_features
from .models import save_artifacts, train_neural_net
from .preprocess import preprocess_features, save_dataframe


def run_full_pipeline(
    cfg_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> dict:
    """
    Execute the end-to-end pipeline: validate data, extract features,
    preprocess, train the neural net, and persist artifacts.
    """
    cfg = PipelineConfig.from_yaml(cfg_path)
    raw_dir = Path(raw_dir) if raw_dir else cfg.paths.raw
    processed_dir = Path(processed_dir) if processed_dir else cfg.paths.processed
    models_dir = Path(models_dir) if models_dir else cfg.paths.models

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    files = find_edf_files(raw_dir)
    validation_df = validate_dataset(files, cfg.channels, cfg.features.min_duration_sec)

    features_df = extract_features_from_dataset(raw_dir, cfg)
    features_path = processed_dir / "all_participants_features.csv"
    save_features(features_df, features_path)

    pre = preprocess_features(features_df, cfg)
    wins_path = processed_dir / "winsorized_bandpower_features.csv"
    save_dataframe(pre["winsorized"], wins_path)

    final10, scaler10, pca10 = pre["final_10"]
    finaln, scaler_n, pca_n = pre["final_n"]

    save_dataframe(final10, processed_dir / "final_preprocessed_10.csv")
    save_dataframe(finaln, processed_dir / f"final_preprocessed_{cfg.features.pca_components}.csv")

    artifacts = train_neural_net(finaln, cfg, include_battery=True)
    save_artifacts(artifacts, models_dir)

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": cfg_path.as_posix() if cfg_path else "default",
        "data_files": {
            "features": features_path.as_posix(),
            "winsorized": wins_path.as_posix(),
            "final": (processed_dir / f"final_preprocessed_{cfg.features.pca_components}.csv").as_posix(),
        },
        "validation": validation_df.to_dict(orient="list"),
        "metrics": {"neural_net_accuracy": artifacts.accuracy},
        "feature_order": artifacts.feature_names,
        "pca_components": cfg.features.pca_components,
    }
    manifest_path = models_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "validation": validation_df,
        "features": features_df,
        "final": finaln,
        "artifacts": artifacts,
        "manifest_path": manifest_path,
    }
