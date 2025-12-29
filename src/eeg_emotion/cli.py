from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import uvicorn

from .api import create_app
from .config import PipelineConfig
from .features import extract_features_from_dataset, save_features
from .inference import predict_file
from .models import save_artifacts, train_neural_net
from .pipeline import run_full_pipeline
from .preprocess import preprocess_features, save_dataframe

app = typer.Typer(add_completion=False, help="EEG emotional state classification pipeline.")


@app.command()
def pipeline(
    cfg: Optional[Path] = typer.Option(None, help="Path to YAML config."),
    raw_dir: Optional[Path] = typer.Option(None, help="Directory with raw EDF files."),
    processed_dir: Optional[Path] = typer.Option(None, help="Where to store processed CSVs."),
    models_dir: Optional[Path] = typer.Option(None, help="Where to store trained artifacts."),
):
    """Run the end-to-end pipeline."""
    result = run_full_pipeline(cfg, raw_dir, processed_dir, models_dir)
    typer.echo(f"Pipeline finished. Manifest: {result['manifest_path']}")


@app.command()
def features(
    raw_dir: Path = typer.Option(..., help="Root folder containing scenario sub-folders with EDF files."),
    out: Path = typer.Option(Path("data/processed/all_participants_features.csv"), help="Output CSV path."),
    cfg: Optional[Path] = typer.Option(None, help="Optional YAML config."),
):
    """Extract features from raw EDF files."""
    config = PipelineConfig.from_yaml(cfg)
    df = extract_features_from_dataset(raw_dir, config)
    save_features(df, out)
    typer.echo(f"Saved features to {out}")


@app.command()
def preprocess(
    features_csv: Path = typer.Option(..., help="CSV from the feature extraction step."),
    components: int = typer.Option(20, help="Number of PCA components for the main dataset."),
    out_dir: Path = typer.Option(Path("data/processed"), help="Directory to store preprocessed CSVs."),
    cfg: Optional[Path] = typer.Option(None, help="Optional YAML config."),
):
    """Winsorize band powers and compute PCA features."""
    config = PipelineConfig.from_yaml(cfg)
    config.features.pca_components = components
    raw_df = pd.read_csv(features_csv)
    pre = preprocess_features(raw_df, config)
    save_dataframe(pre["winsorized"], out_dir / "winsorized_bandpower_features.csv")
    final10, _, _ = pre["final_10"]
    finaln, _, _ = pre["final_n"]
    save_dataframe(final10, out_dir / "final_preprocessed_10.csv")
    save_dataframe(finaln, out_dir / f"final_preprocessed_{components}.csv")
    typer.echo(f"Preprocessed data stored in {out_dir}")


@app.command()
def train(
    data: Path = typer.Option(..., help="Preprocessed CSV (e.g., final_preprocessed_20.csv)."),
    models_dir: Path = typer.Option(Path("models"), help="Where to store trained artifacts."),
    cfg: Optional[Path] = typer.Option(None, help="Optional YAML config."),
):
    """Train the neural net classifier and save artifacts."""
    config = PipelineConfig.from_yaml(cfg)
    df = pd.read_csv(data)
    artifacts = train_neural_net(df, config, include_battery=True)
    save_artifacts(artifacts, models_dir)
    typer.echo(f"Model saved to {models_dir} (accuracy={artifacts.accuracy:.3f})")


@app.command()
def predict(
    data: Path = typer.Option(..., help="Preprocessed CSV to run inference on."),
    model_dir: Path = typer.Option(Path("models"), help="Directory containing saved artifacts."),
    out: Optional[Path] = typer.Option(None, help="Optional path to save predictions CSV."),
    cfg: Optional[Path] = typer.Option(None, help="Optional YAML config."),
):
    """Predict scenarios for a CSV of PCA features."""
    preds = predict_file(data, model_dir, PipelineConfig.from_yaml(cfg))
    if out:
        preds.to_csv(out, index=False)
        typer.echo(f"Wrote predictions to {out}")
    else:
        typer.echo(preds)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host for the inference API."),
    port: int = typer.Option(8000, help="Port for the inference API."),
    model_dir: Path = typer.Option(Path("models"), help="Directory with trained artifacts."),
    cfg: Optional[Path] = typer.Option(None, help="Optional YAML config."),
):
    """Start a FastAPI inference server."""
    app_factory = lambda: create_app(model_dir=model_dir, cfg_path=cfg)
    uvicorn.run(app_factory, host=host, port=port, factory=True)


if __name__ == "__main__":
    app()
