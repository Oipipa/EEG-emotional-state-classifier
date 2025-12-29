from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import PipelineConfig
from .inference import LoadedModel, load_model, predict


class Item(BaseModel):
    features: List[float] = Field(..., description="Ordered feature values matching the model's expected order.")
    feature_names: Optional[List[str]] = Field(
        None, description="Optional feature names; if provided, they will be reordered to the trained order."
    )


class BatchRequest(BaseModel):
    items: List[Item]


def _build_dataframe(payload: BatchRequest, loaded: LoadedModel) -> pd.DataFrame:
    rows = []
    for item in payload.items:
        if item.feature_names:
            provided = dict(zip(item.feature_names, item.features))
            row = [provided.get(col) for col in loaded.feature_order]
        else:
            if len(item.features) != len(loaded.feature_order):
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {len(loaded.feature_order)} features, got {len(item.features)}.",
                )
            row = item.features
        rows.append(row)
    return pd.DataFrame(rows, columns=loaded.feature_order)


def create_app(model_dir: Path = Path("models"), cfg_path: Path | None = None) -> FastAPI:
    cfg = PipelineConfig.from_yaml(cfg_path)
    loaded = load_model(model_dir, cfg)

    app = FastAPI(title="EEG Emotional State Classifier", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "classes": loaded.label_encoder.classes_.tolist()}

    @app.post("/predict")
    def predict_endpoint(payload: BatchRequest):
        df = _build_dataframe(payload, loaded)
        preds = predict(df, loaded)
        return preds.to_dict(orient="records")

    return app
