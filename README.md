# EEG Emotional State Classifier – Backend

It ingests raw EDF recordings, engineers features, trains a classifier, and serves predictions via a FastAPI service.

## Project Layout
- `config/default.yaml` – tunable paths, channels, bands, PCA + model hyperparameters.
- `data/raw/` – place scenario folders with `.edf` files here.
- `data/processed/` – generated feature and PCA datasets (pre-populated with your Colab outputs).
- `models/` – trained artifacts (`scenario_model.pth`, `scaler.joblib`, `label_encoder.joblib`, `feature_order.joblib`, `manifest.json`).
- `src/eeg_emotion/` – production code: ingestion, feature engineering, preprocessing, training, inference API, CLI.
- `notebooks/` – original Colab notebooks for reference.
- `docs/REPORT.pdf` – report.
- `tests/` – lightweight sanity tests.

## Quickstart
1) Install deps (editable for dev):
   ```bash
   pip install -e .  # or pip install -r requirements.txt
   ```
2) (Optional) Adjust settings in `config/default.yaml`.
3) Run the full pipeline (uses defaults unless paths provided):
   ```bash
   eeg-emotion pipeline --cfg config/default.yaml
   ```
   Outputs go to `data/processed/` and `models/`.

## Individual Steps
- **Feature extraction** from EDF:
  ```bash
  eeg-emotion features --raw-dir data/raw --out data/processed/all_participants_features.csv
  ```
- **Preprocess + PCA (winsorization + PCs):**
  ```bash
  eeg-emotion preprocess --features-csv data/processed/all_participants_features.csv --components 20
  ```
- **Train neural net & persist artifacts:**
  ```bash
  eeg-emotion train --data data/processed/final_preprocessed_20.csv --models-dir models
  ```
- **Predict on preprocessed CSV:**
  ```bash
  eeg-emotion predict --data data/processed/final_preprocessed_20.csv --model-dir models --out predictions.csv
  ```

## Inference API
- Start server:
  ```bash
  eeg-emotion serve --model-dir models --cfg config/default.yaml --port 8000
  ```
- POST to `/predict`:
  ```json
  {
    "items": [
      { "features": [/* Brain_PC1..Brain_PC20, BATTERY_mean, BATTERY_std */] }
    ]
  }
  ```
- Health check: `GET /health`.

## Existing Artifacts
- Preprocessed datasets have been moved to `data/processed/`:
  - `all_participants_features.csv`
  - `winsorized_bandpower_features.csv`
  - `final_preprocessed_10.csv`
  - `final_preprocessed_20.csv`
  - `final_preprocessed.csv`
- Trained model artifacts reside in `models/`.