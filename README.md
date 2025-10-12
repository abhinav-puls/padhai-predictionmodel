# padhai-predictionmodel

Lightweight Flask API and pipeline for running a trained prediction model that produces language and maths predictions for student records.

## Project overview
- DataTransformation: reads raw/source data, transforms it into model-ready features and writes `artifact/test.csv` (and other artifact CSVs).
- PredictPipeline: loads pre-trained models (`artifact/xgb_lang.pkl`, `artifact/xgb_maths.pkl`), runs predictions and attaches `el_prediction_lang` and `el_prediction_maths` to the DataFrame.
- app.py: Flask app with `/predict` endpoint — runs transformation (in-process), invokes the prediction pipeline, saves the prediction CSV under `artifact/output/` and returns JSON results.

## Prerequisites
- Python 3.10+ (match your virtualenv/python used to create the model artifacts)
- Virtual environment recommended

## Install
From the project root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Note: `requirements.txt` contains `-e .` (editable install). Ensure `setup.py` or `pyproject.toml` exists if you keep `-e .`.

## Important files / expected artifacts
- `src/components/data_transformation.py` — data transformation logic; expected to write `artifact/test.csv`.
- `src/components/data_prediction.py` — prediction pipeline that returns a DataFrame with `el_prediction_lang` and `el_prediction_maths`.
- `artifact/xgb_lang.pkl`, `artifact/xgb_maths.pkl` — serialized trained models (required).
- `artifact/test.csv` — test dataset produced by the transformer (or returned by the transformer method).
- `artifact/output/` — prediction CSVs are saved here by the API.

If model `.pkl` files are missing, create/restore them or retrain with your training script.

## Running the pipeline and API

1. (Optional) Run transformation manually to inspect errors:
```bash
python3 src/components/data_transformation.py
# This should create artifact/test.csv and other artifacts
```

2. Start the Flask API:
```bash
python3 app.py
```
By default it runs on `http://127.0.0.1:5000/`.

3. Call the endpoint:
```bash
curl -s http://127.0.0.1:5000/predict | jq .
```
Response includes prediction lists and `output_path` of the saved CSV.

If port 5000 is in use, either free the port or run the app on another port (modify `app.run(...)` or start programmatically).

## Example: use PredictPipeline directly (for debugging)
```bash
python3 - <<'PY'
import pandas as pd
from src.components.data_prediction import PredictPipeline
df = pd.read_csv("artifact/test.csv")
pred_df = PredictPipeline().predict(df)
print(pred_df[['el_prediction_lang','el_prediction_maths']].head())
PY
```

## Troubleshooting
- "No such file or directory: 'artifact/xgb_maths.pkl'": ensure both model files are in `artifact/`.
- "DataFrame constructor not properly called!": inspect what `DataTransformation.initiate_data_transformation()` returns — app expects a DataFrame or path to CSV; transformer should write `artifact/test.csv`.
- If push to GitHub is rejected: pull/rebase remote changes, resolve conflicts, then push.
- If Flask reports the port is in use: find and kill the process (example):
```bash
lsof -i :5000
lsof -t -i :5000 | xargs -r kill
```

## Development notes
- Logging is implemented in the code; check console output for detailed errors.
- The API runs the transformation in-process for convenience; in production you may want to run transformation as a separate batch job and point the API directly to the resulting `artifact/test.csv`.
- Editable install (`-e .`) allows immediate use of local changes without reinstalling.

