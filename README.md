# padhai-learning levels prediction and Mistake profiling

Lightweight Flask API and pipeline to run trained prediction models, tag student mistakes, and produce simple profiling reports.

This project:
- transforms raw/source data into model-ready features (DataTransformation),
- runs two models (language & maths) to produce predictions and confidences (PredictPipeline),
- tags letter-level mistakes (DataTagging) and produces profiling summaries (dataprofiling),
- exposes a minimal Flask API (app.py) to run the transformation → prediction flow and save results.

---

## Key functionality

1. Prediction
- Reads transformed test data (artifact/test.csv) produced by the transformation step.
- Loads pre-trained models (`artifact/xgb_lang.pkl`, `artifact/xgb_maths.pkl`) and runs predictions.
- Appends prediction columns to the DataFrame:
  - `el_prediction_lang`, `el_prediction_maths`
  - `el_lang_confidence`, `el_maths_confidence` (if model supports `predict_proba`)
- Saves prediction CSV to `artifact/output/` with a timestamp and returns an API JSON payload including `output_path`.

2. Tagging
- Tagging compares canonical vs transcribed answers and assigns flags (visual/phonetic/decoding/etc).
- Tagging is implemented in `src/components/data_tagging.py` (functions: `tag_letter_level`, `DataTagging.initiate_data_tagging`).
- Tagged output (rows with `letFlag` and `tags`) is saved under `artifact/output/`.

3. Profiling
- `dataprofiling(df_combined)` (in `src/components/data_tagging.py`) accepts a combined DataFrame and returns tag distributions, breakdowns by level and optional wcpm bands, plus CSV exports to `artifact/output/`.

---

## Setup & prerequisites

1. Python & virtual environment (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# for DB driver (development)
pip install psycopg2-binary
```

2. Required artifacts / files
- Create a `.env` file in the project root (see below). The DB credentials must be present before running DataTransformation, because the transformer reads from the DB.
- Place trained model files in `artifact/`:
  - `artifact/xgb_lang.pkl`
  - `artifact/xgb_maths.pkl`

3. .env file is needed to run this code
# or a single SQLAlchemy URL:

```
- Do NOT commit `.env` to VCS. Add it to `.gitignore`.

---

## Running

#### Check the flask port when running the below commands

1. Optional: run data transformation manually to inspect errors
```bash
python3 src/components/data_transformation.py
# This should create artifact/test.csv (or return a DataFrame when called programmatically)
```

2. Start the Flask app
```bash
python3 app.py
# default host/port come from .env or fallback to 0.0.0.0:5002
```

3. API usage
- Health check:
```bash
curl http://127.0.0.1:5002/health
```

- Predict (runs transformation in-process, then prediction):
```bash
curl http://127.0.0.1:5002/predict
# JSON contains lang/math predictions and "output_path" for saved CSV
```

4. Tagging & profiling (programmatic usage)
```bash
curl http://127.0.0.1:5002/tagging
# check artifact output for the profiles.csv
```
- If you want to run tagging or profiling directly from Python (no dedicated endpoint), use snippets:

Tagging (letter-level):
```python
from src.components.data_tagging import DataTagging
dt = DataTagging()
letters_tagged = dt.initiate_data_tagging()   # loads artifact/test.csv if not passed a df
```

Profiling:
```python
from src.components.data_tagging import DataTagging
dt = DataTagging()
df = ...  # df_combined with tags/letFlag
results = dt.dataprofiling(df)
# results contains summary DataFrames and CSVs under artifact/output/ by default
```

---

## Output locations
- Transformed test data: `artifact/test.csv`
- Models: `artifact/xgb_lang.pkl`, `artifact/xgb_maths.pkl`
- Prediction outputs & tagging/profiling CSVs: `artifact/output/` (timestamped filenames)

---

## Troubleshooting (common issues)

- Missing DB credentials / connection errors:
  - Ensure `.env` is present with DB_* variables or `DATABASE_URL`.
  - Verify Postgres is reachable and credentials are correct.

- psycopg2 build error:
  - For development install `psycopg2-binary`:
    ```
    pip install psycopg2-binary
    ```
  - For production build, install `libpq` via Homebrew and add `pg_config` to PATH.

- Missing model files:
  - Place the model `.pkl` files into the `artifact/` directory.

- KeyError for expected columns:
  - Inspect `artifact/test.csv` columns and ensure the transformer produces the expected feature names.

- Prediction length mismatch:
  - Models must receive the exact feature subsets they were trained on. If you see "length mismatch", check X shapes logged by PredictPipeline and the models' output shapes.

- Port in use:
  ```
  lsof -i :5002
  lsof -t -i :5002 | xargs -r kill
  ```

---

## Development notes
- The Flask app runs the data transformation in-process by default for convenience. For production, run transformation as a separate job and point the API to the produced `artifact/test.csv`.
- Logging is available: check the console where the app runs for detailed tracebacks.
- The project supports editable install (`-e .`) for local development.

---

