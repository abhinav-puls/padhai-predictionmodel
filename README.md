# Padhai Learning – Level Prediction and Mistake Profiling

This project provides a lightweight Flask API and data pipeline for predicting student learning levels, tagging mistakes, and generating profiles.

---

## Overview

- Transformation of raw/source data into model-ready features (`DataTransformation`).
- Prediction of **Language** and **Maths** levels with trained models (`PredictPipeline`).
- Tagging of **mistakes** and generation of individual/class profiling summaries (`DataTagging`, `dataprofiling`).
- A Flask-powered API (`app.py`) orchestrating end-to-end data flow: transformation → prediction → tagging → profiling.

---

## Setup

### 1. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install psycopg2-binary # for local PostgreSQL use
```

### 2. Required Files and Directories

- **Model files**: Place trained model `.pkl` files in `artifact/`:

```
artifact/xgb_lang.pkl
artifact/xgb_maths.pkl
```

- **Environment file**: Create a `.env` file in the project root for DB credentials or a SQLAlchemy URL:

```
DATABASE_URL=postgresql://username:password@host:port/dbname
```

> _Do not commit `.env` to version control. Add to `.gitignore`._

---

## Important: Update Table Names Before Running

Review and update all SQL queries to match your local database schema.

- **Where to check**:
  - `src/components/data_transformation.py`
  - Other components/scripts interacting with your DB

- **Example** (update as needed):

```
SELECT * FROM student_responses; -- Replace 'student_responses' with your actual table name
```

---

## Running the Application

### (Optional) Run Data Transformation Manually

```bash
python3 src/components/data_transformation.py
```

_(Generates `artifact/test.csv` for model input.)_

### Start the Flask API

```bash
python3 app.py
```

By default, the API serves at `0.0.0.0:5001` (as configured in `.env`).

---

## API Endpoints

Run endpoints in sequence:

```bash
curl http://127.0.0.1:5001/predict
curl http://127.0.0.1:5001/tagging
curl http://127.0.0.1:5001/classprofiles
```

### `/predict`

- Runs data transformation and model prediction
- Generates:
  - `el_prediction_lang`, `el_prediction_maths`
  - `el_lang_confidence`, `el_maths_confidence`
- Saves outputs to `artifact/output/` (timestamped CSV)

### `/tagging`

- Compares canonical/transcribed answers
- Flags letter-level mistakes (visual, phonetic, decoding, etc.)
- Saves output to `artifact/output/`

### `/classprofiles`

- Generates summaries and tag distributions (by level and word count)

---

## Programmatic Usage

**Tagging:**

```python
from src.components.data_tagging import DataTagging

dt = DataTagging()
tagged_df = dt.initiate_data_tagging()
```

**Profiling:**

```python
results = dt.dataprofiling(tagged_df)
```

Outputs automatically saved in `artifact/output/`.

---

## Output Structure

| Type                                | Location                                                       |
|-------------------------------------|----------------------------------------------------------------|
| Transformed data                    | artifact/test.csv                                              |
| Model files                         | artifact/xgb_lang.pkl, artifact/xgb_maths.pkl                  |
| Prediction, tagging, profiling      | artifact/output/                                               |

---

## Troubleshooting

| Issue                     | Resolution                                                       |
|---------------------------|------------------------------------------------------------------|
| Missing DB credentials    | Verify `.env` contains correct DB variables (`DATABASE_URL`)     |
| psycopg2 build errors     | Install with `pip install psycopg2-binary`                       |
| Missing model files       | Place required `.pkl` files in `artifact/`                       |
| Data shape mismatch       | Ensure transformation outputs match model training features      |
| Port 5001 in use          | Run: `lsof -i :5001` or `lsof -t -i :5001`                      |

---

## Development Notes

- The API runs data transformation in-process by default for convenience.
- For production, separate transformation and point the API to prepared artefacts.
- Logging and tracebacks appear in the console where the app runs.
- Editable installs (`-e .`) in `requirements.txt` are only for local dev; remove if not needed.

---

## Summary Before Running

- **Update all SQL table names to match your DB.**
- **Ensure models and `.env` are correctly placed.**
- Use the endpoints:

```bash
curl http://127.0.0.1:5001/predict
curl http://127.0.0.1:5001/tagging
curl http://127.0.0.1:5001/classprofiles
```

All outputs are stored under `artifact/output/`.

