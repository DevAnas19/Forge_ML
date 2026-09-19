# ForgeML

Intelligent ML Experimentation & Deployment Platform — a full-stack app that
takes a user from a raw CSV to a trained, explained, and deployable ML model.

This repo currently holds **Phase 1**: dataset upload, profiling, and
structural validation. Everything else (training, tracking, registry, SHAP,
frontend, LLM assistant, Docker) comes in later phases — see the roadmap at
the bottom.

---

## What's implemented so far (Steps 1-8)

- `backend/app/services/profiler.py` — profiles a dataset: row/column counts,
  missing values, duplicates, numerical vs categorical column split, and
  per-column statistics (mean/median/std for numeric, frequency counts for
  categorical).
- `backend/app/services/validator.py` — validates a dataset: duplicate rows,
  empty columns, constant columns, high-cardinality categorical columns
  (structural checks), plus missing-target-value and class-imbalance checks
  once a target column is chosen.
- `backend/app/main.py` — a FastAPI app exposing `POST /api/datasets/upload`,
  which accepts a CSV and returns the profile + validation result as JSON.

No database yet — results are computed and returned directly. Persistence
comes in Phase 3.

---

## Project setup

### 1. Prerequisites

- Python 3.10+ installed
- A terminal / VS Code

### 2. Set up the backend

```bash
cd backend
python -m venv venv

# activate the virtual environment
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

Run this from inside `backend/` (not `backend/app/`), so Python can find the
`app` package correctly.

### 4. Try it out

Open **http://localhost:8000/docs** — FastAPI's auto-generated interactive
API page. Expand `POST /api/datasets/upload`, click "Try it out", upload any
CSV file (e.g. the Kaggle Loan Prediction `train.csv`), and click Execute.
You should get back a JSON response with `profile` and `validation` keys.

---

## Project structure

```
forgeML/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + routes (Phase 1: upload endpoint)
│   │   ├── api/                 # future: route files split out as the app grows
│   │   ├── services/
│   │   │   ├── profiler.py      # dataset profiling logic
│   │   │   └── validator.py     # dataset validation logic
│   │   ├── ml/                  # future: preprocessing, classification, regression
│   │   ├── models/              # future: SQLAlchemy DB table definitions
│   │   └── core/                # future: config.py, database.py
│   ├── artifacts/                # future: saved model .pkl files
│   └── requirements.txt
├── frontend/                     # not started yet — Phase 6
├── docker-compose.yml             # not started yet — Phase 8
└── README.md
```

`app/models/` is for **database tables**, not ML models — that naming comes
from the original spec, worth remembering since it reads oddly at first.

---

## Roadmap (from the original spec)

| Phase | What | Status |
|---|---|---|
| 1 | Dataset upload + profiling + structural validation | ✅ Done |
| 2 | Preprocessing + model training | Not started |
| 3 | Experiment tracking + PostgreSQL | Not started |
| 4 | Model registry + prediction API | Not started |
| 5 | SHAP explainability | Not started |
| 6 | Next.js dashboard | Not started |
| 7 | LLM experiment planner | Not started |
| 8 | Docker + tests + deployment | Not started |
