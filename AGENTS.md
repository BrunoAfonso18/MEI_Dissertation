# AGENTS.md

## Project Overview

MEI_Dissertation is an academic project implementing **Aspect-Based Sentiment Analysis (ABSA)** with fuzzy logic for Portuguese and English reviews of restaurants. It's a dissertation/Master's project, not a production system.

## Stack

- **Backend**: Python 3.11+, FastAPI, Transformers, Torch, SQLAlchemy, PostgreSQL
- **Frontend**: Plotly Dash web app. The live smiley still runs its own copy of the
  ABSA + fuzzy-sentiment pipeline in-process (no HTTP call to the backend, unchanged
  interaction/latency); submitting a review or a bulk CSV calls the backend API
  (`POST /query`, `POST /reviews/upload`) so the already-processed result is
  persisted straight into the data warehouse.
- **Infrastructure**: Docker Compose (PostgreSQL + Backend + Frontend).

## Running the Application

```bash
cd app
docker-compose up --build
```

- Backend API: http://localhost:8000
- Frontend (Dash): http://localhost:8050
- PostgreSQL: localhost:5432 (db: `dw`, user: `password`)

Alternatively, run the frontend natively:
```bash
.\venv\Scripts\activate.bat
python app/frontend/app.py
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `app/backend/` | FastAPI API, models, database |
| `app/backend/absa_model_final/` | Trained NER model for aspect extraction |
| `app/absa_module/` | Training scripts, dataset, checkpoints |
| `app/frontend/` | Plotly Dash web app — the "smiley" sentiment demo + review submission |
| `docs/` | Architecture and design decisions |

## Environment

Environment variables are in `app/backend/.env`:
```
DATABASE_URL=postgresql://user:password@postgres:5432/dw
MODEL_PATH=./absa_model_final
```

## API Endpoints

- `POST /analyze` - Analyze a review text
- `GET /analytics/sentiment-by-category` - Aggregated sentiment
- `GET /analytics/top-negative-aspects` - Top negative aspects
- `GET /analytics/sentiment-over-time` - Time-series sentiment

## Known Limitations

- Model performance is below expectations (mentioned in `docs/decisoes.md`)
- No formal test suite exists

## Documentation

- `docs/arquitetura.md` - System architecture diagram
- `docs/decisoes.md` - Technology choices and model decisions