# AGENTS.md

## Project Overview

MEI_Dissertation is an academic project implementing **Aspect-Based Sentiment Analysis (ABSA)** with fuzzy logic for Portuguese and English reviews of restaurants. It's a dissertation/Master's project, not a production system.

## Stack

- **Backend**: Python 3.11+, FastAPI, Transformers, Torch, SQLAlchemy, PostgreSQL
- **Frontend**: Python desktop GUI (customtkinter), not a web app. Runs its own
  copy of the ABSA + fuzzy-sentiment pipeline in-process (no HTTP call to the backend).
- **Infrastructure**: Docker Compose (PostgreSQL + Backend + Frontend). The frontend
  container has no real display, so it runs Xvfb + x11vnc + websockify/novnc internally
  and streams the GUI to a browser via noVNC.

## Running the Application

```bash
cd app
docker-compose up --build
```

- Backend API: http://localhost:8000
- Frontend (noVNC viewer): http://localhost:6080/vnc.html
- PostgreSQL: localhost:5432 (db: `dw`, user: `password`)

Alternatively, run the frontend natively (opens a real desktop window instead of a
browser-streamed one):
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
| `app/frontend/` | Python desktop GUI (customtkinter) — the "smiley" sentiment demo |
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