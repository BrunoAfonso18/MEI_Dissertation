# AGENTS.md

## Project Overview

MEI_Dissertation is an academic project implementing **Aspect-Based Sentiment Analysis (ABSA)** with fuzzy logic for Portuguese and English reviews of restaurants. It's a dissertation/Master's project, not a production system.

## Stack

- **Backend**: Python 3.11+, FastAPI, Transformers, Torch, SQLAlchemy, PostgreSQL
- **Frontend**: React 19, TypeScript, Vite (port 3000)
- **Infrastructure**: Docker Compose (PostgreSQL + Backend + Frontend)

## Running the Application

```bash
cd app
docker-compose up --build
```

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- PostgreSQL: localhost:5432 (db: `dw`, user: `password`)

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `app/backend/` | FastAPI API, models, database |
| `app/backend/absa_model_final/` | Trained NER model for aspect extraction |
| `app/absa_module/` | Training scripts, dataset, checkpoints |
| `app/frontend/` | React application |
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