# AGENTS.md

## Project Overview

MEI_Dissertation is an academic project implementing **Aspect-Based Sentiment Analysis (ABSA)** with fuzzy logic for Portuguese and English reviews of restaurants. It's a dissertation/Master's project, not a production system.

## Stack

- **Backend**: Python 3.11+, FastAPI, Transformers, Torch, SQLAlchemy, PostgreSQL
- **Frontend**: Plotly Dash app using Dash Pages (previously Streamlit,
  previously a customtkinter desktop app streamed via noVNC). The live smiley
  on the "Submeter Review" page runs its own copy of the ABSA + fuzzy-sentiment
  pipeline in-process (no HTTP call to the backend) and reacts keystroke by
  keystroke via a ~300ms `dcc.Interval` poll. Submitting a review or a bulk
  CSV calls the backend API (`POST /query`, `POST /reviews/upload`) so the
  already-processed result is persisted straight into the data warehouse. A
  sidebar (built from `dash.page_registry`, grouped via a custom `category`
  kwarg passed to `dash.register_page`) lists the "Submeter Review" page and
  four read-only dashboard pages (Visão Geral, Categorias, Aspetos &
  Tendência, Restaurantes) that visualize the data warehouse through the
  backend's `/analytics/*` endpoints. Only the active page's layout is
  mounted, so background components (like the smiley's Interval) only run
  while that page is open.
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
| `app/frontend/` | Dash app — the "smiley" sentiment demo, review submission, and dashboard |
| `app/frontend/pages/` | One file per sidebar page (Dash Pages convention, `dash.register_page`) |
| `docs/` | Architecture and design decisions |

## Environment

Environment variables are in `app/backend/.env`:
```
DATABASE_URL=postgresql://user:password@postgres:5432/dw
MODEL_PATH=./absa_model_final
```

## API Endpoints

- `GET /restaurants` - List all registered restaurants
- `POST /query` - Analyze a single review and persist it into the DW
- `POST /reviews/upload` - Bulk CSV upload (`restaurant_id`, `text` columns), persisted into the DW
- `GET /analytics/overview` - KPIs (total reviews, % positive/neutral/negative, avg score)
- `GET /analytics/sentiment-by-category` - Aggregated sentiment by aspect category
- `GET /analytics/top-negative-aspects` - Top negative aspects
- `GET /analytics/sentiment-over-time` - Time-series sentiment
- `GET /analytics/restaurant-performance` - Avg score + review count per restaurant

## Known Limitations

- Model performance is below expectations (mentioned in `docs/decisoes.md`)
- No formal test suite exists

## Documentation

- `docs/arquitetura.md` - System architecture diagram
- `docs/decisoes.md` - Technology choices and model decisions