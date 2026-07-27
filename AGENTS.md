# AGENTS.md

## Project Overview

MEI_Dissertation is an academic project implementing **Aspect-Based Sentiment Analysis (ABSA)** with fuzzy logic for Portuguese and English reviews of restaurants. It's a dissertation/Master's project, not a production system.

## Stack

- **Backend**: Python 3.11+, FastAPI, Transformers, Torch, SQLAlchemy, PostgreSQL
- **Frontend**: Streamlit multi-page app (previously Plotly Dash, previously a
  customtkinter desktop app streamed via noVNC). The live smiley on the
  "Submeter Review" page still runs its own copy of the ABSA + fuzzy-sentiment
  pipeline in-process (no HTTP call to the backend), cached with
  `st.cache_resource` so the model loads once per server process; it updates
  whenever the review textbox loses focus or Ctrl+Enter is pressed (Streamlit
  reruns the script on interaction rather than pushing live updates per
  keystroke). Submitting a review or a bulk CSV calls the backend API
  (`POST /query`, `POST /reviews/upload`) so the already-processed result is
  persisted straight into the data warehouse. A sidebar (via `st.navigation`)
  groups the "Submeter Review" page and four read-only dashboard pages
  (Visão Geral, Categorias, Aspetos & Tendência, Restaurantes) that visualize
  the data warehouse through the backend's `/analytics/*` endpoints.
- **Infrastructure**: Docker Compose (PostgreSQL + Backend + Frontend).

## Running the Application

```bash
cd app
docker-compose up --build
```

- Backend API: http://localhost:8000
- Frontend (Streamlit): http://localhost:8050
- PostgreSQL: localhost:5432 (db: `dw`, user: `password`)

Alternatively, run the frontend natively:
```bash
.\venv\Scripts\activate.bat
streamlit run app/frontend/streamlit_app.py
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `app/backend/` | FastAPI API, models, database |
| `app/backend/absa_model_final/` | Trained NER model for aspect extraction |
| `app/absa_module/` | Training scripts, dataset, checkpoints |
| `app/frontend/` | Streamlit multi-page app — the "smiley" sentiment demo, review submission, and dashboard |
| `app/frontend/pages/` | One file per sidebar page (Streamlit's multi-page convention) |
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