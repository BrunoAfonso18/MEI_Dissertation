FROM python:3.11-slim

WORKDIR /app/frontend

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .

EXPOSE 8050

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8050", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
