FROM python:3.11-slim

WORKDIR /app/frontend

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .

EXPOSE 8050

CMD ["python", "app.py"]
