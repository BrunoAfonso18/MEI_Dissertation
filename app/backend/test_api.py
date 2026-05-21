"""
Script de teste dos endpoints da API ABSA.
Corre com: python test_api.py

Pré-requisitos:
  1. docker compose up   (ou uvicorn main:app --reload na pasta backend/)
  2. python seed_restaurants.py   (para criar os restaurantes na DB)
"""

import json
import sys
import requests

BASE = "http://localhost:8000"


def ok(label: str, resp: requests.Response):
    status = "OK" if resp.ok else "FAIL"
    print(f"[{status}] {label} -> HTTP {resp.status_code}")
    if resp.ok:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2)[:800])
    else:
        print("  ERRO:", resp.text[:300])
    print()


# ── 1. Health check ───────────────────────────────────────────────
def test_health():
    ok("GET /health", requests.get(f"{BASE}/health"))


# ── 2. Listar restaurantes ────────────────────────────────────────
def test_restaurants():
    ok("GET /restaurants", requests.get(f"{BASE}/restaurants"))


# ── 3. Query simples (texto + restaurant_id) ──────────────────────
def test_query():
    payload = {
        "text": "A comida estava deliciosa mas o serviço foi muito lento.",
        "restaurant_id": 1
    }
    ok("POST /query", requests.post(f"{BASE}/query", json=payload))


def test_query_negative():
    payload = {
        "text": "O ambiente era barulhento e o prato principal chegou frio.",
        "restaurant_id": 4
    }
    ok("POST /query (negativo)", requests.post(f"{BASE}/query", json=payload))


# ── 4. Upload CSV em bulk ─────────────────────────────────────────
def test_bulk_upload():
    csv_content = (
        "restaurant_id,text\n"
        '1,"A comida estava excelente e o ambiente muito acolhedor."\n'
        '2,"A pizza estava fria e o serviço foi péssimo."\n'
        '3,"O sushi estava fresco. Recomendo vivamente."\n'
        '4,"Porções generosas a preço acessível. Voltarei com certeza."\n'
        '99,"Este restaurante não existe."\n'  # deve falhar (restaurant 99 not found)
    )
    files = {"file": ("reviews.csv", csv_content.encode("utf-8"), "text/csv")}
    ok("POST /reviews/upload", requests.post(f"{BASE}/reviews/upload", files=files))


# ── 5. Analytics ──────────────────────────────────────────────────
def test_analytics():
    ok("GET /analytics/sentiment-by-category",
       requests.get(f"{BASE}/analytics/sentiment-by-category"))
    ok("GET /analytics/top-negative-aspects",
       requests.get(f"{BASE}/analytics/top-negative-aspects?limit=5"))
    ok("GET /analytics/sentiment-over-time",
       requests.get(f"{BASE}/analytics/sentiment-over-time"))


# ── 6. Erro esperado: restaurante inexistente ─────────────────────
def test_invalid_restaurant():
    payload = {"text": "Ótima experiência!", "restaurant_id": 9999}
    ok("POST /query (restaurante inexistente)", requests.post(f"{BASE}/query", json=payload))


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"A testar API em {BASE}\n{'='*50}\n")

    try:
        requests.get(f"{BASE}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"ERRO: Servidor nao esta a correr em {BASE}")
        print("Inicia com: docker compose up")
        print("       ou: uvicorn main:app --reload  (na pasta backend/)")
        sys.exit(1)

    test_health()
    test_restaurants()
    test_query()
    test_query_negative()
    test_bulk_upload()
    test_analytics()
    test_invalid_restaurant()

    print("Testes concluidos.")
