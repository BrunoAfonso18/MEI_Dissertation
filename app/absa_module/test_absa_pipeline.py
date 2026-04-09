from absa_pipeline import processar_review
import json
from pathlib import Path
from datetime import datetime

REVIEWS_TXT_PATH = Path("../../data/raw/reviews_pt_sint.txt")
OUTPUT_DIR = Path("../../data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"absa_results_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def run_batch_test():
    with open(REVIEWS_TXT_PATH, "r", encoding="utf-8") as f:
        reviews = [linha.strip() for linha in f if linha.strip()]

    print(f"{len(reviews)} reviews carregadas\n")

    resultados = []
    for i, texto in enumerate(reviews, 1):
        resultado = processar_review(texto, f"rev_{i:04d}")
        resultados.append(resultado)

        gl = resultado["overall_sentiment"]
        print(f"[{i:03d}/{len(reviews)}] → {resultado['num_aspects']} aspetos | Global score: {gl['score']:.3f}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_reviews": len(reviews),
                "processed_at": datetime.now().isoformat(),
                "version": "3.1"
            },
            "results": resultados
        }, f, indent=2, ensure_ascii=False)

    print(f"\nConcluído! Resultados guardados em:\n{OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    run_batch_test()