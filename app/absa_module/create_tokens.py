import pandas as pd
import json

df = pd.read_csv("data_prep/absa_reviews_2014/reviews_with_opinions.csv")

def build_bio_dataset(df: pd.DataFrame) -> list[dict]:
    results = []
    
    for (sentence_id, sentence), group in df.groupby(["id", "Sentence"]):
        words = sentence.split()
        labels = ["O"] * len(words)
        
        for _, row in group.iterrows():
            aspect_term   = str(row["Aspect Term"])
            opinion_term  = str(row["Opinion Term"]) if pd.notna(row.get("Opinion Term")) else None

            # Marcar Aspect Term
            aspect_words = aspect_term.split()
            for i in range(len(words) - len(aspect_words) + 1):
                window = [w.lower().strip(".,!?\"'") for w in words[i:i+len(aspect_words)]]
                target = [w.lower().strip(".,!?\"'") for w in aspect_words]
                if window == target:
                    labels[i] = "B-ASP"
                    for j in range(1, len(aspect_words)):
                        labels[i + j] = "I-ASP"
                    break

            if opinion_term and opinion_term != "NULL":
                opinion_words = opinion_term.split()
                for i in range(len(words) - len(opinion_words) + 1):
                    window = [w.lower().strip(".,!?\"'") for w in words[i:i+len(opinion_words)]]
                    target = [w.lower().strip(".,!?\"'") for w in opinion_words]
                    if window == target:
                        if all(labels[i+j] == "O" for j in range(len(opinion_words))):
                            labels[i] = "B-OPN"
                            for j in range(1, len(opinion_words)):
                                labels[i + j] = "I-OPN"
                        break

        results.append({
            "id":       sentence_id,
            "sentence": sentence,
            "tokens":   words,
            "labels":   labels,
            "polarities": group[["Aspect Term", "polarity"]].to_dict(orient="records")
        })
    
    return results


bio_data = build_bio_dataset(df)

# Estatísticas
total     = len(bio_data)
has_opn   = sum(1 for item in bio_data if "B-OPN" in item["labels"])
has_asp   = sum(1 for item in bio_data if "B-ASP" in item["labels"])
print(f"\n Total de frases:              {total}")
print(f"   Com aspect term marcado:      {has_asp} ({has_asp/total*100:.1f}%)")
print(f"   Com opinion term marcado:     {has_opn} ({has_opn/total*100:.1f}%)")
print(f"   Só aspeto (sem opinião):      {total - has_opn} ({(total-has_opn)/total*100:.1f}%)")

with open("bio_dataset.json", "w", encoding="utf-8") as f:
    json.dump(bio_data, f, ensure_ascii=False, indent=2)