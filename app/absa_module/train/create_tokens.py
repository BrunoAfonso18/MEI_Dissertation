import pandas as pd

df = pd.read_csv("absa_reviews_2014/reviews_pt.csv")

def convert_to_bio(sentence: str, aspect_term: str, from_idx: int, to_idx: int) -> dict:
    tokens = []
    labels = []
    
    i = 0
    token_start = 0
    
    words = sentence.split()
    char_pos = 0
    
    for word in words:
        # Encontrar posição real desta palavra na frase
        word_start = sentence.find(word, char_pos)
        word_end = word_start + len(word)
        
        tokens.append(word)
        
        # Determinar label baseado nos offsets
        if word_start >= from_idx and word_end <= to_idx:
            if word_start == from_idx:
                labels.append("B-ASP")
            else:
                labels.append("I-ASP")
        else:
            labels.append("O")
        
        char_pos = word_end
    
    return {"tokens": tokens, "labels": labels}


def build_bio_dataset(df: pd.DataFrame, use_portuguese: bool = True) -> list[dict]:
    sentence_col = "Sentence_pt" if use_portuguese else "Sentence"
    aspect_col   = "Aspect Term_pt" if use_portuguese else "Aspect Term"
    
    results = []
    
    for sentence_id, group in df.groupby("id"):
        sentence = group[sentence_col].iloc[0]
        words = sentence.split()
        labels = ["O"] * len(words)
        
        # Usamos string matching no texto PT com o aspect term PT
        for _, row in group.iterrows():
            aspect_pt = str(row[aspect_col])
            aspect_words = aspect_pt.split()
            
            # Procurar o aspeto PT na frase PT
            for i in range(len(words) - len(aspect_words) + 1):
                window = [w.lower().strip(".,!?\"'") for w in words[i:i+len(aspect_words)]]
                target = [w.lower().strip(".,!?\"'") for w in aspect_words]
                
                if window == target:
                    labels[i] = "B-ASP"
                    for j in range(1, len(aspect_words)):
                        labels[i + j] = "I-ASP"
                    break
        
        results.append({
            "id":       sentence_id,
            "sentence": sentence,
            "tokens":   words,
            "labels":   labels,
            "polarities": group[["Aspect Term_pt" if use_portuguese else "Aspect Term", "polarity"]]
                            .to_dict(orient="records")
        })
    
    return results



bio_data = build_bio_dataset(df, use_portuguese=True)

# Verificar resultado
for item in bio_data[:3]:
    print(f"\n {item['sentence'][:60]}...")
    print(f"   Polarities: {item['polarities']}")
    for token, label in zip(item["tokens"], item["labels"]):
        if label != "O":
            print(f"   {token:20} → {label}")

# Guardar
import json
with open("bio_dataset.json", "w", encoding="utf-8") as f:
    json.dump(bio_data, f, ensure_ascii=False, indent=2)