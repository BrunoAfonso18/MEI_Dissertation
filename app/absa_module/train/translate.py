import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
from pathlib import Path
import time

input_file = Path("./absa_reviews_2014/reviews.csv")
output_file = Path("./absa_reviews_2014/reviews_pt.csv")

columns_to_translate = ["Sentence", "Aspect Term"]

df = pd.read_csv(input_file, encoding="utf-8", low_memory=False)

missing = [col for col in columns_to_translate if col not in df.columns]
if missing:
    print(f"ERRO: As seguintes colunas não foram encontradas: {missing}")
    exit()

def translate_text(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return text
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            translated = GoogleTranslator(source='auto', target='pt').translate(text)
            return translated
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[TRADUÇÃO_FALHOU] {str(text)[:150]}"
            time.sleep(1.2)

tqdm.pandas(desc="Traduzindo para PT")

for col in columns_to_translate:
    new_col_name = f"{col}_pt"
    print(f"\n→ A traduzir coluna: '{col}' → '{new_col_name}'")
    
    df[new_col_name] = df[col].progress_apply(translate_text)
    
    time.sleep(2)

df.to_csv(output_file, index=False, encoding="utf-8")

print("    TRADUÇÃO DUAS COLUNAS CONCLUÍDA COM SUCESSO!")