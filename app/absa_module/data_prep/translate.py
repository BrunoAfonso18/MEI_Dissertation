import re
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from pathlib import Path
import time

input_file = Path("./absa_reviews_2014/reviews_with_opinions.csv")
output_file = Path("./absa_reviews_2014/reviews_pt.csv")

columns_to_translate = ["Sentence", "Aspect Term", "Opinion Term"]

df = pd.read_csv(input_file, encoding="utf-8", low_memory=False)

missing = [col for col in columns_to_translate if col not in df.columns]
if missing:
    print(f"ERRO: As seguintes colunas não foram encontradas: {missing}")
    exit()

# Usa-se um modelo de tradução local (Helsinki-NLP/
# opus-mt-tc-big-en-pt via Transformers/MarianMT), traduzido em lotes -
# chamada a chamada (como o original) seriam ~2h só para as ~6000 linhas;
# em lotes de 32 com decodificação greedy (sem pesquisa em feixe, ~14x mais
# rápida), demora ~20-30 min.
print("A carregar o modelo de tradução (Helsinki-NLP/opus-mt-tc-big-en-pt)...")
_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
_model.eval()
_BATCH_SIZE = 32


def _translate_batch(texts: list[str]) -> list[str]:
    out = []
    for i in tqdm(range(0, len(texts), _BATCH_SIZE), desc="  a traduzir"):
        chunk = texts[i:i + _BATCH_SIZE]
        batch = _tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            gen = _model.generate(**batch, max_new_tokens=128, num_beams=1, do_sample=False)
        out.extend(_tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


# Traduzir um termo curto ISOLADO (ex: "food") produz sistematicamente uma
# tradução diferente da que aparece dentro da frase completa ("food" ->
# "Géneros alimentícios" isolado, mas sempre "comida" em contexto). Para os
# termos de aspeto/opinião (frases curtas), traduz-se também uma versão do
# termo dentro de uma frase-veículo fixa ("The X was good."), de onde se
# extrai a forma nominal correta por expressão regular - a frase completa
# usa sempre a tradução direta.
_CARRIER = "The {term} was good."
_CARRIER_RE = re.compile(r"^\s*(?:a|o|as|os)\s+(.+?)\s+(?:foi|foram|era|eram)\s+(?:bo[amns]?s?)\.?\s*$", re.IGNORECASE)


for col in columns_to_translate:
    new_col_name = f"{col}_pt"
    print(f"\n A traduzir coluna: '{col}' → '{new_col_name}'")

    values = df[col].tolist()
    idx = [i for i, v in enumerate(values) if isinstance(v, str) and v.strip()]
    texts = [values[i] for i in idx]

    if col == "Sentence":
        translated = _translate_batch(texts)
    else:
        bare = _translate_batch(texts)
        carried = _translate_batch([_CARRIER.format(term=t) for t in texts])
        translated = []
        for b, c in zip(bare, carried):
            m = _CARRIER_RE.match(c)
            translated.append(m.group(1).strip() if m else b)

    out = list(values)
    for i, t in zip(idx, translated):
        out[i] = t
    df[new_col_name] = out

df.to_csv(output_file, index=False, encoding="utf-8")

print("    TRADUÇÃO DAS COLUNAS CONCLUÍDA COM SUCESSO!")