"""
Gera reviews sintéticas em português, anotadas diretamente ao nível de
token (sem tradução, sem o modelo a classificar nada) - as etiquetas BIO
são calculadas a partir dos próprios blocos de texto usados para construir
cada frase, portanto são garantidamente corretas por construção, não
inferidas.

Cada review combina 2 a 4 pares (termo de aspeto, termo de opinião) de
listas com marcação de género gramatical, para que a concordância com o
verbo funcione, ligados por conectores ("e"/"mas", conforme a polaridade
muda ou não de uma cláusula para a seguinte). As listas de aspetos e de
opiniões incluem deliberadamente uma proporção elevada de termos com mais
de uma palavra (ex. "tempo de espera", "relação qualidade-preço",
"extremamente lento", "nada saboroso"), para reforçar exatamente os dois
padrões que o modelo tem mais dificuldade em aprender.

Corre-se depois de create_tokens.py (que gera a base em inglês a partir do
SemEval) e antes de train.py:
    python -m absa_module.data_prep.generate_pt_reviews
"""

import json
import random
import re
from pathlib import Path

# ── Vocabulário ──────────────────────────────────────────────────────
#
# Cada aspeto: (frase, género 'm'/'f', categoria - só para referência/
# rastreabilidade, não é usada pelo treino). Cada opinião: (frase_m,
# frase_f, polaridade) - frase_m == frase_f nos adjetivos invariáveis
# (ex. "agradável").

ASPECTS = [
    ("comida", "f", "FOOD_QUALITY"),
    ("prato", "m", "FOOD_QUALITY"),
    ("sabor", "m", "FOOD_QUALITY"),
    ("tempero", "m", "FOOD_QUALITY"),
    ("sobremesa", "f", "FOOD_QUALITY"),
    ("entrada", "f", "FOOD_QUALITY"),
    ("massa", "f", "FOOD_QUALITY"),
    ("peixe", "m", "FOOD_QUALITY"),
    ("carne", "f", "FOOD_QUALITY"),
    ("sopa", "f", "FOOD_QUALITY"),
    ("prato principal", "m", "FOOD_QUALITY"),
    ("qualidade da comida", "f", "FOOD_QUALITY"),
    ("tempero dos pratos", "m", "FOOD_QUALITY"),
    ("ponto de cozedura da carne", "m", "FOOD_QUALITY"),
    ("frescura do peixe", "f", "FOOD_QUALITY"),
    ("serviço", "m", "SERVICE"),
    ("atendimento", "m", "SERVICE"),
    ("empregado", "m", "SERVICE"),
    ("simpatia da equipa", "f", "SERVICE"),
    ("rapidez do serviço", "f", "SERVICE"),
    ("atendimento ao cliente", "m", "SERVICE"),
    ("profissionalismo da equipa", "m", "SERVICE"),
    ("simpatia do empregado", "f", "SERVICE"),
    ("preço", "m", "PRICE_AND_VALUE"),
    ("conta", "f", "PRICE_AND_VALUE"),
    ("relação qualidade-preço", "f", "PRICE_AND_VALUE"),
    ("preço final", "m", "PRICE_AND_VALUE"),
    ("preço das bebidas", "m", "PRICE_AND_VALUE"),
    ("ambiente", "m", "AMBIENCE_AND_ATMOSPHERE"),
    ("decoração", "f", "AMBIENCE_AND_ATMOSPHERE"),
    ("música", "f", "AMBIENCE_AND_ATMOSPHERE"),
    ("iluminação", "f", "AMBIENCE_AND_ATMOSPHERE"),
    ("ambiente do restaurante", "m", "AMBIENCE_AND_ATMOSPHERE"),
    ("nível de ruído", "m", "AMBIENCE_AND_ATMOSPHERE"),
    ("temperatura da sala", "f", "AMBIENCE_AND_ATMOSPHERE"),
    ("localização", "f", "LOCATION"),
    ("estacionamento", "m", "LOCATION"),
    ("acesso de transportes públicos", "m", "LOCATION"),
    ("vista para o rio", "f", "LOCATION"),
    ("dose", "f", "PORTION_SIZE"),
    ("tamanho da dose", "m", "PORTION_SIZE"),
    ("quantidade de comida", "f", "PORTION_SIZE"),
    ("menu", "m", "MENU_VARIETY"),
    ("variedade do menu", "f", "MENU_VARIETY"),
    ("carta de vinhos", "f", "MENU_VARIETY"),
    ("opções vegetarianas", "f", "MENU_VARIETY"),
    ("limpeza", "f", "CLEANLINESS"),
    ("casa de banho", "f", "CLEANLINESS"),
    ("higiene da cozinha", "f", "CLEANLINESS"),
    ("espera", "f", "WAITING_TIME"),
    ("tempo de espera", "m", "WAITING_TIME"),
    ("fila de espera", "f", "WAITING_TIME"),
    ("demora da conta", "f", "WAITING_TIME"),
]

# Opiniões agrupadas por categoria, para não emparelhar ao acaso um termo
# de opinião com um aspeto semanticamente incompatível (ex: "fila de
# espera estava deliciosa" - sintaticamente válido, mas sem sentido). Cada
# aspeto usa a pool GENERAL (adjetivos amplos, aplicáveis a quase tudo)
# mais a pool específica da sua própria categoria.
# Formato: (frase_masc, frase_fem, polaridade); adjetivo invariável tem as
# duas formas iguais.
GENERAL = [
    ("bom", "boa", "positive"),
    ("ótimo", "ótima", "positive"),
    ("excelente", "excelente", "positive"),
    ("impecável", "impecável", "positive"),
    ("muito bom", "muito boa", "positive"),
    ("surpreendentemente bom", "surpreendentemente boa", "positive"),
    ("mau", "má", "negative"),
    ("péssimo", "péssima", "negative"),
    ("horrível", "horrível", "negative"),
    ("desagradável", "desagradável", "negative"),
    ("muito mau", "muito má", "negative"),
    ("longe do esperado", "longe do esperado", "negative"),
    ("razoável", "razoável", "neutral"),
    ("aceitável", "aceitável", "neutral"),
    ("normal", "normal", "neutral"),
    ("dentro do esperado", "dentro do esperado", "neutral"),
    ("relativamente aceitável", "relativamente aceitável", "neutral"),
]

OPINIONS_BY_CATEGORY = {
    "FOOD_QUALITY": [
        ("delicioso", "deliciosa", "positive"),
        ("saboroso", "saborosa", "positive"),
        ("fresco", "fresca", "positive"),
        ("bem temperado", "bem temperada", "positive"),
        ("verdadeiramente delicioso", "verdadeiramente deliciosa", "positive"),
        ("insosso", "insossa", "negative"),
        ("queimado", "queimada", "negative"),
        ("requentado", "requentada", "negative"),
        ("mal cozinhado", "mal cozinhada", "negative"),
        ("nada saboroso", "nada saborosa", "negative"),
    ],
    "SERVICE": [
        ("simpático", "simpática", "positive"),
        ("atencioso", "atenciosa", "positive"),
        ("prestável", "prestável", "positive"),
        ("extremamente simpático", "extremamente simpática", "positive"),
        ("super atencioso", "super atenciosa", "positive"),
        ("grosseiro", "grosseira", "negative"),
        ("mal-educado", "mal-educada", "negative"),
        ("pouco profissional", "pouco profissional", "negative"),
        ("pouco simpático", "pouco simpática", "negative"),
    ],
    "PRICE_AND_VALUE": [
        ("justo", "justa", "positive"),
        ("em conta", "em conta", "positive"),
        ("muito em conta", "muito em conta", "positive"),
        ("caro", "cara", "negative"),
        ("exagerado", "exagerada", "negative"),
        ("bastante caro", "bastante cara", "negative"),
        ("demasiado caro", "demasiado cara", "negative"),
    ],
    "AMBIENCE_AND_ATMOSPHERE": [
        ("acolhedor", "acolhedora", "positive"),
        ("confortável", "confortável", "positive"),
        ("romântico", "romântica", "positive"),
        ("muito acolhedor", "muito acolhedora", "positive"),
        ("ruidoso", "ruidosa", "negative"),
        ("sobrelotado", "sobrelotada", "negative"),
        ("extremamente ruidoso", "extremamente ruidosa", "negative"),
        ("completamente frio", "completamente fria", "negative"),
    ],
    "LOCATION": [
        ("central", "central", "positive"),
        ("acessível", "acessível", "positive"),
        ("bem localizado", "bem localizada", "positive"),
        ("afastado", "afastada", "negative"),
        ("mal localizado", "mal localizada", "negative"),
        ("difícil de encontrar", "difícil de encontrar", "negative"),
    ],
    "PORTION_SIZE": [
        ("generoso", "generosa", "positive"),
        ("abundante", "abundante", "positive"),
        ("bastante generoso", "bastante generosa", "positive"),
        ("escasso", "escassa", "negative"),
        ("insuficiente", "insuficiente", "negative"),
        ("muito escasso", "muito escassa", "negative"),
    ],
    "MENU_VARIETY": [
        ("variado", "variada", "positive"),
        ("extenso", "extensa", "positive"),
        ("bastante variado", "bastante variada", "positive"),
        ("limitado", "limitada", "negative"),
        ("pouco variado", "pouco variada", "negative"),
        ("bastante limitado", "bastante limitada", "negative"),
    ],
    "CLEANLINESS": [
        ("limpo", "limpa", "positive"),
        ("higiénico", "higiénica", "positive"),
        ("impecavelmente limpo", "impecavelmente limpa", "positive"),
        ("sujo", "suja", "negative"),
        ("pouco higiénico", "pouco higiénica", "negative"),
    ],
    "WAITING_TIME": [
        ("rápido", "rápida", "positive"),
        ("ágil", "ágil", "positive"),
        ("bastante rápido", "bastante rápida", "positive"),
        ("lento", "lenta", "negative"),
        ("demorado", "demorada", "negative"),
        ("extremamente lento", "extremamente lenta", "negative"),
        ("muito demorado", "muito demorada", "negative"),
    ],
}

VERBS = ["estava", "foi", "é"]
CONNECT_SAME = [", ", " e "]
CONNECT_SWITCH = [", mas ", " mas "]

_NORM_RE = re.compile(r"[^\w\-à-úÀ-Ú]", re.UNICODE)


def _norm(w: str) -> str:
    return _NORM_RE.sub("", w.lower())


def _find_span(tokens: list[str], phrase: str) -> tuple[int, int]:
    parts = phrase.split()
    target = [_norm(p) for p in parts]
    for i in range(len(tokens) - len(parts) + 1):
        if [_norm(tokens[i + j]) for j in range(len(parts))] == target:
            return i, i + len(parts)
    raise ValueError(f"Span nao encontrado: {phrase!r} em {tokens}")


POLARITY_CODE = {"positive": "pos", "negative": "neg", "neutral": "neu"}


def _clause(aspect: str, opinion_m: str, opinion_f: str, gender: str, verb: str, capitalize: bool) -> str:
    opinion = opinion_m if gender == "m" else opinion_f
    text = f"{aspect} {verb} {opinion}"
    if capitalize:
        text = text[0].upper() + text[1:]
    return text


def generate_example(rng: random.Random, idx: int, n_aspects: int) -> dict:
    chosen_aspects = rng.sample(ASPECTS, n_aspects)
    clauses = []
    polarities = []
    aspect_phrases = []
    opinion_phrases = []
    prev_polarity = None

    for i, (aspect, gender, category) in enumerate(chosen_aspects):
        pool = OPINIONS_BY_CATEGORY.get(category, []) + GENERAL
        opinion_m, opinion_f, polarity = rng.choice(pool)
        verb = rng.choice(VERBS)
        clause = _clause(aspect, opinion_m, opinion_f, gender, verb, capitalize=(i == 0))
        opinion = opinion_m if gender == "m" else opinion_f

        if i > 0:
            connector = rng.choice(CONNECT_SWITCH if polarity != prev_polarity else CONNECT_SAME)
            clauses.append(connector + clause)
        else:
            clauses.append(clause)

        polarities.append({"Aspect Term": aspect, "polarity": polarity, "category": category})
        aspect_phrases.append((aspect, polarity))
        opinion_phrases.append(opinion)
        prev_polarity = polarity

    sentence = "".join(clauses) + "."
    tokens = sentence.split()
    labels = ["O"] * len(tokens)

    for (aspect, polarity), opinion in zip(aspect_phrases, opinion_phrases):
        a_start, a_end = _find_span(tokens, aspect)
        if any(labels[i] != "O" for i in range(a_start, a_end)):
            raise ValueError(f"Sobreposicao de aspeto em: {sentence!r}")
        pol_code = POLARITY_CODE[polarity]
        labels[a_start] = f"B-ASP-{pol_code}"
        for i in range(a_start + 1, a_end):
            labels[i] = f"I-ASP-{pol_code}"

        o_start, o_end = _find_span(tokens, opinion)
        if any(labels[i] != "O" for i in range(o_start, o_end)):
            raise ValueError(f"Sobreposicao de opiniao em: {sentence!r}")
        labels[o_start] = "B-OPN"
        for i in range(o_start + 1, o_end):
            labels[i] = "I-OPN"

    return {
        "id": f"pt-gen-{idx}",
        "sentence": sentence,
        "tokens": tokens,
        "labels": labels,
        "polarities": polarities,
    }


def _strip_polarity(labels: list[str]) -> list[str]:
    out = []
    for label in labels:
        if label.startswith("B-ASP-"):
            out.append("B-ASP")
        elif label.startswith("I-ASP-"):
            out.append("I-ASP")
        else:
            out.append(label)
    return out


def main(target_count: int = 2100, seed: int = 42):
    rng = random.Random(seed)

    out_dir = Path(__file__).parent.parent
    polarity_path = out_dir / "bio_dataset_polarity.json"
    plain_path = out_dir / "bio_dataset.json"

    with open(polarity_path, encoding="utf-8") as f:
        polarity_data = json.load(f)
    with open(plain_path, encoding="utf-8") as f:
        plain_data = json.load(f)

    existing_ids = {d["id"] for d in polarity_data}

    seen_sentences = set()
    new_examples = []
    idx = 0
    attempts = 0
    # 2 aspetos: 15%, 3 aspetos: 45%, 4 aspetos: 40% - a maioria das
    # frases geradas tem mais de dois aspetos, como pedido.
    n_choices = [2] * 15 + [3] * 45 + [4] * 40

    while len(new_examples) < target_count and attempts < target_count * 5:
        attempts += 1
        idx += 1
        n = rng.choice(n_choices)
        try:
            ex = generate_example(rng, idx, n)
        except ValueError:
            # Colisão de spans (ex: a mesma palavra de opinião escolhida ao
            # acaso para duas cláusulas da mesma frase) - descarta esta
            # tentativa e continua, em vez de produzir uma etiqueta errada.
            continue
        if ex["sentence"] in seen_sentences or ex["id"] in existing_ids:
            continue
        seen_sentences.add(ex["sentence"])
        new_examples.append(ex)

    if len(new_examples) < target_count:
        raise RuntimeError(f"Só foi possível gerar {len(new_examples)} frases únicas de {target_count} pedidas.")

    polarity_data.extend(new_examples)
    plain_data.extend({**ex, "labels": _strip_polarity(ex["labels"])} for ex in new_examples)

    with open(polarity_path, "w", encoding="utf-8") as f:
        json.dump(polarity_data, f, ensure_ascii=False, indent=2)
    with open(plain_path, "w", encoding="utf-8") as f:
        json.dump(plain_data, f, ensure_ascii=False, indent=2)

    three_plus = sum(1 for ex in new_examples if len(ex["polarities"]) >= 3)
    multi_asp = sum(1 for ex in new_examples for p in ex["polarities"] if len(p["Aspect Term"].split()) > 1)
    multi_opn = sum(1 for ex in new_examples if "I-OPN" in ex["labels"])

    print(f"[OK] {len(new_examples)} reviews sintéticas em português geradas e fundidas.")
    print(f"     Total bio_dataset_polarity.json: {len(polarity_data)} frases")
    print(f"     Frases com 3+ aspetos:           {three_plus}/{len(new_examples)}")
    print(f"     Termos de aspeto multi-palavra:  {multi_asp}")
    print(f"     Frases com opinião multi-palavra (I-OPN): {multi_opn}/{len(new_examples)}")


if __name__ == "__main__":
    main()
