"""
BIO dataset creation from SemEval 2014 ABSA datasets (English) and from the
Portuguese restaurant reviews annotated in the same format
(Restaurants_PT_Train.csv).

Key improvements over previous version:
  - Uses character offsets (from/to columns) instead of fragile string matching
  - Only uses restaurant domain data (excludes laptops)
  - Generates polarity-aware labels (B-ASP-pos/neg/neu/con) directly
  - Combines Train + Trial splits for maximum data
  - Handles multiple aspects per sentence correctly
  - Outputs both bio_dataset.json (plain) and bio_dataset_polarity.json

The Portuguese CSV follows the SemEval columns (id, Sentence, Aspect Term,
polarity, from, to) plus the annotated opinion term of each aspect
(Opinion Term, opinion_from, opinion_to), with one row per aspect-opinion
pair. Its opinion terms are tagged from those character offsets; spaCy is
only used to infer opinion terms for the English SemEval sentences, which
have no opinion annotation.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False
    print("[AVISO] spaCy não disponível — opinion terms serão omitidos.")

POLARITY_MAP = {
    "positive": "pos",
    "negative": "neg",
    "neutral":  "neu",
    "conflict": "con",
}

DATA_DIR = Path(__file__).parent.parent.parent / "zips" / "absa_reviews_2014"

RESTAURANT_FILES = [
    DATA_DIR / "Restaurants_Train_v2.csv",
    DATA_DIR / "restaurants-trial.csv",
]

PT_FILE = DATA_DIR / "Restaurants_PT_Train.csv"


# ── Tokenisation preserving character positions ────────────────────

def word_spans(sentence: str) -> list[tuple[str, int, int]]:
    """Return [(word, char_start, char_end), ...] for every whitespace token."""
    tokens = []
    i = 0
    while i < len(sentence):
        if sentence[i].isspace():
            i += 1
            continue
        j = i
        while j < len(sentence) and not sentence[j].isspace():
            j += 1
        tokens.append((sentence[i:j], i, j))
        i = j
    return tokens


# ── Opinion term extraction (spaCy dependency parse) ──────────────

def _find_opinion(sentence: str, aspect_term: str) -> str | None:
    if not _SPACY_OK:
        return None
    doc = nlp_doc(sentence)
    aspect_words = set(aspect_term.lower().split())
    asp_tokens   = [t for t in doc if t.text.lower() in aspect_words]
    if not asp_tokens:
        return None

    candidates = []
    for at in asp_tokens:
        for child in at.children:
            if child.pos_ in ("ADJ", "ADV") and child.dep_ in ("amod", "advmod"):
                candidates.append((child.text, child.i))
        if at.dep_ in ("nsubj", "nsubjpass"):
            verb = at.head
            for child in verb.children:
                if child.pos_ == "ADJ" and child.dep_ in ("acomp", "attr"):
                    candidates.append((child.text, child.i))
        if at.dep_ in ("dobj", "pobj"):
            verb = at.head
            for child in verb.children:
                if child.pos_ == "ADV" and child.dep_ == "advmod":
                    candidates.append((child.text, child.i))
        for sib in at.head.children:
            if sib.pos_ == "ADJ" and sib != at:
                candidates.append((sib.text, sib.i))

    if not candidates:
        return None
    asp_idx = asp_tokens[0].i
    candidates.sort(key=lambda x: abs(x[1] - asp_idx))
    return candidates[0][0]


_doc_cache: dict[str, object] = {}

def nlp_doc(sentence: str):
    if sentence not in _doc_cache:
        _doc_cache[sentence] = _nlp(sentence)
    return _doc_cache[sentence]


# ── Core BIO builder ──────────────────────────────────────────────

def _tag_opinion(words: list[str], labels: list[str], opinion_word: str) -> bool:
    """Try to place B-OPN / I-OPN for opinion_word in the token list."""
    opn_parts = opinion_word.split()
    norm = lambda w: re.sub(r"[.,!?\"']+", "", w.lower())
    for i in range(len(words) - len(opn_parts) + 1):
        window = [norm(words[i + j]) for j in range(len(opn_parts))]
        target = [norm(p) for p in opn_parts]
        if window == target and all(labels[i + j] == "O" for j in range(len(opn_parts))):
            labels[i] = "B-OPN"
            for j in range(1, len(opn_parts)):
                labels[i + j] = "I-OPN"
            return True
    return False


def _tag_opinion_span(tokens: list[tuple[str, int, int]], labels: list[str], opn_from: int, opn_to: int) -> bool:
    """Place B-OPN / I-OPN on the words overlapping the character span [opn_from, opn_to)."""
    idxs = [i for i, (_, ws, we) in enumerate(tokens) if ws < opn_to and we > opn_from]
    if not idxs or not all(labels[i] == "O" for i in idxs):
        return False
    labels[idxs[0]] = "B-OPN"
    for i in idxs[1:]:
        labels[i] = "I-OPN"
    return True


def build_bio_dataset(df: pd.DataFrame, opinion_spans: dict | None = None) -> tuple[list[dict], dict]:
    """
    Build BIO-tagged records from a DataFrame that has:
        id, Sentence, Aspect Term, polarity, from, to
    `opinion_spans` maps a sentence id to its annotated opinion character
    spans [(from, to), ...] (Portuguese CSV); when it is None the opinion
    term is inferred with spaCy (English SemEval).
    Returns (records, stats).
    """
    stats = Counter()
    results = []

    for (sid, sentence), group in df.groupby(["id", "Sentence"], sort=False):
        tokens = word_spans(sentence)
        if not tokens:
            continue
        words  = [t[0] for t in tokens]
        labels = ["O"] * len(words)
        polarities = []
        stats["sentences"] += 1

        for _, row in group.iterrows():
            asp_from = int(row["from"])
            asp_to   = int(row["to"])
            pol_code = POLARITY_MAP.get(str(row["polarity"]).lower(), "neu")
            asp_text = str(row["Aspect Term"])

            # Words that overlap with [asp_from, asp_to)
            asp_idxs = [
                i for i, (_, ws, we) in enumerate(tokens)
                if ws < asp_to and we > asp_from
            ]

            if not asp_idxs:
                stats["asp_missed"] += 1
                continue

            stats["asp_found"] += 1

            # Skip if already tagged (overlapping aspects)
            if not all(labels[i] == "O" for i in asp_idxs):
                stats["asp_overlap_skipped"] += 1
                continue

            labels[asp_idxs[0]] = f"B-ASP-{pol_code}"
            for i in asp_idxs[1:]:
                labels[i] = f"I-ASP-{pol_code}"

            polarities.append({"Aspect Term": asp_text, "polarity": str(row["polarity"])})

        if opinion_spans is not None:
            # Annotated opinion terms: every one of them is tagged.
            for opn_from, opn_to in opinion_spans.get(sid, []):
                if _tag_opinion_span(tokens, labels, opn_from, opn_to):
                    stats["opn_found"] += 1
                else:
                    stats["opn_missed"] += 1
        # Opinion term: one per sentence (first aspect that yields a candidate)
        elif _SPACY_OK:
            for _, row in group.iterrows():
                opn = _find_opinion(sentence, str(row["Aspect Term"]))
                if opn and _tag_opinion(words, labels, opn):
                    stats["opn_found"] += 1
                    break

        results.append({
            "id":         sid,
            "sentence":   sentence,
            "tokens":     words,
            "labels":     labels,
            "polarities": polarities,
        })

    return results, stats


# ── Plain labels (no polarity in BIO tags) ────────────────────────

def _strip_polarity(labels: list[str]) -> list[str]:
    """Convert B-ASP-pos → B-ASP, I-ASP-neg → I-ASP, keep OPN/O unchanged."""
    out = []
    for label in labels:
        if label.startswith("B-ASP-"):
            out.append("B-ASP")
        elif label.startswith("I-ASP-"):
            out.append("I-ASP")
        else:
            out.append(label)
    return out


# ── Entry point ───────────────────────────────────────────────────

def main():
    # 1. Load and combine restaurant CSVs
    dfs = []
    for p in RESTAURANT_FILES:
        if not p.exists():
            print(f"[AVISO] Ficheiro não encontrado: {p}")
            continue
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"  Carregado {p.name}: {len(df)} linhas")

    if not dfs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {DATA_DIR}")

    df_all = pd.concat(dfs, ignore_index=True)

    # 2. Basic cleaning
    required = {"id", "Sentence", "Aspect Term", "polarity", "from", "to"}
    missing  = required - set(df_all.columns)
    if missing:
        raise ValueError(f"Colunas em falta: {missing}")

    df_all = df_all.dropna(subset=["Sentence", "Aspect Term", "polarity"])
    df_all = df_all[df_all["Aspect Term"].astype(str).str.strip() != ""]
    df_all["from"] = pd.to_numeric(df_all["from"], errors="coerce")
    df_all["to"]   = pd.to_numeric(df_all["to"],   errors="coerce")
    df_all = df_all.dropna(subset=["from", "to"])
    df_all = df_all.drop_duplicates(subset=["id", "Sentence", "Aspect Term", "from", "to"])

    print(f"\nTotal de instâncias após limpeza: {len(df_all)}")
    print(f"Distribuição de polaridade:\n{df_all['polarity'].value_counts().to_string()}")

    # 3. Build BIO records
    bio_data, stats = build_bio_dataset(df_all)

    # 3b. Portuguese reviews (same format + annotated opinion terms). One row
    # per aspect-opinion pair: aspects are deduplicated before tagging and
    # every annotated opinion span is tagged.
    if not PT_FILE.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {PT_FILE}")
    df_pt = pd.read_csv(PT_FILE, dtype={"id": str})
    print(f"\n  Carregado {PT_FILE.name}: {len(df_pt)} linhas")
    opn = df_pt.dropna(subset=["opinion_from", "opinion_to"])
    opinion_spans = {}
    for sid, o_from, o_to in zip(opn["id"], opn["opinion_from"], opn["opinion_to"]):
        span = (int(o_from), int(o_to))
        if span not in opinion_spans.setdefault(sid, []):
            opinion_spans[sid].append(span)
    df_pt = df_pt.drop_duplicates(subset=["id", "Sentence", "Aspect Term", "from", "to"])
    print(f"Distribuição de polaridade (PT):\n{df_pt['polarity'].value_counts().to_string()}")
    bio_pt, stats_pt = build_bio_dataset(df_pt, opinion_spans=opinion_spans)
    bio_data += bio_pt
    for k, v in stats_pt.items():
        stats[f"pt_{k}"] += v

    has_asp = sum(1 for d in bio_data if any(l.startswith("B-ASP") for l in d["labels"]))
    has_opn = sum(1 for d in bio_data if "B-OPN" in d["labels"])

    print(f"\n=== Estatísticas de tagging ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")
    print(f"  {'sentences_with_aspect':25s}: {has_asp}")
    print(f"  {'sentences_with_opinion':25s}: {has_opn}")

    # 4. Label distribution
    all_labels = [l for d in bio_data for l in d["labels"]]
    cnt = Counter(all_labels)
    total_tok = sum(cnt.values())
    print(f"\n=== Distribuição de labels ({total_tok} tokens) ===")
    for label, count in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {label:20s}: {count:5d} ({count/total_tok*100:.1f}%)")

    out_dir = Path(__file__).parent

    # 5. Save bio_dataset_polarity.json (polarity-aware labels)
    polarity_path = out_dir / "bio_dataset_polarity.json"
    with open(polarity_path, "w", encoding="utf-8") as f:
        json.dump(bio_data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] bio_dataset_polarity.json -> {len(bio_data)} frases")

    # 6. Save bio_dataset.json (plain labels, for compatibility)
    plain_data = [
        {**d, "labels": _strip_polarity(d["labels"])}
        for d in bio_data
    ]
    plain_path = out_dir / "bio_dataset.json"
    with open(plain_path, "w", encoding="utf-8") as f:
        json.dump(plain_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] bio_dataset.json         -> {len(plain_data)} frases")


if __name__ == "__main__":
    main()
