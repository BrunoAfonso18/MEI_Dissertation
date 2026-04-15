import spacy
import pandas as pd

nlp_en = spacy.load("en_core_web_sm")


def find_opinion_term(sentence: str, aspect_term: str) -> str | None:
    doc = nlp_en(sentence)
    aspect_words = set(aspect_term.lower().split())

    aspect_tokens = [
        t for t in doc
        if t.text.lower() in aspect_words
    ]

    if not aspect_tokens:
        return None

    candidates = []

    for asp_token in aspect_tokens:
        
        for child in asp_token.children:
            if child.pos_ in ("ADJ", "ADV") and child.dep_ in ("amod", "advmod"):
                candidates.append((child.text, child.i))

        if asp_token.dep_ in ("nsubj", "nsubjpass"):
            verb = asp_token.head
            for child in verb.children:
                if child.pos_ == "ADJ" and child.dep_ in ("acomp", "attr"):
                    candidates.append((child.text, child.i))

        if asp_token.dep_ in ("dobj", "pobj"):
            verb = asp_token.head
            for child in verb.children:
                if child.pos_ == "ADV" and child.dep_ == "advmod":
                    candidates.append((child.text, child.i))

        head = asp_token.head
        for sibling in head.children:
            if sibling.pos_ == "ADJ" and sibling != asp_token:
                candidates.append((sibling.text, sibling.i))

    if not candidates:
        return None

    asp_idx = aspect_tokens[0].i
    candidates.sort(key=lambda x: abs(x[1] - asp_idx))
    return candidates[0][0]


def add_opinion_terms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    opinion_en = []
    found_count = 0

    for _, row in df.iterrows():
        op_en = find_opinion_term(str(row["Sentence"]), str(row["Aspect Term"]))
        opinion_en.append(op_en)
        if op_en:
            found_count += 1

    df["Opinion Term"] = opinion_en

    return df

df = pd.read_csv("absa_reviews_2014/reviews.csv")
df_with_opinions = add_opinion_terms(df)

cols = ["Sentence", "Aspect Term", "Opinion Term", "polarity"]
print(df_with_opinions[cols].head(15).to_string())

failed = df_with_opinions[df_with_opinions["Opinion Term"].isna()]
print(f"\n Frases sem opinion term ({len(failed)}):")
print(failed[["Sentence", "Aspect Term"]].head(10).to_string())

# Guardar
df_with_opinions.to_csv("absa_reviews_2014/reviews_with_opinions.csv", index=False)