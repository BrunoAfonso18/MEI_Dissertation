import json
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

LABELS    = ["O", "B-ASP", "I-ASP", "B-OPN", "I-OPN"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}
MODEL_NAME = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_bio_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    filtered = [d for d in data if "B-ASP" in d["labels"]]
    print(f"Total: {len(data)} | Com aspetos: {len(filtered)} | Descartadas: {len(data)-len(filtered)}")
    return filtered


def tokenize_and_align(examples: dict) -> dict:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned  = []
        prev     = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)           
            elif word_id != prev:
                aligned.append(LABEL2ID[labels[word_id]])
            else:
                label = labels[word_id]
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                aligned.append(LABEL2ID.get(label, LABEL2ID["O"]))
            prev = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized


def build_splits(path: str):
    data = load_bio_data(path)

    records = [{"tokens": d["tokens"], "labels": d["labels"]} for d in data]

    train_data, temp   = train_test_split(records, test_size=0.2,  random_state=42)
    val_data,   test_data = train_test_split(temp,  test_size=0.5, random_state=42)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_ds = Dataset.from_list(train_data).map(tokenize_and_align, batched=True)
    val_ds   = Dataset.from_list(val_data).map(tokenize_and_align,   batched=True)
    test_ds  = Dataset.from_list(test_data).map(tokenize_and_align,  batched=True)

    return train_ds, val_ds, test_ds