import json
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import classification_report
from dataset import build_splits, ID2LABEL, LABEL2ID
import torch

def evaluate(model_path: str = "./absa_model_final",
             bio_path:   str = "bio_dataset.json"):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    _, _, test_ds = build_splits(bio_path)

    all_true, all_pred = [], []

    for example in test_ds:
        input_ids      = torch.tensor([example["input_ids"]])
        attention_mask = torch.tensor([example["attention_mask"]])
        labels         = example["labels"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = outputs.logits.argmax(-1)[0].tolist()

        true_seq = [ID2LABEL[l] for l in labels      if l != -100]
        pred_seq = [ID2LABEL[p] for p, l in zip(predictions, labels) if l != -100]

        all_true.append(true_seq)
        all_pred.append(pred_seq)

    print(classification_report(all_true, all_pred))

    print("\nERROS:")
    errors = [
        (t, p) for t, p in zip(all_true, all_pred) if t != p
    ]
    for true, pred in errors[:5]:
        print(f"  True: {true}")
        print(f"  Pred: {pred}")
        print()


if __name__ == "__main__":
    evaluate()