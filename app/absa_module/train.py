import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from absa_module.dataset import build_splits, tokenizer, LABELS, LABEL2ID, ID2LABEL, MODEL_NAME


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions    = np.argmax(logits, axis=-1)

    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]
    pred_labels = [
        [ID2LABEL[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    
    # Overall metrics
    metrics = {
        "f1":        f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall":    recall_score(true_labels, pred_labels),
    }
    
    # Per-class metrics for polarities
    for polarity in ["pos", "neg", "neu", "con"]:
        pos_labels = [l for l in LABELS if polarity in l]
        true_filtered = [[t for t in seq if t in pos_labels or t == "O"] for seq in true_labels]
        pred_filtered = [[p for p in seq if p in pos_labels or p == "O"] for seq in pred_labels]
        if any("B-ASP-" + polarity in l for l in true_labels):
            try:
                metrics[f"f1_{polarity}"] = f1_score(true_filtered, pred_filtered, average="binary", pos_label="B-ASP-" + polarity)
            except:
                pass
    
    return metrics


def train(bio_path: str = "absa_module/bio_dataset_polarity.json", output_dir: str = "./absa_model_final_polarity"):
    train_ds, val_ds, _ = build_splits(bio_path)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir="./checkpoints_polarity",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=100,               
        logging_steps=50,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    import os
    save_path = os.path.join(os.path.dirname(bio_path), output_dir)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Modelo guardado em {save_path}")

    return trainer


if __name__ == "__main__":
    train()