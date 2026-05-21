import json
import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
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

    return {
        "f1":        f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall":    recall_score(true_labels, pred_labels),
    }


def compute_class_weights(bio_path: str) -> list[float]:
    """Inverse-frequency weights to address 81% 'O' class imbalance."""
    with open(bio_path, encoding="utf-8") as f:
        data = json.load(f)
    counts = Counter(label for d in data for label in d["labels"])
    total  = sum(counts.values())
    n_classes = len(LABELS)
    # weight = total / (n_classes * count); cap at 20 to avoid extreme values
    weights = []
    for label in LABELS:
        count = counts.get(label, 1)
        weights.append(min(total / (n_classes * count), 20.0))
    return weights


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        if self.class_weights is not None:
            weights  = torch.tensor(self.class_weights, dtype=torch.float, device=logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train(bio_path: str = "absa_module/bio_dataset_polarity.json", output_dir: str = "absa_module/absa_model_final_polarity"):
    train_ds, val_ds, _ = build_splits(bio_path)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class_weights = compute_class_weights(bio_path)
    print("Class weights:", {l: round(w, 2) for l, w in zip(LABELS, class_weights)})

    batch_size   = 16
    num_epochs   = 10
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    total_steps  = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    args = TrainingArguments(
        output_dir="./checkpoints_polarity",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=warmup_steps,
        logging_steps=50,
        fp16=False,
        label_smoothing_factor=0.1,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    save_path = output_dir if os.path.isabs(output_dir) else os.path.abspath(output_dir)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Modelo guardado em {save_path}")

    return trainer


if __name__ == "__main__":
    train()