import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

POLARITY_MAP = {
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
    "con": "conflict",
}

_PUNCT_RE = re.compile(r"[^\w\s\-]")


def _clean_term(tokens: list[str]) -> str:
    return _PUNCT_RE.sub("", " ".join(tokens)).strip()


class AspectExtractor:
    def __init__(self, model_path: str = "./absa_model_final"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label  = self.model.config.id2label

    def predict(self, text: str) -> dict:
        words  = text.split()
        inputs = self.tokenizer(
            words,
            return_tensors="pt",
            is_split_into_words=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits      = outputs.logits[0]                      # [seq_len, num_labels]
        probs       = F.softmax(logits, dim=-1)              # [seq_len, num_labels]
        predictions = logits.argmax(-1).tolist()
        word_ids    = inputs.word_ids()

        aspects  = []
        opinions = []
        current_asp      = []
        current_opn      = []
        current_polarity = None
        current_conf     = 0.0
        prev_word_id     = None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word_id:
                prev_word_id = word_id
                continue

            pred_id = predictions[token_idx]
            label   = self.id2label[pred_id]
            word    = words[word_id]
            score   = float(probs[token_idx, pred_id])

            if label.startswith("B-ASP-"):
                if current_asp:
                    aspects.append({
                        "term":       _clean_term(current_asp),
                        "polarity":   current_polarity,
                        "confidence": current_conf,
                    })
                if current_opn:
                    opinions.append(_clean_term(current_opn))
                current_asp      = [word]
                current_opn      = []
                pol_code         = label.split("-")[-1]
                current_polarity = POLARITY_MAP.get(pol_code, "neutral")
                current_conf     = score

            elif label.startswith("I-ASP-"):
                current_asp.append(word)

            elif label == "B-OPN":
                if current_opn:
                    opinions.append(_clean_term(current_opn))
                if current_asp:
                    aspects.append({
                        "term":       _clean_term(current_asp),
                        "polarity":   current_polarity,
                        "confidence": current_conf,
                    })
                current_opn      = [word]
                current_asp      = []
                current_polarity = None
                current_conf     = 0.0

            elif label == "I-OPN":
                current_opn.append(word)

            else:
                if current_asp:
                    aspects.append({
                        "term":       _clean_term(current_asp),
                        "polarity":   current_polarity,
                        "confidence": current_conf,
                    })
                    current_asp  = []
                    current_conf = 0.0
                if current_opn:
                    opinions.append(_clean_term(current_opn))
                    current_opn = []
                current_polarity = None

            prev_word_id = word_id

        if current_asp:
            aspects.append({
                "term":       _clean_term(current_asp),
                "polarity":   current_polarity,
                "confidence": current_conf,
            })
        if current_opn:
            opinions.append(_clean_term(current_opn))

        return {
            "aspects":     [a["term"]       for a in aspects],
            "polarities":  [a["polarity"]   for a in aspects],
            "confidences": [a["confidence"] for a in aspects],
            "opinions":    opinions,
        }
