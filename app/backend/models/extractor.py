import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

POLARITY_MAP = {
    "pos": "positive",
    "neg": "negative", 
    "neu": "neutral",
    "con": "conflict",
}

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

        predictions = outputs.logits.argmax(-1)[0].tolist()
        word_ids    = inputs.word_ids()

        aspects = []
        opinions = []
        current_asp = []
        current_opn = []
        current_polarity = None
        prev_word_id = None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word_id:
                prev_word_id = word_id
                continue

            label = self.id2label[predictions[token_idx]]
            word  = words[word_id]

            if label.startswith("B-ASP-"):
                if current_asp:
                    aspects.append({
                        "term": " ".join(current_asp),
                        "polarity": current_polarity
                    })
                if current_opn:
                    opinions.append(" ".join(current_opn))
                current_asp, current_opn = [word], []
                pol_code = label.split("-")[-1]
                current_polarity = POLARITY_MAP.get(pol_code, "neutral")
                
            elif label.startswith("I-ASP-"):
                current_asp.append(word)
                pol_code = label.split("-")[-1]
                current_polarity = POLARITY_MAP.get(pol_code, "neutral")
                
            elif label == "B-OPN":
                if current_opn:
                    opinions.append(" ".join(current_opn))
                if current_asp:
                    aspects.append({
                        "term": " ".join(current_asp),
                        "polarity": current_polarity
                    })
                current_opn, current_asp = [word], []
                current_polarity = None
                
            elif label == "I-OPN":
                current_opn.append(word)
                
            else:
                if current_asp:
                    aspects.append({
                        "term": " ".join(current_asp),
                        "polarity": current_polarity
                    })
                    current_asp = []
                if current_opn:
                    opinions.append(" ".join(current_opn))
                    current_opn = []
                current_polarity = None

            prev_word_id = word_id

        if current_asp:
            aspects.append({
                "term": " ".join(current_asp),
                "polarity": current_polarity
            })
        if current_opn:
            opinions.append(" ".join(current_opn))

        return {
            "aspects": [a["term"] for a in aspects],
            "polarities": [a["polarity"] for a in aspects],
            "opinions": opinions
        }