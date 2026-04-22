import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

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

        aspects, opinions    = [], []
        current_asp, current_opn = [], []
        prev_word_id = None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word_id:
                prev_word_id = word_id
                continue

            label = self.id2label[predictions[token_idx]]
            word  = words[word_id]

            if label == "B-ASP":
                if current_asp: aspects.append(" ".join(current_asp))
                if current_opn: opinions.append(" ".join(current_opn))
                current_asp, current_opn = [word], []
            elif label == "I-ASP":
                current_asp.append(word)
            elif label == "B-OPN":
                if current_opn: opinions.append(" ".join(current_opn))
                if current_asp: aspects.append(" ".join(current_asp))
                current_opn, current_asp = [word], []
            elif label == "I-OPN":
                current_opn.append(word)
            else:
                if current_asp: aspects.append(" ".join(current_asp)); current_asp = []
                if current_opn: opinions.append(" ".join(current_opn)); current_opn = []

            prev_word_id = word_id

        if current_asp: aspects.append(" ".join(current_asp))
        if current_opn: opinions.append(" ".join(current_opn))

        return {"aspects": aspects, "opinions": opinions}