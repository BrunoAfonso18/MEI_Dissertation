import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_PATH = "./absa_model_final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

ID2LABEL = model.config.id2label


def predict(text: str) -> dict:
    words  = text.split()
    inputs = tokenizer(
        words,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(-1)[0].tolist()
    word_ids    = inputs.word_ids()

    aspects, opinions = [], []
    current_asp, current_opn = [], []

    prev_word_id = None
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            prev_word_id = word_id
            continue

        label = ID2LABEL[predictions[token_idx]]
        word  = words[word_id]

        if label == "B-ASP":
            if current_asp:
                aspects.append(" ".join(current_asp))
            current_asp = [word]
            if current_opn:
                opinions.append(" ".join(current_opn))
                current_opn = []

        elif label == "I-ASP":
            current_asp.append(word)

        elif label == "B-OPN":
            if current_opn:
                opinions.append(" ".join(current_opn))
            current_opn = [word]
            if current_asp:
                aspects.append(" ".join(current_asp))
                current_asp = []

        elif label == "I-OPN":
            current_opn.append(word)

        else:  # O
            if current_asp:
                aspects.append(" ".join(current_asp))
                current_asp = []
            if current_opn:
                opinions.append(" ".join(current_opn))
                current_opn = []

        prev_word_id = word_id

    # Flush final
    if current_asp: aspects.append(" ".join(current_asp))
    if current_opn: opinions.append(" ".join(current_opn))

    return {
        "sentence":     text,
        "aspect_terms": aspects,
        "opinion_terms": opinions,
    }


def print_result(result: dict):
    print(f'\n "{result["sentence"]}"')
    print(f'   Aspetos  : {result["aspect_terms"]}')
    print(f'   Opiniões : {result["opinion_terms"]}')


if __name__ == "__main__":
    reviews = [
        "The battery life is amazing but the screen is terrible.",
        "A duração da bateria é incrível mas o ecrã é péssimo.",
        "Great food but the service was incredibly slow.",
    ]
    for review in reviews:
        print_result(predict(review))