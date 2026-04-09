import spacy
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import rdflib
from rdflib.namespace import RDF

logging.basicConfig(level=logging.INFO, format="[ABSA] %(message)s")
logger = logging.getLogger(__name__)

nlp = spacy.load("pt_core_news_sm")

_ONTO_PATH = Path(__file__).parent / "ontosenticnet_pt" / "ontosenticnet.owl"
_ONTO: Dict[str, Dict] = {}

def _carregar_ontosenticnet_owl():
    global _ONTO
    if not _ONTO_PATH.exists():
        logger.warning("ontosenticnet.owl não encontrado. Usando apenas léxico manual.")
        return {}

    g = rdflib.Graph()
    g.parse(str(_ONTO_PATH), format="xml")

    SENTIC = rdflib.Namespace("http://sentic.net/")
    onto = {}

    for s, _, _ in g.triples((None, RDF.type, None)):
        concept = str(s).split("/")[-1].lower().replace(" ", "_").replace("-", "_")
        if not concept:
            continue
        onto[concept] = {
            "pleasantness": 0.0,
            "attention": 0.0,
            "sensitivity": 0.0,
            "aptitude": 0.0,
            "emotion": "neutral",
            "polarity": "neutral"
        }

    logger.info(f"OntoSenticNet carregado: {len(onto)} conceitos")
    _ONTO = onto
    return onto

_ONTO = _carregar_ontosenticnet_owl()

OPINION_LEXICON: Dict[str, float] = {
    "excelente": 1.0, "ótimo": 1.0, "fantástico": 1.0, "incrível": 0.9,
    "bom": 0.7, "boa": 0.7, "rápido": 0.7, "bonito": 0.7,
    "mau": -0.7, "péssimo": -1.0, "horrível": -1.0, "fraco": -0.6,
    "caro": -0.6, "lento": -0.7, "pouco": -0.5,
}

NEGATION_WORDS = {"não", "nunca", "jamais", "nem", "nenhum", "sem"}

ASPECT_CATEGORIES: Dict[str, str] = {
    "bateria": "Bateria", "autonomia": "Bateria", "duracao": "Bateria",
    "ecrã": "Ecrã", "ecra": "Ecrã", "tela": "Ecrã",
    "preço": "Preço", "preco": "Preço",
    "design": "Design", "qualidade": "Qualidade",
    "câmara": "Câmara", "camera": "Câmara",
    "som": "Som", "desempenho": "Desempenho",
}

def _lookup_onto(termo: str) -> Optional[Dict]:
    if not _ONTO:
        return None
    t = termo.lower().strip().replace(" ", "_").replace("-", "_")
    return _ONTO.get(t)

def _score_lexico(opinion_term: Optional[str]) -> Tuple[float, float, Optional[str]]:
    if not opinion_term:
        return 0.0, 0.0, None
    onto = _lookup_onto(opinion_term)
    if onto:
        return onto["pleasantness"], onto["attention"], onto["emotion"]
    score = OPINION_LEXICON.get(opinion_term.lower(), 0.0)
    return score, abs(score) * 0.8, None

def _tem_negacao(token) -> bool:
    for child in token.children:
        if child.dep_ == "neg" or child.text.lower() in NEGATION_WORDS:
            return True
    return token.head.text.lower() in NEGATION_WORDS

def _extrair_aspetos_e_opiniao(texto: str) -> List[Dict]:
    doc = nlp(texto.lower())
    aspetos = []
    vistos = set()

    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ not in ("NOUN", "PROPN") or root.lemma_ in vistos:
            continue
        visto = root.lemma_
        termo = chunk.text.strip()
        categoria = ASPECT_CATEGORIES.get(termo.lower(), "Geral")

        opinion_term = None
        negado = False
        for child in root.children:
            if child.dep_ in ("amod", "advmod"):
                opinion_term = child.text.lower()
                negado = _tem_negacao(child)
                break

        pleasantness, attention, emotion = _score_lexico(opinion_term)
        if negado:
            pleasantness = -pleasantness

        vistos.add(visto)
        aspetos.append({
            "aspect_term": termo,
            "aspect_category": categoria,
            "opinion_term": opinion_term or "não_identificado",
            "negado": negado,
            "pleasantness": pleasantness,
            "attention": attention,
            "emotion": emotion,
            "start": chunk.start_char,
            "end": chunk.end_char,
        })
    return aspetos

def processar_review(review_texto: str, review_id: str = None) -> Dict:
    if review_id is None:
        review_id = f"rev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    aspetos_raw = _extrair_aspetos_e_opiniao(review_texto)

    aspetos_finais = []
    for asp in aspetos_raw:
        aspetos_finais.append({
            "aspect_term": asp["aspect_term"],
            "aspect_category": asp["aspect_category"],
            "opinion_term": asp["opinion_term"],
            "sentiment_polarity": {
                "score": round((asp["pleasantness"] + 1) / 2, 4),
                "intensity": round(abs(asp["attention"]), 4),
                "negado": asp["negado"]
            },
            "afeto": {
                "pleasantness": asp["pleasantness"],
                "attention": asp["attention"],
                "emotion": asp["emotion"]
            }
        })

    return {
        "review_id": review_id,
        "raw_text": review_texto,
        "processed_at": datetime.now().isoformat(),
        "num_aspects": len(aspetos_finais),
        "aspects": aspetos_finais,
        "overall_sentiment": {"score": 0.5, "intensity": 0.0}
    }


if __name__ == "__main__":
    texto = "A bateria dura pouco mas o ecrã é excelente."
    print(json.dumps(processar_review(texto), indent=2, ensure_ascii=False))