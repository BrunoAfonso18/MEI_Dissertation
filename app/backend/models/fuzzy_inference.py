"""
Sistema de inferência difusa de Mamdani que mede a intensidade da opinião
sobre cada aspeto.

Entradas:
- V, valência lexical da opinião [-1, 1] (models/opinion_lexicon.py);
- M, modificador de grau [0, 2], 1 = nenhum (models/opinion_lexicon.py);
- E, evidência do modelo [-1, 1] = sinal(polaridade) x confiança, após
  adjust_polarity (neutro e conflito -> 0).
Saída:
- S, sentimento [0, 1], 7 termos (muito negativo ... muito positivo).

19 regras (12 de concordância, 3 de neutralidade, 4 de discordância
graduada), implicação pelo mínimo, agregação aditiva e defuzzificação pelo
centróide. A certeza (confiança do modelo após as correções de negação e
ironia) é devolvida em separado e não entra no valor de S.
"""

import re
import threading

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from models.fuzzy_sentiment import fuzzy_analyzer
from models.linguistic_adjustments import (
    _CLAUSE_BREAK_RE,
    _CLAUSE_SPLIT_WORDS,
    _normalize,
    adjust_polarity,
    detect_language,
)
from models.opinion_lexicon import opinion_features

POSITIVE_THRESHOLD = 7 / 12   # S >= 0.583 -> positivo
NEGATIVE_THRESHOLD = 5 / 12   # S <= 0.417 -> negativo

# Aspeto implícito das avaliações sem nenhum aspeto explícito (score_implicit).
IMPLICIT_ASPECT = "geral"
IMPLICIT_CATEGORY = "GENERAL"
IMPLICIT_CERTAINTY = 0.5      # sem evidência do modelo: leitura só lexical
# Orações concessivas ("embora caro, …") são a ressalva e não a opinião principal.
_CONCESSIVE = {
    "pt": {"embora", "apesar", "ainda que", "mesmo que"},
    "en": {"although", "though", "despite", "even though"},
}
_GLUED_PUNCT_RE = re.compile(r"([.,;:!?])(?=[^\s.,;:!?])")


def _build_system() -> ctrl.ControlSystemSimulation:
    V = ctrl.Antecedent(np.linspace(-1, 1, 201), "V")
    M = ctrl.Antecedent(np.linspace(0, 2, 201), "M")
    E = ctrl.Antecedent(np.linspace(-1, 1, 201), "E")
    S = ctrl.Consequent(np.linspace(0, 1, 201), "S", defuzzify_method="centroid")
    S.accumulation_method = np.add

    V["MN"] = fuzz.trapmf(V.universe, [-1, -1, -0.8, -0.5])
    V["N"] = fuzz.trimf(V.universe, [-0.8, -0.5, -0.15])
    V["Z"] = fuzz.trimf(V.universe, [-0.3, 0, 0.3])
    V["P"] = fuzz.trimf(V.universe, [0.15, 0.5, 0.8])
    V["MP"] = fuzz.trapmf(V.universe, [0.5, 0.8, 1, 1])

    M["ATEN"] = fuzz.trapmf(M.universe, [0, 0, 0.5, 0.9])
    M["NORM"] = fuzz.trimf(M.universe, [0.6, 1, 1.4])
    M["INT"] = fuzz.trapmf(M.universe, [1.1, 1.5, 2, 2])

    E["NEG"] = fuzz.trapmf(E.universe, [-1, -1, -0.5, 0])
    E["INC"] = fuzz.trimf(E.universe, [-0.5, 0, 0.5])
    E["POS"] = fuzz.trapmf(E.universe, [0, 0.5, 1, 1])

    peaks = {"MN": 0, "N": 1 / 6, "LN": 2 / 6, "Z": 0.5, "LP": 4 / 6, "P": 5 / 6, "MP": 1}
    for term, peak in peaks.items():
        S[term] = fuzz.trimf(S.universe, [max(0, peak - 1 / 6), peak, min(1, peak + 1 / 6)])

    not_neg, not_pos = ~E["NEG"], ~E["POS"]
    rules = [
        # Concordância: a valência dá o nível, o modificador sobe/desce um nível.
        ctrl.Rule(V["MP"] & M["ATEN"] & not_neg, S["P"]),
        ctrl.Rule(V["MP"] & M["NORM"] & not_neg, S["MP"]),
        ctrl.Rule(V["MP"] & M["INT"] & not_neg, S["MP"]),
        ctrl.Rule(V["P"] & M["ATEN"] & not_neg, S["LP"]),
        ctrl.Rule(V["P"] & M["NORM"] & not_neg, S["P"]),
        ctrl.Rule(V["P"] & M["INT"] & not_neg, S["MP"]),
        ctrl.Rule(V["N"] & M["ATEN"] & not_pos, S["LN"]),
        ctrl.Rule(V["N"] & M["NORM"] & not_pos, S["N"]),
        ctrl.Rule(V["N"] & M["INT"] & not_pos, S["MN"]),
        ctrl.Rule(V["MN"] & M["ATEN"] & not_pos, S["N"]),
        ctrl.Rule(V["MN"] & M["NORM"] & not_pos, S["MN"]),
        ctrl.Rule(V["MN"] & M["INT"] & not_pos, S["MN"]),
        # Neutralidade: opinião neutra, o modelo dá a direção com intensidade fraca.
        ctrl.Rule(V["Z"] & E["INC"], S["Z"]),
        ctrl.Rule(V["Z"] & E["POS"], S["LP"]),
        ctrl.Rule(V["Z"] & E["NEG"], S["LN"]),
        # Discordância graduada: o modelo decide a direção, com baixa intensidade.
        ctrl.Rule(V["N"] & E["POS"], S["LP"]),
        ctrl.Rule(V["MN"] & E["POS"], S["Z"]),
        ctrl.Rule(V["P"] & E["NEG"], S["LN"]),
        ctrl.Rule(V["MP"] & E["NEG"], S["Z"]),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


_simulation = _build_system()
# A simulação guarda estado interno; o upload em lote processa várias
# avaliações em paralelo, por isso a inferência é serializada.
_lock = threading.Lock()


def infer(valence: float, modifier: float, evidence: float) -> float:
    """Corre o sistema de Mamdani e devolve S em [0, 1]."""
    with _lock:
        _simulation.input["V"] = float(np.clip(valence, -1, 1))
        _simulation.input["M"] = float(np.clip(modifier, 0, 2))
        _simulation.input["E"] = float(np.clip(evidence, -1, 1))
        try:
            _simulation.compute()
            return float(_simulation.output["S"])
        except (ValueError, KeyError):
            return 0.5  # nenhuma regra ativada


def label_for(score: float) -> str:
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def _memberships(score: float) -> dict:
    """Pertinência de S aos três triângulos de FuzzySentimentAnalyzer (campos pos/neut/neg_score)."""
    x = fuzzy_analyzer.x_confidence
    return {
        "positive_score": float(fuzz.interp_membership(x, fuzzy_analyzer.positive, score)),
        "neutral_score": float(fuzz.interp_membership(x, fuzzy_analyzer.neutral, score)),
        "negative_score": float(fuzz.interp_membership(x, fuzzy_analyzer.negative, score)),
    }


def score_aspect(
    text: str,
    aspect: str,
    opinion: str | None,
    polarity: str | None,
    confidence: float,
) -> dict:
    """
    Intensidade da opinião sobre um aspeto.

    Devolve {"crisp", "label", "certainty", "positive_score", "neutral_score",
    "negative_score", "negation_detected", "irony_detected", "valence",
    "modifier", "evidence", "fallback"}.
    """
    # Aspetos a que o modelo não atribuiu polaridade (spans espúrios, como o
    # artigo "a") ficam neutros sem passar pelo sistema difuso, para não irem
    # buscar a valência de uma opinião da sua oração que não lhes pertence.
    if polarity is None:
        return {
            "crisp": 0.5, "label": "neutral", "certainty": 0.0, **_memberships(0.5),
            "negation_detected": False, "irony_detected": False,
            "valence": None, "modifier": None, "evidence": 0.0, "fallback": None,
        }

    language = detect_language(text)
    adjustment = adjust_polarity(text, aspect, opinion, polarity, confidence, language)
    adj_polarity = adjustment["polarity"]
    certainty = float(adjustment["confidence"])
    sign = {"positive": 1.0, "negative": -1.0}.get(adj_polarity, 0.0)
    evidence = sign * certainty

    # adjust_polarity só inverte por ironia quando a polaridade era positiva.
    ironic = adjustment["irony_detected"] and polarity == "positive"
    features = opinion_features(text, aspect, opinion, evidence, language, ironic=ironic)

    crisp = infer(features["valence"], features["modifier"], evidence)
    return {
        "crisp": crisp,
        "label": label_for(crisp),
        "certainty": certainty,
        **_memberships(crisp),
        "negation_detected": adjustment["negation_detected"] or features["negated"],
        "irony_detected": adjustment["irony_detected"],
        "valence": features["valence"],
        "modifier": features["modifier"],
        "evidence": evidence,
        "fallback": features["fallback"],
    }


def _clauses(text: str, language: str) -> list[str]:
    """Orações do texto, cortadas como em _local_clause (pontuação forte, vírgulas e conjunções)."""
    split_words = _CLAUSE_SPLIT_WORDS[language]
    clauses, current = [], []
    for word in _GLUED_PUNCT_RE.sub(r"\1 ", text).split():
        if current and _normalize(word) in split_words:
            clauses.append(current)
            current = []
        current.append(word)
        if _CLAUSE_BREAK_RE.search(word) or "," in word:
            clauses.append(current)
            current = []
    if current:
        clauses.append(current)
    return [" ".join(c) for c in clauses]


def score_implicit(text: str) -> dict | None:
    """
    Opinião global de uma avaliação sem nenhum aspeto explícito ("Muito bom!"),
    atribuída ao aspeto implícito IMPLICIT_ASPECT da categoria GENERAL.

    Sem evidência do modelo (E = 0), a direção e a intensidade vêm só do
    léxico: em cada oração não concessiva procura-se a palavra de opinião
    (com o seu modificador e negação) e S é a média das orações com opinião.
    A regra de ironia não é aplicada, por não haver polaridade do modelo a
    inverter. Devolve None se nenhuma oração tiver palavra do léxico.
    """
    language = detect_language(text)
    concessive = _CONCESSIVE[language]
    scored = []
    for clause in _clauses(text, language):
        first = [_normalize(w) for w in clause.split()[:2]]
        if first and (first[0] in concessive or " ".join(first) in concessive):
            continue
        features = opinion_features(clause, "", None, 0.0, language)
        if features["fallback"]:
            continue
        scored.append((infer(features["valence"], features["modifier"], 0.0), features))
    if not scored:
        return None

    crisp = sum(s for s, _ in scored) / len(scored)
    strongest = max((f for _, f in scored), key=lambda f: abs(f["valence"]))
    return {
        "crisp": crisp,
        "label": label_for(crisp),
        "certainty": IMPLICIT_CERTAINTY,
        **_memberships(crisp),
        "negation_detected": any(f["negated"] for _, f in scored),
        "irony_detected": False,
        "valence": strongest["valence"],
        "modifier": strongest["modifier"],
        "evidence": 0.0,
        "fallback": False,
        "opinion_phrase": strongest["opinion_phrase"],
    }
