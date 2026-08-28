"""
Deteção de negação e ironia/sarcasmo (baseada em regras).

O modelo de token classification (extractor.py) e o classificador de
categoria (category.py) não têm qualquer noção de contexto sintático mais
amplo: preveem a polaridade de um aspeto isoladamente, pelo que frases como
"o serviço não foi nada simpático" ou "ah, claro, o empregado foi *super*
rápido..." (sarcasmo) tendem a ser mal classificadas. Este módulo aplica uma
camada de pós-processamento leve e determinística - sem custo de latência
adicional relevante nem necessidade de mais dados de treino - que corrige a
polaridade prevista nesses dois casos antes da fuzzificação.

Não substitui uma solução aprendida (um classificador de negação/ironia
treinado exigiria dados anotados que não existem para este domínio em
português); é uma heurística explícita, documentada como tal na dissertação,
que cobre os padrões mais comuns de negação e de ironia em avaliações de
restaurantes.
"""

import re

# ── Negação ──────────────────────────────────────────────────────────
#
# Marcadores de negação em português. A procura é feita numa janela de
# tokens que precede o termo de opinião (ou o termo de aspeto, se não houver
# termo de opinião), sem atravessar pontuação forte (".", "!", "?", ";"),
# para evitar negar um aspeto por causa de uma negação de uma oração anterior.
NEGATION_CUES = {
    "não", "nao", "nunca", "jamais", "nem", "nenhum", "nenhuma", "ninguém",
    "ninguem", "nada", "tampouco", "sem",
}

NEGATION_WINDOW = 4  # nº máximo de tokens antes do termo a considerar
_CLAUSE_BREAK_RE = re.compile(r"[.!?;]")

# Exceção conhecida: "nunca" seguido de "tão"/"tanto" é, em português, uma
# construção de superlativo ("nunca esteve tão boa" = "está melhor do que
# nunca esteve"), não uma negação - o oposto do padrão habitual de "nunca" +
# adjetivo ("nunca esteve boa" esse sim, negativo). Sem esta exceção, o
# superlativo seria invertido para negativo incorretamente.
_SUPERLATIVE_CUES = {"tão", "tao", "tanto", "tanta"}

_POLARITY_FLIP = {"positive": "negative", "negative": "positive"}

# O modelo de extração já vê a frase inteira (não só o termo isolado), pelo
# que muitas vezes já resolve a negação corretamente por si só - testes
# manuais mostram previsões como "servico" -> negative com confiança 0.91
# para "o serviço não foi nada simpático", em que o próprio modelo já
# inverteu corretamente a leitura de superfície de "simpático". Inverter a
# polaridade sempre que se deteta um marcador de negação, sem olhar à
# confiança do modelo, faz mais mal do que bem nesses casos (uma dupla
# negação acidental). Por isso só se aplica a correção heurística quando a
# confiança do modelo é baixa - um sinal de que a previsão é mais provável
# de não ter tido em conta a negação.
NEGATION_CONFIDENCE_THRESHOLD = 0.8


def _normalize(word: str) -> str:
    return re.sub(r"[^\wà-úÀ-Ú]+", "", word.lower())


def detect_negation(sentence: str, aspect: str, opinion: str | None) -> bool:
    """
    Verifica se o termo de opinião (ou, na sua ausência, o termo de
    aspeto) é antecedido - dentro da mesma oração - por um marcador de
    negação a uma distância de até NEGATION_WINDOW palavras.
    """
    target = (opinion or aspect or "").strip()
    if not target or not sentence:
        return False

    words = sentence.split()
    target_first_word = _normalize(target.split()[0])
    if not target_first_word:
        return False

    for idx, raw_word in enumerate(words):
        if _normalize(raw_word) != target_first_word:
            continue

        window_start = max(0, idx - NEGATION_WINDOW)
        window_words = words[window_start:idx]

        # Não atravessar uma quebra de oração (outro sinal de pontuação forte)
        # dentro da janela - a negação tem de pertencer à mesma oração do
        # termo. Percorrendo de trás para a frente, paramos assim que
        # encontramos pontuação forte: tudo o que vem antes pertence a outra
        # oração e é ignorado.
        search_space = []
        for w in reversed(window_words):
            if _CLAUSE_BREAK_RE.search(w):
                break
            search_space.append(w)

        normalized_space = [_normalize(w) for w in search_space]
        for i, w in enumerate(normalized_space):
            if w not in NEGATION_CUES:
                continue
            if w == "nunca" and any(n in _SUPERLATIVE_CUES for n in normalized_space[:i]):
                continue  # "nunca ... tão/tanto" = superlativo, não negação
            return True

    return False


# ── Ironia / sarcasmo ────────────────────────────────────────────────
#
# Deteção de ironia em texto livre é um problema em aberto na literatura de
# NLP; a heurística abaixo cobre os padrões mais frequentes em avaliações de
# restaurantes em português - discurso claramente sarcástico, marcado por
# expressões idiomáticas, e o padrão clássico de "contradição": um termo de
# opinião claramente positivo dentro de uma frase cujo restante conteúdo é
# fortemente negativo (ou vice-versa), o que costuma indicar sarcasmo em vez
# de sentimento misto genuíno.
IRONY_PHRASES = [
    "claro que",
    "pois é",
    "pois é...",
    "para variar",
    "como sempre",
    "não podia ser melhor",
    "não podia ser pior",
    "que surpresa",
    "surpresa das surpresas",
    "uau, que",
    "sim, claro",
    "ah, claro",
    "adorei esperar",
    "adorei ter de esperar",
]

POSITIVE_OPINION_WORDS = {
    "bom", "boa", "ótimo", "otimo", "ótima", "otima", "excelente",
    "maravilhoso", "maravilhosa", "fantástico", "fantastico", "delicioso",
    "deliciosa", "incrível", "incrivel", "adorável", "adoravel", "perfeito",
    "perfeita", "rápido", "rapido", "rápida", "rapida", "simpático",
    "simpatico", "simpática", "simpatica", "agradável", "agradavel",
}

NEGATIVE_CONTEXT_WORDS = {
    "mau", "má", "péssimo", "pessimo", "péssima", "pessima", "horrível",
    "horrivel", "terrível", "terrivel", "lento", "lenta", "frio", "fria",
    "sujo", "suja", "caro", "cara", "grosseiro", "grosseira", "insosso",
    "insossa", "queimado", "queimada", "desagradável", "desagradavel",
    "nojento", "nojenta", "demora", "demorado", "demorada", "reclamação",
    "reclamacao",
}

# Excesso de pontuação (ex: "ótimo!!!", "incrível...") também é um sinal
# clássico de sarcasmo em texto informal.
_EXCESS_PUNCT_RE = re.compile(r"(!{2,}|\?{2,}|\.{3,})")
_QUOTED_RE = re.compile(r"[\"“'‘]\s*([^\"”'’]{2,30})\s*[\"”'’]")

# Palavras que tipicamente separam afirmações sobre aspetos distintos dentro
# da mesma frase ("o serviço foi ótimo, MAS a comida estava fria" - dois
# aspetos, dois sentimentos genuinamente diferentes, não uma contradição
# sobre o mesmo aspeto). Os sinais de contradição/pontuação em excesso só
# fazem sentido quando restritos à oração do próprio termo, para não
# confundir sentimento genuinamente misto entre vários aspetos com ironia.
_CLAUSE_SPLIT_WORDS = {"mas", "porém", "contudo", "todavia", "entanto", "e"}


def _local_clause(sentence: str, target_first_word: str) -> str:
    """
    Devolve a sub-frase (oração) que contém a primeira palavra do termo
    indicado, cortando em pontuação forte, vírgulas e conjunções que
    tipicamente introduzem uma afirmação sobre outro aspeto.
    """
    words = sentence.split()
    target_idx = next(
        (i for i, w in enumerate(words) if _normalize(w) == target_first_word),
        None,
    )
    if target_idx is None:
        return sentence

    start = target_idx
    while start > 0:
        prev = _normalize(words[start - 1])
        if _CLAUSE_BREAK_RE.search(words[start - 1]) or "," in words[start - 1] or prev in _CLAUSE_SPLIT_WORDS:
            break
        start -= 1

    end = target_idx
    while end < len(words) - 1:
        nxt = _normalize(words[end + 1])
        if _CLAUSE_BREAK_RE.search(words[end]) or "," in words[end] or nxt in _CLAUSE_SPLIT_WORDS:
            break
        end += 1

    return " ".join(words[start:end + 1])


def detect_irony(sentence: str, aspect: str, opinion: str | None, polarity: str) -> bool:
    """
    Sinaliza como potencialmente irónica uma frase que combine um termo de
    opinião positivo (previsto) com fortes indícios contextuais negativos,
    ou que contenha uma expressão idiomática de sarcasmo, ou em que o
    próprio termo de opinião apareça entre aspas (aspas de distanciamento -
    "scare quotes" - um marcador comum de ironia escrita). Todas as
    verificações à exceção das aspas de distanciamento são restringidas à
    oração do próprio termo (secção `_local_clause`), para não confundir
    sentimento genuinamente misto entre vários aspetos da mesma frase com
    ironia sobre um único aspeto.
    """
    if not sentence:
        return False

    # As expressões idiomáticas de sarcasmo (IRONY_PHRASES) são um marcador
    # de discurso que tipicamente enquadra a frase inteira (ex: "ah,
    # claro," no início de "ah, claro, o empregado foi super rápido..."),
    # pelo que esta verificação continua a ser feita sobre a frase inteira
    # - ao contrário do padrão de contradição abaixo, o risco de uma destas
    # expressões pertencer a um aspeto diferente do que está a ser avaliado
    # é baixo, dado o conjunto restrito e pouco ambíguo de expressões.
    if any(phrase in sentence.lower() for phrase in IRONY_PHRASES):
        return True

    target = (opinion or aspect or "").strip()
    target_first_word = _normalize(target.split()[0]) if target else ""
    clause = _local_clause(sentence, target_first_word) if target_first_word else sentence

    if opinion:
        for quoted in _QUOTED_RE.findall(sentence):
            if _normalize(quoted) == _normalize(opinion):
                return True

    if polarity == "positive":
        clause_words = {_normalize(w) for w in clause.split()}
        has_positive_opinion = bool(clause_words & POSITIVE_OPINION_WORDS)
        has_negative_context = bool(clause_words & NEGATIVE_CONTEXT_WORDS)

        if _EXCESS_PUNCT_RE.search(clause) and has_negative_context:
            return True
        if has_positive_opinion and has_negative_context:
            return True

    return False


# ── Ponto de entrada único ────────────────────────────────────────────

def adjust_polarity(
    sentence: str,
    aspect: str,
    opinion: str | None,
    polarity: str,
    confidence: float,
) -> dict:
    """
    Aplica deteção de negação e de ironia sobre a polaridade/confiança
    previstas pelo modelo para um aspeto, devolvendo valores corrigidos.

    - Negação confirmada: só inverte positive<->negative se a confiança do
      modelo for inferior a NEGATION_CONFIDENCE_THRESHOLD (ver comentário
      acima) - caso contrário, assume-se que o modelo já resolveu a
      negação corretamente e a previsão não é alterada. Quando inverte,
      reduz ligeiramente a confiança, refletindo a incerteza acrescida de
      uma correção heurística.
    - Ironia detetada: também inverte positive<->negative (o padrão mais
      comum é elogio sarcástico = crítica real) e reduz a confiança de
      forma mais acentuada, já que a deteção de ironia é menos fiável do
      que a de negação. Ao contrário da negação, não há aqui um sinal de
      confiança equivalente para decidir não intervir: um termo de opinião
      sarcástico é, à superfície, exatamente igual a um termo genuíno (ex:
      "rápido" é positivo em ambos os casos), pelo que o modelo pode
      prevê-lo com confiança alta nos dois cenários sem que isso indique
      que já teve em conta o sarcasmo.

    As duas heurísticas não se acumulam sobre o mesmo aspeto (a negação, a
    ser aplicada, tem prioridade) para evitar inverter a polaridade duas
    vezes.
    """
    negated = detect_negation(sentence, aspect, opinion)
    ironic = False

    if negated and polarity in _POLARITY_FLIP and confidence < NEGATION_CONFIDENCE_THRESHOLD:
        polarity = _POLARITY_FLIP[polarity]
        confidence = max(0.0, confidence * 0.85)
    else:
        ironic = detect_irony(sentence, aspect, opinion, polarity)
        # O padrão de sarcasmo tratado aqui é assimétrico: em avaliações de
        # restaurantes, ironia quase sempre significa "elogio dito a doer" -
        # positivo à superfície, negativo na intenção. O inverso (queixa dita
        # ironicamente para significar elogio) é raro o suficiente para não
        # compensar o risco de inverter frases genuinamente negativas cujo
        # texto apenas contenha uma das expressões-gatilho (ex: "para
        # variar" também aparece em queixas literais, não só irónicas).
        # Por isso só se inverte a polaridade quando esta era "positive".
        if ironic and polarity == "positive":
            polarity = "negative"
            confidence = max(0.0, confidence * 0.6)

    return {
        "polarity": polarity,
        "confidence": confidence,
        "negation_detected": negated,
        "irony_detected": ironic,
    }
