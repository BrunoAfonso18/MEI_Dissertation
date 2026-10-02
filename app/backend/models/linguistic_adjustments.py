"""
Deteção de negação e ironia/sarcasmo (baseada em regras), em português e
em inglês.

O modelo de token classification (extractor.py) e o classificador de
categoria (category.py) não têm qualquer noção de contexto sintático mais
amplo: preveem a polaridade de um aspeto isoladamente, pelo que frases como
"o serviço não foi nada simpático" / "the service wasn't friendly at all" ou
"ah, claro, o empregado foi *super* rápido..." / "oh great, the waiter was
*super* fast..." (sarcasmo) tendem a ser mal classificadas. Este módulo
aplica uma camada de pós-processamento leve e determinística - sem custo de
latência adicional relevante nem necessidade de mais dados de treino - que
corrige a polaridade prevista nesses dois casos antes da fuzzificação.

Os léxicos (marcadores de negação, expressões irónicas, palavras de opinião
e de contexto, conjunções que separam orações) são definidos por idioma. O
idioma pode ser indicado explicitamente; caso contrário é inferido a partir
do próprio texto por uma heurística simples de palavras funcionais
(`detect_language`). Os léxicos não são simplesmente fundidos num só porque
há colisões entre idiomas - ex: "no" é negação em inglês mas a contração
"em + o" em português ("o serviço no balcão foi lento").

Não substitui uma solução aprendida (um classificador de negação/ironia
treinado exigiria dados anotados que não existem para este domínio); é uma
heurística explícita, documentada como tal na dissertação, que cobre os
padrões mais comuns de negação e de ironia em avaliações de restaurantes.
"""

import re

SUPPORTED_LANGUAGES = ("pt", "en")
DEFAULT_LANGUAGE = "pt"

# ── Deteção de idioma ────────────────────────────────────────────────
#
# Palavras funcionais muito frequentes e pouco ambíguas entre os dois
# idiomas (exclui-se, por exemplo, "a", que existe em ambos). Conta-se
# quantas palavras do texto pertencem a cada conjunto; em caso de empate
# (ou texto sem nenhuma) assume-se o idioma por omissão do projeto.
_PT_STOPWORDS = {
    "o", "os", "as", "de", "do", "da", "dos", "das", "que", "não", "nao",
    "é", "um", "uma", "com", "para", "foi", "estava", "muito", "mas", "em",
    "no", "na", "nos", "nas", "e", "se", "por", "mais", "comida", "serviço",
    "servico", "também", "tambem", "era", "são", "sao", "está", "esta",
}
_EN_STOPWORDS = {
    "the", "and", "is", "was", "were", "of", "to", "it", "with", "for",
    "but", "not", "very", "this", "that", "i", "we", "they", "food",
    "service", "had", "have", "my", "our", "at", "be", "been", "are",
    "too", "so", "really", "also", "would", "could",
}


def detect_language(text: str) -> str:
    """Inferência leve de idioma ("pt" ou "en") por contagem de palavras funcionais."""
    words = [_normalize(w) for w in (text or "").split()]
    pt_hits = sum(w in _PT_STOPWORDS for w in words)
    en_hits = sum(w in _EN_STOPWORDS for w in words)
    if en_hits > pt_hits:
        return "en"
    if pt_hits > en_hits:
        return "pt"
    return DEFAULT_LANGUAGE


def _resolve_language(language: str | None, sentence: str) -> str:
    """Aceita códigos como "pt", "pt-PT", "en_US"; se ausente/desconhecido, deteta."""
    if language:
        code = language.strip().lower()[:2]
        if code in SUPPORTED_LANGUAGES:
            return code
    return detect_language(sentence)


# ── Negação ──────────────────────────────────────────────────────────
#
# Marcadores de negação por idioma. A procura é feita numa janela de tokens
# que precede o termo de opinião (ou o termo de aspeto, se não houver termo
# de opinião), sem atravessar pontuação forte (".", "!", "?", ";"), para
# evitar negar um aspeto por causa de uma negação de uma oração anterior.
#
# Em inglês, as contrações ("don't", "wasn't", ...) ficam sem apóstrofo após
# `_normalize` ("dont", "wasnt"); as mais comuns estão listadas, e qualquer
# outra palavra terminada em "n't" é também tratada como negação (ver
# `_is_negation_cue`).
NEGATION_CUES = {
    "pt": {
        "não", "nao", "nunca", "jamais", "nem", "nenhum", "nenhuma", "ninguém",
        "ninguem", "nada", "tampouco", "sem",
    },
    "en": {
        "not", "no", "never", "nothing", "nobody", "none", "neither", "nor",
        "without", "hardly", "barely", "scarcely", "cannot",
        "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
        "cant", "couldnt", "wont", "wouldnt", "shouldnt", "havent", "hasnt",
        "hadnt", "aint",
    },
}
_EN_CONTRACTION_RE = re.compile(r"n['’]t\b", re.IGNORECASE)

NEGATION_WINDOW = 4  # nº máximo de tokens antes do termo a considerar
_CLAUSE_BREAK_RE = re.compile(r"[.!?;]")

# Exceção conhecida: "nunca"/"never" seguido de "tão"/"tanto" ou "so"/"this"/
# "such" é uma construção de superlativo ("nunca esteve tão boa" / "it has
# never been so good" = "está melhor do que nunca"), não uma negação - o
# oposto do padrão habitual de "nunca" + adjetivo ("nunca esteve boa" / "it
# was never good", esse sim, negativo). O mesmo vale quando o próprio termo
# é o comparativo "melhor"/"better" ("nunca esteve melhor" / "never been
# better"). Sem esta exceção, o superlativo seria invertido para negativo
# incorretamente.
_SUPERLATIVE_TRIGGERS = {"pt": {"nunca"}, "en": {"never"}}
_SUPERLATIVE_CUES = {
    "pt": {"tão", "tao", "tanto", "tanta"},
    "en": {"so", "this", "such"},
}
_COMPARATIVE_TARGETS = {"pt": {"melhor"}, "en": {"better"}}

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


def _is_negation_cue(raw_word: str, normalized: str, language: str) -> bool:
    if normalized in NEGATION_CUES[language]:
        return True
    return language == "en" and bool(_EN_CONTRACTION_RE.search(raw_word))


def detect_negation(
    sentence: str,
    aspect: str,
    opinion: str | None,
    language: str | None = None,
) -> bool:
    """
    Verifica se o termo de opinião (ou, na sua ausência, o termo de
    aspeto) é antecedido - dentro da mesma oração - por um marcador de
    negação a uma distância de até NEGATION_WINDOW palavras.
    """
    target = (opinion or aspect or "").strip()
    if not target or not sentence:
        return False

    language = _resolve_language(language, sentence)

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
            if not _is_negation_cue(search_space[i], w, language):
                continue
            if w in _SUPERLATIVE_TRIGGERS[language] and (
                any(n in _SUPERLATIVE_CUES[language] for n in normalized_space[:i])
                or target_first_word in _COMPARATIVE_TARGETS[language]
            ):
                continue  # "nunca ... tão" / "never ... so" = superlativo, não negação
            return True

    return False


# ── Ironia / sarcasmo ────────────────────────────────────────────────
#
# Deteção de ironia em texto livre é um problema em aberto na literatura de
# NLP; a heurística abaixo cobre os padrões mais frequentes em avaliações de
# restaurantes - discurso claramente sarcástico, marcado por expressões
# idiomáticas, e o padrão clássico de "contradição": um termo de opinião
# claramente positivo dentro de uma frase cujo restante conteúdo é
# fortemente negativo (ou vice-versa), o que costuma indicar sarcasmo em vez
# de sentimento misto genuíno.
#
# Em inglês não se incluem "of course", "as always" nem "as usual": ao
# contrário das expressões listadas, aparecem com muita frequência em elogios
# literais ("great as always", "of course we'll be back"), e como a ironia
# inverte sempre a polaridade positiva, o custo de um falso positivo é alto.
IRONY_PHRASES = {
    "pt": [
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
    ],
    "en": [
        "yeah right",
        "yeah, right",
        "oh great",
        "oh, great",
        "oh sure",
        "oh, sure",
        "yeah, sure",
        "just great",
        "just perfect",
        "just what i needed",
        "for a change",
        "what a surprise",
        "surprise, surprise",
        "surprise surprise",
        "thanks for nothing",
        "so much for",
        "wow, what a",
        "loved waiting",
        "loved having to wait",
        "love waiting",
    ],
}

POSITIVE_OPINION_WORDS = {
    "pt": {
        "bom", "boa", "ótimo", "otimo", "ótima", "otima", "excelente",
        "maravilhoso", "maravilhosa", "fantástico", "fantastico", "delicioso",
        "deliciosa", "incrível", "incrivel", "adorável", "adoravel", "perfeito",
        "perfeita", "rápido", "rapido", "rápida", "rapida", "simpático",
        "simpatico", "simpática", "simpatica", "agradável", "agradavel",
    },
    "en": {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "delicious", "incredible", "lovely", "perfect", "fast", "quick",
        "friendly", "nice", "pleasant", "awesome", "superb", "brilliant",
        "tasty", "outstanding",
    },
}

NEGATIVE_CONTEXT_WORDS = {
    "pt": {
        "mau", "má", "péssimo", "pessimo", "péssima", "pessima", "horrível",
        "horrivel", "terrível", "terrivel", "lento", "lenta", "frio", "fria",
        "sujo", "suja", "caro", "cara", "grosseiro", "grosseira", "insosso",
        "insossa", "queimado", "queimada", "desagradável", "desagradavel",
        "nojento", "nojenta", "demora", "demorado", "demorada", "reclamação",
        "reclamacao",
    },
    "en": {
        "bad", "awful", "terrible", "horrible", "slow", "cold", "dirty",
        "expensive", "overpriced", "rude", "bland", "tasteless", "burnt",
        "burned", "unpleasant", "disgusting", "gross", "waited",
        "waiting", "delay", "delayed", "complaint", "stale", "soggy",
    },
}

# Excesso de pontuação (ex: "ótimo!!!", "great..."), também é um sinal
# clássico de sarcasmo em texto informal.
_EXCESS_PUNCT_RE = re.compile(r"(!{2,}|\?{2,}|\.{3,})")
_QUOTED_RE = re.compile(r"[\"“'‘]\s*([^\"”'’]{2,30})\s*[\"”'’]")

# Palavras que tipicamente separam afirmações sobre aspetos distintos dentro
# da mesma frase ("o serviço foi ótimo, MAS a comida estava fria" / "the
# service was great BUT the food was cold" - dois aspetos, dois sentimentos
# genuinamente diferentes, não uma contradição sobre o mesmo aspeto). Os
# sinais de contradição/pontuação em excesso só fazem sentido quando
# restritos à oração do próprio termo, para não confundir sentimento
# genuinamente misto entre vários aspetos com ironia.
_CLAUSE_SPLIT_WORDS = {
    "pt": {"mas", "porém", "contudo", "todavia", "entanto", "e"},
    "en": {"but", "however", "although", "though", "yet", "whereas", "while", "and"},
}


def _local_clause(sentence: str, target_first_word: str, language: str) -> str:
    """
    Devolve a sub-frase (oração) que contém a primeira palavra do termo
    indicado, cortando em pontuação forte, vírgulas e conjunções que
    tipicamente introduzem uma afirmação sobre outro aspeto.
    """
    split_words = _CLAUSE_SPLIT_WORDS[language]
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
        if _CLAUSE_BREAK_RE.search(words[start - 1]) or "," in words[start - 1] or prev in split_words:
            break
        start -= 1

    end = target_idx
    while end < len(words) - 1:
        nxt = _normalize(words[end + 1])
        if _CLAUSE_BREAK_RE.search(words[end]) or "," in words[end] or nxt in split_words:
            break
        end += 1

    return " ".join(words[start:end + 1])


def detect_irony(
    sentence: str,
    aspect: str,
    opinion: str | None,
    polarity: str,
    language: str | None = None,
) -> bool:
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

    language = _resolve_language(language, sentence)

    # As expressões idiomáticas de sarcasmo (IRONY_PHRASES) são um marcador
    # de discurso que tipicamente enquadra a frase inteira (ex: "ah,
    # claro," no início de "ah, claro, o empregado foi super rápido..."),
    # pelo que esta verificação continua a ser feita sobre a frase inteira
    # - ao contrário do padrão de contradição abaixo, o risco de uma destas
    # expressões pertencer a um aspeto diferente do que está a ser avaliado
    # é baixo, dado o conjunto restrito e pouco ambíguo de expressões.
    lowered = sentence.lower().replace("’", "'")
    if any(phrase in lowered for phrase in IRONY_PHRASES[language]):
        return True

    target = (opinion or aspect or "").strip()
    target_first_word = _normalize(target.split()[0]) if target else ""
    clause = _local_clause(sentence, target_first_word, language) if target_first_word else sentence

    if opinion:
        for quoted in _QUOTED_RE.findall(sentence):
            if _normalize(quoted) == _normalize(opinion):
                return True

    if polarity == "positive":
        clause_words = {_normalize(w) for w in clause.split()}
        has_positive_opinion = bool(clause_words & POSITIVE_OPINION_WORDS[language])
        has_negative_context = bool(clause_words & NEGATIVE_CONTEXT_WORDS[language])

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
    language: str | None = None,
) -> dict:
    """
    Aplica deteção de negação e de ironia sobre a polaridade/confiança
    previstas pelo modelo para um aspeto, devolvendo valores corrigidos.

    `language` ("pt"/"en", aceitando também variantes como "pt-PT") escolhe
    os léxicos a usar; se omitido, o idioma é inferido a partir da frase
    (`detect_language`).

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
      "rápido"/"fast" é positivo em ambos os casos), pelo que o modelo pode
      prevê-lo com confiança alta nos dois cenários sem que isso indique
      que já teve em conta o sarcasmo.

    As duas heurísticas não se acumulam sobre o mesmo aspeto (a negação, a
    ser aplicada, tem prioridade) para evitar inverter a polaridade duas
    vezes.
    """
    language = _resolve_language(language, sentence)

    negated = detect_negation(sentence, aspect, opinion, language)
    ironic = False

    if negated and polarity in _POLARITY_FLIP and confidence < NEGATION_CONFIDENCE_THRESHOLD:
        polarity = _POLARITY_FLIP[polarity]
        confidence = max(0.0, confidence * 0.85)
    else:
        ironic = detect_irony(sentence, aspect, opinion, polarity, language)
        # O padrão de sarcasmo tratado aqui é assimétrico: em avaliações de
        # restaurantes, ironia quase sempre significa "elogio dito a doer" -
        # positivo à superfície, negativo na intenção. O inverso (queixa dita
        # ironicamente para significar elogio) é raro o suficiente para não
        # compensar o risco de inverter frases genuinamente negativas cujo
        # texto apenas contenha uma das expressões-gatilho (ex: "para
        # variar"/"for a change" também aparecem em queixas literais, não só
        # irónicas). Por isso só se inverte a polaridade quando esta era
        # "positive".
        if ironic and polarity == "positive":
            polarity = "negative"
            confidence = max(0.0, confidence * 0.6)

    return {
        "polarity": polarity,
        "confidence": confidence,
        "negation_detected": negated,
        "irony_detected": ironic,
        "language": language,
    }
