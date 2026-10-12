"""
Léxico de opinião: calcula as entradas lexicais do sistema difuso
(models/fuzzy_inference.py) para um aspeto.

- V (valência da opinião, em [-1, 1]): valor da palavra/expressão de opinião
  no léxico (VADER para inglês, LeIA para português, com o léxico de domínio
  a sobrepor-se ao LeIA), após negação (deslocamento, Taboada et al. 2011) e
  ironia.
- M (modificador de grau, em [0, 2], 1 = nenhum): grau do intensificador ou
  atenuador que antecede a opinião ("ligeiramente" 0.5, "extremamente" 1.9).

A palavra de opinião é procurada apenas na oração do aspeto (`_local_clause`):
primeiro o termo de opinião previsto pelo modelo, se estiver nessa oração;
senão a palavra do léxico com maior valência absoluta; se não houver
nenhuma, usa-se V = 0.6·E (direção do modelo com intensidade moderada).
"""

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from models.linguistic_adjustments import (
    _EN_STOPWORDS,
    _PT_STOPWORDS,
    NEGATION_CUES,
    _local_clause,
    _normalize,
    detect_negation,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

MAX_NGRAM = 4           # nº máximo de palavras de uma expressão do léxico
MODIFIER_WINDOW = 3     # nº de palavras antes da opinião onde se procura o modificador
NEGATION_SHIFT = 0.8    # deslocamento da valência por negação
FALLBACK_FACTOR = 0.6   # V = 0.6·E quando não há palavra do léxico na oração

_TOKEN_RE = re.compile(r"[^\w\-]+")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _key(text: str) -> str:
    """Chave de consulta: minúsculas, sem acentos nem pontuação (exceto hífen)."""
    return " ".join(t for t in (_TOKEN_RE.sub("", w) for w in _fold(text).split()) if t)


def _load_tsv(name: str) -> dict[str, float]:
    table = {}
    with open(DATA_DIR / name, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            term, value = line.rstrip("\n").split("\t")[:2]
            # Ignora entradas com pontuação (emoticons como ":o" ou ":D"), que
            # sem a pontuação colidiriam com palavras ou letras soltas.
            if not term.strip() or _key(term) != " ".join(_fold(term).split()):
                continue
            table[_key(term)] = float(value)
    return table


@lru_cache(maxsize=None)
def _resources(language: str) -> tuple[dict[str, float], dict[str, float]]:
    lexicon = _load_tsv(f"lexicon_{language}.tsv")
    if language == "pt":
        lexicon.update(_load_tsv("lexicon_domain_pt.tsv"))
    modifiers = _load_tsv(f"modifiers_{language}.tsv")
    # Modificadores, marcadores de negação e palavras funcionais não são
    # palavras de opinião (ex: o VADER atribui valência a "no" e o LeIA a
    # "mas"). Também se excluem as expressões que começam por um modificador
    # (ex: "muito caro" no LeIA, traduzido de "dear"), porque o grau é tratado
    # à parte, pela variável M.
    stopwords = _PT_STOPWORDS if language == "pt" else _EN_STOPWORDS
    for word in list(modifiers) + [_key(w) for w in NEGATION_CUES[language] | stopwords]:
        lexicon.pop(word, None)
    for term in list(lexicon):
        if len(term) < 2 or any(term.startswith(m + " ") for m in modifiers):
            del lexicon[term]
    return lexicon, modifiers


def _find(tokens: list[str], table: dict[str, float], start: int) -> tuple[int, float] | None:
    """Expressão mais longa de `table` que começa em tokens[start] -> (nº de palavras, valor)."""
    for n in range(min(MAX_NGRAM, len(tokens) - start), 0, -1):
        value = table.get(" ".join(tokens[start:start + n]))
        if value is not None:
            return n, value
    return None


def _best_in(tokens: list[str], lexicon: dict[str, float], lo: int, hi: int, skip: set[int]):
    """Expressão do léxico com maior |valência| entre tokens[lo:hi] -> (início, nº de palavras, valência)."""
    best = None
    for i in range(lo, hi):
        if i in skip:
            continue
        hit = _find(tokens[:hi], lexicon, i)
        if hit and (best is None or abs(hit[1]) > abs(best[2])):
            best = (i, hit[0], hit[1])
    return best


def _locate(tokens: list[str], phrase: list[str]) -> int | None:
    for i in range(len(tokens) - len(phrase) + 1):
        if tokens[i:i + len(phrase)] == phrase:
            return i
    return None


def _modifier(tokens: list[str], modifiers: dict[str, float], start: int) -> float:
    """Grau do modificador mais próximo nas MODIFIER_WINDOW palavras antes de `start`."""
    for i in range(start - 1, max(-1, start - 1 - MODIFIER_WINDOW), -1):
        for n in range(min(MAX_NGRAM, start - i), 0, -1):
            degree = modifiers.get(" ".join(tokens[i:i + n]))
            if degree is not None and i + n <= start:
                return degree
    return 1.0


def opinion_features(
    text: str,
    aspect: str,
    opinion: str | None,
    evidence: float,
    language: str,
    ironic: bool = False,
) -> dict:
    """
    Devolve V, M e a origem da valência para um aspeto:
    {"valence", "modifier", "opinion_word", "fallback", "negated"}.

    `evidence` é E (sinal da polaridade x confiança, após adjust_polarity) e
    só é usada no fallback. `ironic` indica que adjust_polarity inverteu a
    polaridade por ironia, caso em que V' = -|V|.
    """
    lexicon, modifiers = _resources(language)

    aspect_first = _normalize(aspect.split()[0]) if aspect.strip() else ""
    clause = _local_clause(text, aspect_first, language) if aspect_first else text
    # Palavras da oração (originais, para a negação) e respetivas chaves de consulta.
    words = [w for w in clause.split() if _key(w)]
    tokens = [_key(w) for w in words]

    # Posições do próprio aspeto na oração (não contam como opinião).
    aspect_tokens = _key(aspect).split()
    skip = set()
    pos = _locate(tokens, aspect_tokens) if aspect_tokens else None
    if pos is not None:
        skip = set(range(pos, pos + len(aspect_tokens)))

    best = None
    # 1. Termo de opinião do modelo, se estiver na oração do aspeto.
    opinion_tokens = _key(opinion).split() if opinion else []
    if opinion_tokens:
        pos = _locate(tokens, opinion_tokens)
        if pos is not None:
            best = _best_in(tokens, lexicon, pos, pos + len(opinion_tokens), skip)
    # 2. Palavra do léxico com maior |valência| na oração.
    if best is None:
        best = _best_in(tokens, lexicon, 0, len(tokens), skip)
    # 3. Fallback: direção do modelo com intensidade moderada.
    if best is None:
        return {
            "valence": FALLBACK_FACTOR * evidence,
            "modifier": 1.0,
            "opinion_word": None,
            "fallback": True,
            "negated": False,
        }

    start, length, valence = best
    word = " ".join(words[start:start + length])
    modifier = _modifier(tokens, modifiers, start)

    negated = detect_negation(clause, aspect, word, language)
    if negated and valence != 0:
        # Deslocamento em direção ao polo oposto ("não é excelente" 0.9 -> 0.1,
        # "não é má" -0.6 -> 0.2). O modificador fica sob o âmbito da negação
        # ("não é muito bom" não é "muito mau"), pelo que deixa de contar.
        valence = valence - NEGATION_SHIFT * (1 if valence > 0 else -1)
        modifier = 1.0
    if ironic:
        valence = -abs(valence)

    return {
        "valence": max(-1.0, min(1.0, valence)),
        "modifier": modifier,
        "opinion_word": _key(word),
        "fallback": False,
        "negated": negated,
    }
