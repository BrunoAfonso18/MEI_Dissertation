"""
Categorização de termos de aspeto por léxico + comparação difusa de strings.

Substitui o classificador zero-shot anterior (facebook/bart-large-mnli, o
maior fator de latência do pipeline) por uma abordagem determinística e
muito mais rápida: um dicionário de palavras-chave em português por
categoria, comparado ao termo de aspeto extraído através de fuzzy string
matching (rapidfuzz). Isto apanha variações ortográficas, plurais, géneros e
palavras derivadas do mesmo radical (ex: "servente" -> SERVICE, via
"serviço") sem precisar de listar exaustivamente cada variante no
dicionário. Termos que não atinjam o limiar de semelhança com nenhuma
palavra-chave conhecida ficam na categoria de reserva GENERAL.
"""

from rapidfuzz import fuzz, process

CATEGORIES = [
    "FOOD_QUALITY",
    "SERVICE",
    "PRICE_AND_VALUE",
    "AMBIENCE_AND_ATMOSPHERE",
    "LOCATION",
    "PORTION_SIZE",
    "MENU_VARIETY",
    "CLEANLINESS",
    "WAITING_TIME",
    "GENERAL",
]

# Palavras-chave em português, uma lista por categoria. Não precisam de
# cobrir cada variante morfológica (plural, género, acentuação) - isso é
# tratado pela comparação difusa; servem sobretudo para cobrir sinónimos e
# termos relacionados que não são ortograficamente semelhantes entre si
# (ex: "caro" e "preço").
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "FOOD_QUALITY": [
        "comida", "prato", "sabor", "tempero", "cozinha", "refeição", "entrada",
        "sobremesa", "petisco", "aperitivo", "bacalhau", "carne", "peixe", "marisco",
        "sushi", "pizza", "massa", "molho", "ingrediente", "receita", "chef",
        "cozinheiro", "confeção", "fresco", "frescura", "saboroso", "insosso",
        "queimado", "cru", "delicioso", "saboroso", "condimento",
    ],
    "SERVICE": [
        "serviço", "atendimento", "empregado", "empregada", "garçom", "garçonete",
        "servente", "funcionário", "staff", "equipa", "simpatia", "simpático",
        "gentileza", "profissionalismo", "cortesia", "atencioso", "atenção",
        "receção", "anfitrião", "disponibilidade", "educado", "grosseiro",
        "prestável", "amável",
    ],
    "PRICE_AND_VALUE": [
        "preço", "valor", "custo", "caro", "barato", "conta", "pagamento",
        "dinheiro", "orçamento", "económico", "dispendioso", "promoção",
        "desconto",
    ],
    "AMBIENCE_AND_ATMOSPHERE": [
        "ambiente", "atmosfera", "decoração", "música", "iluminação", "ruído",
        "barulho", "silêncio", "conforto", "confortável", "vista", "espaço",
        "mobiliário", "cadeira", "mesa", "temperatura", "climatização",
        "romântico", "acolhedor", "aconchegante",
    ],
    "LOCATION": [
        "localização", "local", "zona", "rua", "bairro", "estacionamento",
        "parqueamento", "acesso", "transporte", "centro", "distância", "perto",
        "longe", "vizinhança", "endereço",
    ],
    "PORTION_SIZE": [
        "dose", "quantidade", "porção", "tamanho", "fartura", "generoso",
        "abundante", "escasso",
    ],
    "MENU_VARIETY": [
        "menu", "ementa", "variedade", "opções", "cardápio", "escolha",
        "alternativas", "vegetariano", "vegan",
    ],
    "CLEANLINESS": [
        "limpeza", "higiene", "sujo", "limpo", "casa de banho", "wc", "toalete",
        "arrumado", "imundo", "poeira",
    ],
    "WAITING_TIME": [
        "espera", "demora", "demorado", "fila", "atraso", "lento", "atrasado",
        "imediato", "rápido",
    ],
}

SIMILARITY_THRESHOLD = 70  # pontuação rapidfuzz (0-100); abaixo disto, cai em GENERAL


class CategoryClassifier:
    def __init__(self):
        self._keyword_to_category: dict[str, str] = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                self._keyword_to_category[keyword] = category
        self._keywords = list(self._keyword_to_category.keys())

    def _classify_term(self, term: str) -> str:
        term = (term or "").strip().lower()
        if not term:
            return "GENERAL"
        match = process.extractOne(term, self._keywords, scorer=fuzz.WRatio)
        if match is None or match[1] < SIMILARITY_THRESHOLD:
            return "GENERAL"
        matched_keyword = match[0]
        return self._keyword_to_category[matched_keyword]

    def predict(self, aspect: str, sentence: str = "") -> str:
        """
        `sentence` é aceite por compatibilidade com a assinatura do antigo
        classificador zero-shot, mas não é usada - a categorização é feita
        apenas a partir do termo de aspeto.
        """
        return self._classify_term(aspect)

    def predict_batch(self, aspects: list[str], sentence: str = "") -> list[str]:
        return [self._classify_term(aspect) for aspect in aspects]
