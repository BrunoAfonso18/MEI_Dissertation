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
        "comida", "prato", "sabor", "tempero", "cozinha", "refeição", "entrada", "sobremesa", "petisco", "aperitivo", "acompanhamento", "guarnição", "bacalhau", "carne", "peixe", "marisco", "camarão", "sushi", "pizza", "massa", "arroz", "sopa", "salada", "molho", "ingrediente", "receita", "chef", "cozinheiro", "confeção", "apresentação", "textura", "aroma", "condimento", "especiaria", "fresco", "frescura", "saboroso", "insosso", "queimado", "cru", "mal passado", "bem passado", "delicioso", "requintado", "saudável", "calórico", "vegetariano", "vegan", "cozido", "grelhado", "frito", "assado", "estufado",
        "food", "dish", "flavor", "flavour", "seasoning", "kitchen", "cuisine", "meal", "starter", "appetizer", "dessert", "snack", "side dish", "garnish", "cod", "meat", "fish", "seafood", "shrimp", "prawn", "sushi", "pizza", "pasta", "rice", "soup", "salad", "sauce", "ingredient", "recipe", "chef", "cook", "presentation", "texture", "aroma", "spice", "condiment", "fresh", "freshness", "tasty", "bland", "burnt", "raw", "undercooked", "overcooked", "delicious", "exquisite", "healthy", "greasy", "vegetarian", "vegan", "boiled", "grilled", "fried", "roasted", "stewed",
    ],
    "SERVICE": [
        "serviço", "atendimento", "empregado", "empregada", "garçom", "garçonete", "criado", "servente", "funcionário", "staff", "equipa", "pessoal", "simpatia", "simpático", "gentileza", "profissionalismo", "cortesia", "atencioso", "atenção", "receção", "rececionista", "anfitrião", "anfitriã", "disponibilidade", "educado", "mal-educado", "grosseiro", "prestável", "amável", "eficiente", "rápido", "lento", "recomendação", "sugestão", "reclamação", "queixa", "gorjeta", "dono", "gerente", "gerência",
        "service", "staff", "waiter", "waitress", "server", "employee", "crew", "team", "friendliness", "friendly", "kindness", "professionalism", "courtesy", "attentive", "attention", "reception", "receptionist", "host", "hostess", "availability", "polite", "rude", "helpful", "kind", "efficient", "quick", "slow", "recommendation", "suggestion", "complaint", "tip", "owner", "manager", "management",
    ],
    "PRICE_AND_VALUE": [
        "preço", "valor", "custo", "caro", "barato", "conta", "fatura", "pagamento", "dinheiro", "orçamento", "económico", "dispendioso", "promoção", "desconto", "oferta", "menu do dia", "preço fixo", "relação qualidade-preço", "sobrepreço", "gratuito", "cartão", "dinheiro vivo", "gorjeta incluída", "IVA",
        "price", "cost", "value", "expensive", "cheap", "affordable", "bill", "check", "payment", "cash", "budget", "economical", "pricey", "promotion", "discount", "deal", "offer", "fixed price", "set menu", "value for money", "overpriced", "free", "card", "tip included",
    ],
    "AMBIENCE_AND_ATMOSPHERE": [
        "ambiente", "atmosfera", "decoração", "música", "iluminação", "ruído", "barulho", "silêncio", "conforto", "confortável", "vista", "espaço", "mobiliário", "cadeira", "mesa", "temperatura", "climatização", "ar condicionado", "romântico", "acolhedor", "aconchegante", "elegante", "moderno", "rústico", "íntimo", "animado", "sofisticado", "casual", "informal", "descontraído", "esplanada", "terraço", "lotação", "lotado", "sossegado",
        "ambience", "ambiance", "atmosphere", "décor", "decor", "decoration", "music", "lighting", "noise", "silence", "comfort", "comfortable", "view", "space", "furniture", "chair", "table", "temperature", "air conditioning", "romantic", "cozy", "cosy", "elegant", "modern", "rustic", "intimate", "lively", "upscale", "casual", "relaxed", "terrace", "patio", "crowded", "quiet",
    ],
    "LOCATION": [
        "localização", "local", "zona", "rua", "bairro", "morada", "endereço", "estacionamento", "parqueamento", "acesso", "transporte", "transportes públicos", "metro", "autocarro", "centro", "centro da cidade", "distância", "perto", "longe", "vizinhança", "esquina", "praça", "marginal", "baixa", "beira-mar", "vista para o mar", "fácil de encontrar",
        "location", "spot", "area", "street", "neighborhood", "neighbourhood", "address", "parking", "access", "transport", "public transport", "subway", "metro", "bus", "downtown", "city center", "city centre", "distance", "nearby", "far", "corner", "square", "waterfront", "seaview", "sea view", "easy to find",
    ],
    "PORTION_SIZE": [
        "dose", "quantidade", "porção", "tamanho", "fartura", "generoso", "abundante", "escasso", "pequeno", "grande", "meia-dose", "dose completa", "insuficiente", "satisfatório", "tamanho da dose",
        "portion", "serving", "size", "quantity", "amount", "generous", "abundant", "scarce", "small", "large", "half portion", "full portion", "insufficient", "satisfying", "serving size",
    ],
    "MENU_VARIETY": [
        "menu", "ementa", "variedade", "opções", "cardápio", "escolha", "alternativas", "vegetariano", "vegan", "sem glúten", "opções para crianças", "menu infantil", "prato do dia", "especialidade", "especialidades da casa", "carta", "carta de vinhos", "menu de degustação", "variado", "limitado", "extenso", "reduzido",
        "menu", "variety", "options", "choice", "alternatives", "vegetarian", "vegan", "gluten-free", "kids menu", "children's menu", "daily special", "specialty", "speciality", "house specialty", "wine list", "tasting menu", "varied", "limited", "extensive", "restricted",
    ],
    "CLEANLINESS": [
        "limpeza", "higiene", "sujo", "limpo", "casa de banho", "wc", "toalete", "lavabo", "arrumado", "imundo", "poeira", "nojento", "impecável", "cheiro", "mau cheiro", "cuidado", "manutenção",
        "cleanliness", "hygiene", "dirty", "clean", "bathroom", "restroom", "toilet", "washroom", "tidy", "filthy", "dust", "disgusting", "spotless", "smell", "bad smell", "odor", "odour", "upkeep", "maintenance",
    ],
    "WAITING_TIME": [
        "espera", "demora", "demorado", "fila", "atraso", "lento", "atrasado", "imediato", "rápido", "tempo de espera", "fila de espera", "reserva", "marcação", "lista de espera", "meia hora", "sem demora",
        "wait", "waiting time", "delay", "slow", "late", "queue", "line", "immediate", "quick", "fast", "reservation", "booking", "waitlist", "half an hour", "promptly", "without delay",
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
