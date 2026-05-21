-- Runs automatically on first container start
-- Data Warehouse Schema (Dimensional Model)

-- ============================================
-- DIMENSION TABLES
-- ============================================

CREATE TABLE IF NOT EXISTS dim_calendar (
    date_id    DATE PRIMARY KEY,
    date       DATE NOT NULL,
    week       DATE NOT NULL,
    month      DATE NOT NULL,
    year       INTEGER NOT NULL,
    UNIQUE(date)
);

CREATE TABLE IF NOT EXISTS dim_review (
    id_review   SERIAL PRIMARY KEY,
    source      VARCHAR(100),
    text        TEXT,
    language    VARCHAR(10),
    created_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_restaurant (
    id_restaurant    INTEGER PRIMARY KEY,
    district         VARCHAR(100),
    category         VARCHAR(100),
    address          TEXT,
    name             VARCHAR(255),
    inspection_grade VARCHAR(10)
);

-- ============================================
-- FACT TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS fact_sentiment (
    fact_id             SERIAL PRIMARY KEY,
    id_review           INTEGER REFERENCES dim_review(id_review),
    id_restaurant       INTEGER REFERENCES dim_restaurant(id_restaurant),
    date_id             DATE REFERENCES dim_calendar(date_id),
    aspect_term         VARCHAR(200),
    opinion_term        VARCHAR(200),
    aspect_category     VARCHAR(100),
    fuzzy_crisp_score   FLOAT,
    sentiment_polarity  VARCHAR(20),
    confidence_score    FLOAT,
    created_at          TIMESTAMP
);

-- ============================================
-- SEED: Portuguese Restaurants
-- ============================================

INSERT INTO dim_restaurant (id_restaurant, name, district, category, address, inspection_grade) VALUES
  (1,  'O Bacalhau d''Ouro',    'Lisboa',            'Marisqueira',  'Av. da Liberdade, 120, Lisboa',              'A'),
  (2,  'Pizzeria Roma',         'Porto',             'Italiana',     'Rua de Santa Catarina, 45, Porto',           'B'),
  (3,  'Sushi Kyoto',           'Lisboa',            'Japonesa',     'Rua Garrett, 10, Lisboa',                    'A+'),
  (4,  'A Tasca do Zé',         'Braga',             'Portuguesa',   'Rua do Souto, 22, Braga',                    'B'),
  (5,  'Casa da Ribeira',       'Aveiro',            'Marisqueira',  'Rua João Mendonça, 5, Aveiro',               'A'),
  (6,  'Restaurante Solar',     'Évora',             'Alentejana',   'Praça do Giraldo, 8, Évora',                 'A'),
  (7,  'O Churrasqueiro',       'Coimbra',           'Churrascaria', 'Av. Sá da Bandeira, 30, Coimbra',            'B'),
  (8,  'Verde e Saúde',         'Lisboa',            'Vegetariana',  'Rua do Loreto, 15, Lisboa',                  'A+'),
  (9,  'Hamburgueria Lisboeta', 'Lisboa',            'Hamburguer',   'Av. Almirante Reis, 55, Lisboa',             'B'),
  (10, 'Quinta da Palha',       'Sintra',            'Fine Dining',  'Estrada da Pena, 2, Sintra',                 'A+'),
  (11, 'Mar e Sol',             'Faro',              'Marisqueira',  'Rua de Santo António, 12, Faro',             'A'),
  (12, 'Cervejaria Central',    'Porto',             'Portuguesa',   'Rua do Bonjardim, 100, Porto',               'B'),
  (13, 'Cantinho Algarvio',     'Faro',              'Algarvia',     'Av. Tomás Cabreira, 78, Portimão',           'A'),
  (14, 'Taberna do Norte',      'Viana do Castelo',  'Portuguesa',   'Rua Grande, 33, Viana do Castelo',           'B'),
  (15, 'Restaurante Douro',     'Porto',             'Portuguesa',   'Av. de Diogo Leite, 22, Vila Nova de Gaia',  'A')
ON CONFLICT (id_restaurant) DO NOTHING;
