-- Runs automatically on first container start
-- Data Warehouse Schema (Dimensional Model)

-- ============================================
-- DATA WAREHOUSE DIMENSION TABLES
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
    id_review   INTEGER PRIMARY KEY,
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
-- DATA WAREHOUSE FACT TABLE
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

CREATE TABLE IF NOT EXISTS reviews (
    id              SERIAL PRIMARY KEY,
    restaurant_id   INTEGER REFERENCES restaurants(id) ON DELETE RESTRICT,
    text            TEXT NOT NULL,
    source          VARCHAR(100) DEFAULT 'api',
    language        VARCHAR(10)  DEFAULT 'en',
    created_at      TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS aspect_predictions (
    id                 SERIAL PRIMARY KEY,
    review_id          INTEGER REFERENCES reviews(id) ON DELETE CASCADE,
    aspect_term        VARCHAR(200),
    opinion_term       VARCHAR(200),
    sentiment_polarity VARCHAR(20),
    aspect_category    VARCHAR(100),
    confidence         FLOAT,
    pos_score          FLOAT,
    neut_score         FLOAT,
    neg_score          FLOAT,
    fuzzy_crisp_score  FLOAT,
    created_at         TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- DATA WAREHOUSE DIMENSION TABLES
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
    id_review   INTEGER PRIMARY KEY,
    source      VARCHAR(100),
    text        TEXT,
    language    VARCHAR(10),
    created_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_restaurant (
    id_restaurant   INTEGER PRIMARY KEY,
    district        VARCHAR(100),
    category        VARCHAR(100),
    address         TEXT,
    name            VARCHAR(255),
    inspection_grade VARCHAR(10)
);

-- ============================================
-- DATA WAREHOUSE FACT TABLE
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