-- Runs automatically on first container start

CREATE TABLE IF NOT EXISTS reviews (
    id         SERIAL PRIMARY KEY,
    text       TEXT NOT NULL,
    source     VARCHAR(100) DEFAULT 'api',
    language   VARCHAR(10)  DEFAULT 'en',
    created_at TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS aspect_predictions (
    id                 SERIAL PRIMARY KEY,
    review_id          INTEGER REFERENCES reviews(id) ON DELETE CASCADE,
    aspect_term        VARCHAR(200),
    opinion_term       VARCHAR(200),
    sentiment_polarity VARCHAR(20),
    aspect_category    VARCHAR(100),
    confidence         FLOAT,
    created_at         TIMESTAMP DEFAULT NOW()
);

-- Analytical views
CREATE OR REPLACE VIEW fact_sentiments AS
SELECT
    ap.id,
    ap.review_id,
    ap.aspect_term,
    ap.aspect_category,
    ap.sentiment_polarity,
    ap.opinion_term,
    r.source,
    r.language,
    ap.created_at
FROM aspect_predictions ap
JOIN reviews r ON r.id = ap.review_id;

CREATE OR REPLACE VIEW dim_aspects AS
SELECT DISTINCT aspect_term, aspect_category
FROM aspect_predictions;