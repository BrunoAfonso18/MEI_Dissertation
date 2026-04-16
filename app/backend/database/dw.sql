-- Dimension: aspects
CREATE VIEW dim_aspects AS
SELECT DISTINCT
    aspect_term,
    aspect_category
FROM aspect_predictions;

-- Dimension: time
CREATE VIEW dim_time AS
SELECT DISTINCT
    DATE_TRUNC('day', created_at)  AS day,
    DATE_TRUNC('week', created_at) AS week,
    DATE_TRUNC('month', created_at) AS month
FROM aspect_predictions;

-- Fact table: one row per aspect prediction
CREATE VIEW fact_sentiments AS
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