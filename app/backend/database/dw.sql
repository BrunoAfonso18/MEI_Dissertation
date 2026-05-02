-- Dimension: review
CREATE VIEW dim_review AS
SELECT
    id,
    text,
    source,
    language,
    created_at
FROM reviews;

-- Dimension: time
CREATE VIEW dim_time AS
SELECT
    DATE_TRUNC('day', created_at) AS date,
    DATE_TRUNC('week', created_at) AS week,
    DATE_TRUNC('month', created_at) AS month,
    EXTRACT(YEAR FROM created_at) AS year,
    created_at
FROM reviews;

-- Fact table: one row per aspect-opinion pair
CREATE VIEW fact_sentiment AS
SELECT
    ap.id,
    r.id AS review_id,
    ap.aspect_term,
    ap.opinion_term,
    ap.aspect_category,
    ap.positive_score,
    ap.neutral_score,
    ap.negative_score,
    ap.sentiment_polarity,
    ap.confidence,
    ap.created_at,
    DATE_TRUNC('day', ap.created_at) AS date,
    DATE_TRUNC('week', ap.created_at) AS week,
    DATE_TRUNC('month', ap.created_at) AS month,
    EXTRACT(YEAR FROM ap.created_at) AS year
FROM aspect_predictions ap
JOIN reviews r ON r.id = ap.review_id;