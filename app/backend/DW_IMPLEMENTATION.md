# Backend Implementation Guide - Data Warehouse Integration

## Overview

The backend has been completely restructured to support a dimensional data warehouse (star schema) for sentiment analysis. The key improvement is the integration of **fuzzy logic with crisp score calculation** using the centroid method.

## Architecture Changes

### Data Model

The backend now works exclusively with data warehouse tables:

**Dimensions:**
- `dim_calendar` - Temporal dimension with date hierarchy
- `dim_review` - Review master data
- `dim_restaurant` - Restaurant master data

**Fact Table:**
- `fact_sentiment` - Sentiment analysis facts with fuzzy and crisp scores

### Request/Response Model

#### ReviewRequest
```json
{
  "restaurant_id": 1,
  "review_id": 1,
  "text": "The food was excellent but service was slow",
  "source": "api",
  "language": "en"
}
```

#### ReviewResponse
Returns detailed sentiment analysis for each extracted aspect:
```json
{
  "id": 1,
  "restaurant_id": 1,
  "text": "...",
  "created_at": "2026-05-18T18:44:41.741103",
  "aspects": [
    {
      "aspect_term": "food",
      "opinion_term": "excellent",
      "pos_score": 0.066,      # Fuzzy positive membership
      "neut_score": 0.290,     # Fuzzy neutral membership
      "neg_score": 0.644,      # Fuzzy negative membership
      "fuzzy_crisp_score": 0.211,  # Crisp score from centroid method
      "sentiment_polarity": "negative",
      "confidence": 0.211,
      "aspect_category": "FOOD_QUALITY"
    }
  ]
}
```

## Fuzzy Logic & Crisp Score Calculation

### Process Flow

1. **Aspect Extraction** - Extract aspects from review text using NER model
2. **Sentiment Classification** - Classify sentiment for each aspect (produces fuzzy scores)
3. **Fuzzy Membership Calculation** - Map classifier scores to fuzzy membership values
4. **Defuzzification (Centroid Method)** - Convert fuzzy values to crisp score

### Centroid Method Formula

```
crisp_score = (0.0 * neg_score + 0.5 * neu_score + 1.0 * pos_score) / (neg_score + neu_score + pos_score)

Where:
- 0.0 = negative sentiment
- 0.5 = neutral sentiment  
- 1.0 = positive sentiment
```

### Example Calculation

For a review aspect with fuzzy scores:
- Negative: 0.644
- Neutral: 0.290
- Positive: 0.066

```
crisp_score = (0.0 * 0.644 + 0.5 * 0.290 + 1.0 * 0.066) / (0.644 + 0.290 + 0.066)
           = (0 + 0.145 + 0.066) / 1.0
           = 0.211
```

This crisp score of **0.211** indicates a negative sentiment (closer to 0 than 1).

## API Endpoints

### 1. Analyze Review
**POST** `/analyze`

Analyzes a restaurant review and saves results to the fact table.

Request body:
```json
{
  "restaurant_id": 1,
  "review_id": 1,
  "text": "Review text here",
  "source": "api",
  "language": "en"
}
```

Response: `ReviewResponse` with detected aspects and sentiment scores.

**Note:** 
- `restaurant_id` and `review_id` must exist in their respective dimensions
- Use ETL loader to populate dimensions before analyzing reviews

### 2. Sentiment by Category
**GET** `/analytics/sentiment-by-category`

Returns sentiment distribution grouped by aspect category with average crisp scores.

### 3. Top Negative Aspects
**GET** `/analytics/top-negative-aspects?limit=10`

Returns most frequently complained-about aspects with count and average crisp scores.

### 4. Sentiment Over Time
**GET** `/analytics/sentiment-over-time`

Returns sentiment trends grouped by date with average crisp scores.

### 5. Health Check
**GET** `/health`

Returns server status and loaded models.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application with endpoints |
| `schemas.py` | Pydantic request/response models |
| `database/repository.py` | DW queries and ETL functions |
| `database/models.py` | SQLAlchemy ORM models |
| `models/fuzzy_sentiment.py` | Fuzzy logic analyzer with centroid method |
| `etl_loader.py` | ETL utilities for dimension loading |

## ETL & Dimension Loading

Before analyzing reviews, dimensions must be populated:

### Using ETL Loader

```python
from database.connection import SessionLocal
from etl_loader import populate_calendar_dimension, load_restaurants

db = SessionLocal()

# Populate calendar
populate_calendar_dimension(db, date(2024, 1, 1), date(2024, 12, 31))

# Load restaurants
restaurants = [
    {
        "id_restaurant": 1,
        "name": "The Gourmet Place",
        "district": "Downtown",
        "category": "fine dining",
        "address": "123 Main Street",
        "inspection_grade": "A"
    }
]
load_restaurants(db, restaurants)
```

### From Docker Container

```bash
docker exec -i backend bash -c 'cd /app && python3 << "SCRIPT"
from datetime import date
from database.connection import SessionLocal
from etl_loader import populate_calendar_dimension, load_restaurants

db = SessionLocal()
populate_calendar_dimension(db, date(2024, 1, 1), date(2024, 12, 31))
# ... load restaurants
db.close()
SCRIPT
'
```

## Database Schema

### dim_calendar
- `date_id` (PK): Date as YYYY-MM-DD
- `date`: Date value
- `week`: Week start date (hierarchy)
- `month`: Month start date (hierarchy)
- `year`: Year value (hierarchy)

### dim_review
- `id_review` (PK): Source system review ID
- `source`: Data source (e.g., "api", "google")
- `text`: Review text
- `language`: Language code
- `created_at`: Review creation timestamp

### dim_restaurant
- `id_restaurant` (PK): Source system restaurant ID
- `name`: Restaurant name
- `district`: Geographic district
- `category`: Type (e.g., "fine dining", "fast food")
- `address`: Street address
- `inspection_grade`: Health inspection grade

### fact_sentiment
- `fact_id` (PK): Auto-increment surrogate key
- `id_review` (FK): Reference to dim_review
- `id_restaurant` (FK): Reference to dim_restaurant
- `date_id` (FK): Reference to dim_calendar
- `aspect_term`: Extracted aspect (e.g., "food", "service")
- `opinion_term`: Opinion word (e.g., "excellent", "slow")
- `aspect_category`: Classified category (e.g., "FOOD_QUALITY")
- `fuzzy_crisp_score`: Crisp value from centroid method [0.0, 1.0]
- `sentiment_polarity`: Linguistic label ("positive", "neutral", "negative")
- `confidence_score`: Same as fuzzy_crisp_score
- `created_at`: Timestamp of analysis

## Fuzzy Logic Implementation Details

### Membership Functions

Three triangular membership functions are defined:

```
Negative: triangle(0.0, 0.0, 0.5)  # Peak at 0 (very negative)
Neutral:  triangle(0.2, 0.5, 0.8)  # Peak at 0.5 (neutral)
Positive: triangle(0.5, 1.0, 1.0)  # Peak at 1 (very positive)
```

### Aggregation

When multiple aspects are detected in a single review, their fuzzy scores are averaged before defuzzification to get a review-level crisp score.

## Testing

### Test with Sample Data

1. Load dimensions:
```bash
docker exec -i backend bash -c 'cd /app && python3 << "SCRIPT"
from datetime import date
from database.connection import SessionLocal
from etl_loader import populate_calendar_dimension, load_restaurants

db = SessionLocal()
populate_calendar_dimension(db, date(2024, 1, 1), date(2024, 12, 31))
sample_restaurants = [
    {"id_restaurant": 1, "name": "Test Restaurant", "district": "Downtown", 
     "category": "fine dining", "address": "123 Main", "inspection_grade": "A"}
]
load_restaurants(db, sample_restaurants)
db.close()
SCRIPT
'
```

2. Create a review dimension record:
```bash
PGPASSWORD=password psql -h localhost -U user -d dw -c \
  "INSERT INTO dim_review (id_review, source, text, language, created_at) 
   VALUES (1, 'api', 'Test review', 'en', NOW())"
```

3. Analyze a review:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant_id": 1,
    "review_id": 1,
    "text": "The food was great but service was slow",
    "source": "api",
    "language": "en"
  }'
```

4. Query results:
```bash
PGPASSWORD=password psql -h localhost -U user -d dw -c \
  "SELECT aspect_term, sentiment_polarity, fuzzy_crisp_score FROM fact_sentiment LIMIT 10"
```

## Performance Notes

- Calendar dimension is pre-populated for faster lookups
- Restaurant dimension should be loaded once during initialization
- Review dimension created on-demand during analysis
- Fact table records created per aspect detected (1 row per aspect-opinion pair)

## Important Notes

1. **Dimensions must be populated before analysis** - The `/analyze` endpoint expects restaurants to exist in dim_restaurant

2. **Review IDs from source system** - `review_id` in the request should match the source system ID so records can be joined with dim_review

3. **No operational tables** - The backend works exclusively with DW tables. Operational data should be transformed via ETL before ingestion

4. **Fuzzy crisp score interpretation**:
   - 0.0 = Strong negative sentiment
   - 0.5 = Neutral sentiment
   - 1.0 = Strong positive sentiment

## Future Enhancements

- Implement automated ETL from source systems
- Add dimension slowly-changing dimensions (SCD) Type 2 for restaurant updates
- Add more sophisticated aggregation functions
- Implement time-series forecasting
- Add confidence intervals for crisp scores
