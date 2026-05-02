from pydantic import BaseModel

class ReviewRequest(BaseModel):
    text: str

class AspectResult(BaseModel):
    aspect_term:        str
    opinion_term:       str | None
    positive_score:     float
    neutral_score:      float
    negative_score:     float
    sentiment_polarity: str
    confidence:         float
    aspect_category:    str

class ReviewResponse(BaseModel):
    sentence: str
    aspects:  list[AspectResult]