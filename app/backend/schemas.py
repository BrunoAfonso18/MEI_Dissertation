from pydantic import BaseModel

class ReviewRequest(BaseModel):
    text: str

class AspectResult(BaseModel):
    aspect_term:        str
    opinion_term:       str | None
    sentiment_polarity: str
    aspect_category:    str

class ReviewResponse(BaseModel):
    sentence: str
    aspects:  list[AspectResult]