from sqlalchemy import (
    Column, Integer, String, Float,
    DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base


class Review(Base):
    """Raw review input"""
    __tablename__ = "reviews"

    id         = Column(Integer, primary_key=True, index=True)
    text       = Column(Text, nullable=False)
    source     = Column(String(100))          # e.g. "api", "upload", "scraper"
    language   = Column(String(10))           # e.g. "en", "pt"
    created_at = Column(DateTime, server_default=func.now())

    aspects = relationship("AspectPrediction", back_populates="review")


class AspectPrediction(Base):
    """One row per aspect detected in a review"""
    __tablename__ = "aspect_predictions"

    id                 = Column(Integer, primary_key=True, index=True)
    review_id          = Column(Integer, ForeignKey("reviews.id"), nullable=False)
    aspect_term        = Column(String(200))
    opinion_term       = Column(String(200), nullable=True)
    aspect_category    = Column(String(100))
    positive_score    = Column(Float, nullable=True)
    neutral_score     = Column(Float, nullable=True)
    negative_score   = Column(Float, nullable=True)
    sentiment_polarity = Column(String(20))
    confidence        = Column(Float, nullable=True)
    created_at        = Column(DateTime, server_default=func.now())

    review = relationship("Review", back_populates="aspects")