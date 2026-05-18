from sqlalchemy import (
    Column, Integer, String, Float,
    DateTime, ForeignKey, Text, Date
)
from sqlalchemy.sql import func
from database.connection import Base


# ============================================
# DATA WAREHOUSE DIMENSION TABLES
# ============================================

class DimCalendar(Base):
    """Calendar dimension with date hierarchy"""
    __tablename__ = "dim_calendar"

    date_id = Column(Date, primary_key=True, index=True)
    date    = Column(Date, nullable=False, unique=True)
    week    = Column(Date, nullable=False)
    month   = Column(Date, nullable=False)
    year    = Column(Integer, nullable=False)


class DimReview(Base):
    """Review dimension for the data warehouse"""
    __tablename__ = "dim_review"

    id_review   = Column(Integer, primary_key=True, index=True)
    source      = Column(String(100), nullable=True)
    text        = Column(Text, nullable=True)
    language    = Column(String(10), nullable=True)
    created_at  = Column(DateTime, nullable=True)


class DimRestaurant(Base):
    """Restaurant dimension for the data warehouse"""
    __tablename__ = "dim_restaurant"

    id_restaurant    = Column(Integer, primary_key=True, index=True)
    district         = Column(String(100), nullable=True)
    category         = Column(String(100), nullable=True)
    address          = Column(Text, nullable=True)
    name             = Column(String(255), nullable=True)
    inspection_grade = Column(String(10), nullable=True)


class FactSentiment(Base):
    """Fact table for sentiment analysis"""
    __tablename__ = "fact_sentiment"

    fact_id            = Column(Integer, primary_key=True, index=True)
    id_review          = Column(Integer, ForeignKey("dim_review.id_review"), nullable=False)
    id_restaurant      = Column(Integer, ForeignKey("dim_restaurant.id_restaurant"), nullable=False)
    date_id            = Column(Date, ForeignKey("dim_calendar.date_id"), nullable=False)
    aspect_term        = Column(String(200), nullable=True)
    opinion_term       = Column(String(200), nullable=True)
    aspect_category    = Column(String(100), nullable=True)
    fuzzy_crisp_score  = Column(Float, nullable=True)  # Crisp value from centroid method
    sentiment_polarity = Column(String(20), nullable=True)
    confidence_score   = Column(Float, nullable=True)
    created_at         = Column(DateTime, nullable=True)