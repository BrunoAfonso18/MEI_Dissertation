"""
ETL Helper Script - Load Dimensions into Data Warehouse

This script provides utilities to populate the data warehouse dimensions:
- dim_calendar: Generate calendar entries for a date range
- dim_review: Load review records from source
- dim_restaurant: Load restaurant master data

Usage:
    from database.connection import SessionLocal
    from etl_loader import populate_calendar_dimension, load_restaurants
    
    db = SessionLocal()
    populate_calendar_dimension(db, start_date, end_date)
    load_restaurants(db, restaurants_data)
"""

from datetime import datetime, timedelta, date
from sqlalchemy.orm import Session
from database.models import DimCalendar, DimReview, DimRestaurant


def populate_calendar_dimension(
    db: Session,
    start_date: date,
    end_date: date
) -> int:
    """
    Populate dim_calendar with date range.
    
    Args:
        db: Database session
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
    
    Returns:
        Number of calendar records created
    """
    current_date = start_date
    count = 0
    
    while current_date <= end_date:
        # Check if already exists
        existing = db.query(DimCalendar).filter(
            DimCalendar.date_id == current_date
        ).first()
        
        if not existing:
            # Calculate week start (Monday of the week)
            week_start = current_date - timedelta(days=current_date.weekday())
            
            # Month start
            month_start = current_date.replace(day=1)
            
            dim_calendar = DimCalendar(
                date_id=current_date,
                date=current_date,
                week=week_start,
                month=month_start,
                year=current_date.year
            )
            db.add(dim_calendar)
            count += 1
        
        current_date += timedelta(days=1)
    
    if count > 0:
        db.commit()
        print(f"Inserted {count} calendar records from {start_date} to {end_date}")
    
    return count


def load_reviews(
    db: Session,
    reviews_data: list[dict]
) -> int:
    """
    Load review dimension records.
    
    Args:
        db: Database session
        reviews_data: List of dicts with keys:
            - id_review (int): Source system review ID
            - text (str): Review text
            - source (str): Source system (e.g., "google", "api")
            - language (str): Language code
            - created_at (datetime): Review creation time
    
    Returns:
        Number of review records created
    """
    count = 0
    
    for review_data in reviews_data:
        review_id = review_data.get("id_review")
        
        # Check if already exists
        existing = db.query(DimReview).filter(
            DimReview.id_review == review_id
        ).first()
        
        if not existing:
            dim_review = DimReview(
                id_review=review_id,
                text=review_data.get("text"),
                source=review_data.get("source", "api"),
                language=review_data.get("language", "en"),
                created_at=review_data.get("created_at")
            )
            db.add(dim_review)
            count += 1
    
    if count > 0:
        db.commit()
        print(f"Inserted {count} review dimension records")
    
    return count


def load_restaurants(
    db: Session,
    restaurants_data: list[dict]
) -> int:
    """
    Load restaurant dimension records.
    
    Args:
        db: Database session
        restaurants_data: List of dicts with keys:
            - id_restaurant (int): Restaurant ID
            - name (str): Restaurant name
            - district (str): District
            - category (str): Category (e.g., "fine dining", "fast food")
            - address (str): Address
            - inspection_grade (str): Health inspection grade
    
    Returns:
        Number of restaurant records created
    """
    count = 0
    
    for restaurant_data in restaurants_data:
        restaurant_id = restaurant_data.get("id_restaurant")
        
        # Check if already exists
        existing = db.query(DimRestaurant).filter(
            DimRestaurant.id_restaurant == restaurant_id
        ).first()
        
        if not existing:
            dim_restaurant = DimRestaurant(
                id_restaurant=restaurant_id,
                name=restaurant_data.get("name"),
                district=restaurant_data.get("district"),
                category=restaurant_data.get("category"),
                address=restaurant_data.get("address"),
                inspection_grade=restaurant_data.get("inspection_grade")
            )
            db.add(dim_restaurant)
            count += 1
    
    if count > 0:
        db.commit()
        print(f"Inserted {count} restaurant dimension records")
    
    return count


def bulk_load_dimensions(
    db: Session,
    calendar_start: date = None,
    calendar_end: date = None,
    reviews: list[dict] = None,
    restaurants: list[dict] = None
) -> dict:
    """
    Bulk load all dimensions in one operation.
    
    Args:
        db: Database session
        calendar_start: Calendar dimension start date
        calendar_end: Calendar dimension end date
        reviews: Review dimension data
        restaurants: Restaurant dimension data
    
    Returns:
        Dict with counts of records loaded
    """
    results = {
        "calendar": 0,
        "reviews": 0,
        "restaurants": 0
    }
    
    if calendar_start and calendar_end:
        results["calendar"] = populate_calendar_dimension(db, calendar_start, calendar_end)
    
    if reviews:
        results["reviews"] = load_reviews(db, reviews)
    
    if restaurants:
        results["restaurants"] = load_restaurants(db, restaurants)
    
    return results


# Example usage
if __name__ == "__main__":
    from database.connection import SessionLocal
    
    db = SessionLocal()
    
    # Example: Populate calendar for 2024
    populate_calendar_dimension(
        db,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31)
    )
    
    # Example: Load some restaurants
    sample_restaurants = [
        {
            "id_restaurant": 1,
            "name": "Restaurant A",
            "district": "Downtown",
            "category": "fine dining",
            "address": "123 Main St",
            "inspection_grade": "A"
        },
        {
            "id_restaurant": 2,
            "name": "Restaurant B",
            "district": "Uptown",
            "category": "fast food",
            "address": "456 Oak Ave",
            "inspection_grade": "B"
        }
    ]
    
    load_restaurants(db, sample_restaurants)
    
    db.close()
