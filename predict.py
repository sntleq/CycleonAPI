"""
Prediction router for Cycleon predictions
Add to main.py with: app.include_router(predict.router)
"""

from fastapi import APIRouter, HTTPException, Query
from prediction_service import PredictionService
from config import SHOP_CYCLE_MINUTES

router = APIRouter(prefix="/predict", tags=["predictions"])

# Initialize prediction service (lazy load)
_prediction_service = None

def get_prediction_service():
    """Lazy load prediction service"""
    global _prediction_service
    if _prediction_service is None:
        print("Initializing prediction service...")
        _prediction_service = PredictionService()
    return _prediction_service


@router.get("/items/{item_name}")
def get_item_predictions(
    item_name: str,
    shop: str = Query(None, description="Shop name (e.g., 'seeds', 'eggs'). Auto-detected if not provided."),
    cycle_minutes: int = Query(None, description="Shop cycle time in minutes (optional)")
):
    """
    Get predictions for a specific item

    Returns:
    - next_occurrences: 1 predicted timestamp
    - cycle_probabilities: 5 cycle probabilities (as percentage 0-100)
    - confidence_windows: 3 confidence levels (80%, 85%, 90%) or empty if insufficient data
    """
    service = get_prediction_service()

    # Auto-detect shop if not provided
    if shop is None:
        all_items = service.list_available_items()
        matching = [i for i in all_items if i['item'] == item_name]
        if not matching:
            raise HTTPException(status_code=404, detail=f"Item '{item_name}' not found in any shop")
        shop = matching[0]['shop']
        print(f"Auto-detected shop: {shop} for item: {item_name}")

    # Get cycle time from config or use provided value
    if cycle_minutes is None:
        cycle_minutes = SHOP_CYCLE_MINUTES.get(shop.lower(), SHOP_CYCLE_MINUTES['default'])

    try:
        predictions = service.get_item_predictions(
            item_name=item_name,
            shop_name=shop,
            cycle_minutes=cycle_minutes
        )

        if not predictions['next_occurrences'] and not predictions['cycle_probabilities']:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions available for item '{item_name}' in shop '{shop}'"
            )

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/weather/{weather_type}")
def get_weather_predictions(weather_type: str):
    """
    Get predictions for a specific weather type

    Returns:
    - next_occurrences: 1 predicted timestamp
    - time_window_probabilities: 5 time windows (5, 10, 15, 20, 25 min) with probabilities as percentage 0-100
    - confidence_windows: 3 confidence levels (80%, 85%, 90%)
    """
    service = get_prediction_service()

    try:
        predictions = service.get_weather_predictions(weather_type)

        if not predictions['next_occurrences'] and not predictions['time_window_probabilities']:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions available for weather type '{weather_type}'"
            )

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/items")
def list_items():
    """List all available items and their shops"""
    service = get_prediction_service()

    try:
        items = service.list_available_items()
        return {
            "count": len(items),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing items: {str(e)}")


@router.get("/weather")
def list_weather():
    """List all available weather types"""
    service = get_prediction_service()

    try:
        weather_types = service.list_available_weather()
        return {
            "count": len(weather_types),
            "weather_types": weather_types
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing weather: {str(e)}")


@router.get("/health")
def prediction_health():
    """Health check for prediction service"""
    try:
        service = get_prediction_service()
        return {
            "status": "healthy",
            "prediction_mode": service.mode
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }