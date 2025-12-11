import asyncio
import os
from fastapi import APIRouter
import pandas as pd
from db import get_db
from stats import list_items, list_weather

router = APIRouter()

# Shop refresh times
shop_refresh_minutes = {
    "seeds": 5,
    "gear": 5,
    "eventshop": 30,
    "eggs": 30,
    "cosmetics": 240
}


def prepare_features(df: pd.DataFrame):
    df = df.sort_values("timestamp")
    df["delta_minutes"] = df["timestamp"].diff().dt.total_seconds() / 60
    df["prev_delta"] = df["delta_minutes"].shift(1)
    df = df.dropna(subset=["delta_minutes", "prev_delta"])
    return df


def load_item_data(item_name: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
                       SELECT item, shop, timestamp
                       FROM item_snapshot
                       WHERE item = %s
                       ORDER BY timestamp
                       """, (item_name,))
        rows = cursor.fetchall()
        cursor.close()

    df = pd.DataFrame(rows, columns=["item", "shop", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_weather_data(weather_name: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT weather, timestamp
            FROM weather_snapshot
            WHERE weather = %s
            ORDER BY timestamp
        """, (weather_name,))
        rows = cursor.fetchall()
        cursor.close()

    df = pd.DataFrame(rows, columns=["weather", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


async def get_item_csv_for_shop(shop_name: str):
    items_list = await list_items()
    all_data = []

    for item in items_list:
        df = load_item_data(item["name"])
        df_prepared = prepare_features(df)
        # Only include rows for this shop
        df_shop = df_prepared[df_prepared["shop"] == shop_name]
        if not df_shop.empty:
            all_data.append(df_shop)

    if not all_data:
        return None  # No data for this shop

    dataset = pd.concat(all_data, ignore_index=True)

    # Make sure the folder exists
    os.makedirs("datasets", exist_ok=True)

    filename = f"datasets/{shop_name}_item_deltas.csv"
    dataset.to_csv(filename, index=False)
    print(f"{filename} saved for SageMaker!")
    return filename


@router.get("/csv/seeds")
async def get_seeds_csv():
    return await get_item_csv_for_shop("seeds")

@router.get("/csv/gear")
async def get_gear_csv():
    return await get_item_csv_for_shop("gear")

@router.get("/csv/eventshop")
async def get_eventshop_csv():
    return await get_item_csv_for_shop("eventshop")

@router.get("/csv/eggs")
async def get_eggs_csv():
    return await get_item_csv_for_shop("eggs")

@router.get("/csv/cosmetics")
async def get_cosmetics_csv():
    return await get_item_csv_for_shop("cosmetics")

@router.get("/csv/items")
async def get_items_csv():
    results = await asyncio.gather(
        get_seeds_csv(),
        get_gear_csv(),
        get_eventshop_csv(),
        get_eggs_csv(),
        get_cosmetics_csv()
    )
    return results

@router.get("/csv/weather")
async def get_weather_csv():
    weathers_list = await list_weather()
    all_data = []

    for weather in weathers_list:
        df = load_weather_data(weather)
        df_prepared = prepare_features(df)
        all_data.append(df_prepared)

    if not all_data:
        return {"message": "No weather data available."}

    dataset = pd.concat(all_data, ignore_index=True)

    # Make sure the folder exists
    os.makedirs("datasets", exist_ok=True)

    filename = "datasets/weather_deltas.csv"
    dataset.to_csv(filename, index=False)
    print(f"{filename} saved for SageMaker!")
    return {"filename": filename}
