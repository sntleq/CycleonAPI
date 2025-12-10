from typing import List
from db import get_db
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter
import pandas as pd
import asyncio

router = APIRouter()

shop_refresh_minutes = {
    "seeds": 5,
    "gear": 5,
    "eventshop": 30,
    "eggs": 30,
    "cosmetics": 240
}


@router.get("/items")
async def list_items():
    shop_names = {
        "seeds": "Seeds",
        "gear": "Gear",
        "eggs": "Eggs",
        "eventshop": "Event Items",
        "cosmetics": "Cosmetics"
    }

    with get_db() as conn:
        cursor = conn.cursor()

        # 1. Get list of all distinct items
        cursor.execute("""
            SELECT DISTINCT item
            FROM item_snapshot
            ORDER BY item
        """)
        items = [row["item"] for row in cursor.fetchall()]

        results = []

        # 2. For each item, get all shops it appears in
        for item in items:
            cursor.execute("""
                SELECT DISTINCT shop
                FROM item_snapshot
                WHERE item = %s
                ORDER BY shop
            """, (item,))
            shops = [shop_names.get(row["shop"], row["shop"]) for row in cursor.fetchall()]

            results.append({
                "name": item,
                "shops": shops
            })

        cursor.close()

    # 3. Sort the items list by first shop name
    results.sort(key=lambda x: x["shops"][0] if x["shops"] else "~")

    return results


@router.get("/weather")
async def list_weather():
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT weather
            FROM weather_snapshot
            ORDER BY weather
        """)

        weathers = [row["weather"] for row in cursor.fetchall()]

        cursor.close()

    return weathers


@router.get("/item-stats")
async def list_item_stats():
    items_list = await list_items()
    results = []

    coroutines = [item_stats(item["name"]) for item in items_list]
    stats_results = await asyncio.gather(*coroutines)
    results.extend(stats_results)

    return results


@router.get("/item-stats/{item}")
async def item_stats(item: str):
    """Return stats for a specific item: last seen, shops, frequency, appearances for the last 7 days."""
    with get_db() as conn:
        cursor = conn.cursor()

        # LAST SEEN (most recent timestamp)
        cursor.execute("""
            SELECT timestamp
            FROM item_snapshot
            WHERE item = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (item,))
        last_seen_row = cursor.fetchone()
        last_seen = last_seen_row["timestamp"] if last_seen_row else None

        # SHOPS and FREQUENCY
        cursor.execute("""
            SELECT shop, COUNT(*) AS item_count
            FROM item_snapshot
            WHERE item = %s
            GROUP BY shop
            ORDER BY item_count DESC
            """, (item,))
        shops_rows = cursor.fetchall()

        # List of shop names only (for output)
        shops = [row["shop"] for row in shops_rows]

        # Top shop for frequency calculation
        if shops_rows:
            top_shop_row = shops_rows[0]  # shop with most entries
            top_shop = top_shop_row["shop"]
            item_count = top_shop_row["item_count"]

            # Total entries in the top shop (all items)
            cursor.execute("""
                           SELECT COUNT(DISTINCT timestamp) AS total_count
                           FROM item_snapshot
                           WHERE shop = %s
                           """, (top_shop,))
            total_count = cursor.fetchone()["total_count"]

            # Frequency in % (4 decimals)
            frequency = round(item_count / total_count * 100, 4) if total_count > 0 else 0

            # Get the refresh interval for the top shop
            refresh_min = shop_refresh_minutes.get(top_shop)  # default 1 hour if unknown

            # Calculate "items per total_count * refresh interval"
            # Total time spanned by shop snapshots
            total_time_min = total_count * refresh_min  # in minutes

            # Items per total time
            if item_count > 0:
                # average interval in minutes between each item occurrence
                interval_per_item = total_time_min / item_count  # minutes
            else:
                interval_per_item = None

            # Format string nicely
            if interval_per_item is None:
                freq_string = "Has not appeared"
            elif interval_per_item < 60:
                freq_string = f"Appears every {round(interval_per_item)} mins"
            elif interval_per_item < 1440:  # less than a day
                hours = interval_per_item / 60
                hours_str = int(hours) if hours.is_integer() else round(hours, 2)
                freq_string = f"Appears every {hours_str} hrs"
            else:
                days = interval_per_item / 1440
                days_str = int(days) if days.is_integer() else round(days, 2)
                freq_string = f"Appears every {days_str} days"

        else:
            frequency = 0
            frequency = "Has not appeared"
            shops = []

        # Philippine timezone
        ph_tz = timezone(timedelta(hours=8))

        # Get today in PH time
        today = datetime.now(ph_tz).date()
        seven_days_ago = today - timedelta(days=6)

        cursor.execute("""
                       SELECT DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Manila') AS day, SUM(quantity) AS qty
                       FROM item_snapshot
                       WHERE item = %s
                         AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Manila') >= %s
                       GROUP BY day
                       ORDER BY day
                       """, (item, seven_days_ago))

        appearances_rows = cursor.fetchall()

        # APPEARANCES (fill missing days with 0)
        appearances_dict = {row["day"]: row["qty"] for row in appearances_rows}
        appearances: List[int] = [
            appearances_dict.get(seven_days_ago + timedelta(days=i), 0) for i in range(7)
        ]

        cursor.close()

    return {
        "item": item,
        "last_seen": last_seen,
        "shops": shops,
        "frequency": frequency,
        "frequency_string": freq_string,
        "appearances": appearances
    }


@router.get("/weather-stats")
async def list_weather_stats():
    # 1. Get list of weather names from your existing function
    weather_names = await list_weather()
    results = []

    coroutines = [weather_stats(name) for name in weather_names]
    stats_results = await asyncio.gather(*coroutines)
    results.extend(stats_results)

    return results


@router.get("/weather-stats/{weather}")
async def weather_stats(weather: str):
    """Return stats for a specific weather: last seen, count, frequency, appearances for the last 7 days."""
    weather_lower = weather.lower()  # Handle case sensitivity

    with get_db() as conn:
        cursor = conn.cursor()

        # LAST SEEN (most recent timestamp)
        cursor.execute("""
                       SELECT timestamp
                       FROM weather_snapshot
                       WHERE weather = %s
                       ORDER BY timestamp DESC
                       LIMIT 1
                       """, (weather_lower,))
        last_seen_row = cursor.fetchone()
        last_seen = last_seen_row["timestamp"] if last_seen_row else None

        # COUNT occurrences of this weather
        cursor.execute("""
                       SELECT COUNT(*) AS weather_count
                       FROM weather_snapshot
                       WHERE weather = %s
                       """, (weather_lower,))
        weather_count = cursor.fetchone()["weather_count"]

        # EARLIEST timestamp of ANY weather in the database
        cursor.execute("""
                       SELECT MIN(timestamp) AS earliest
                       FROM weather_snapshot
                       """)
        earliest_row = cursor.fetchone()
        earliest = earliest_row["earliest"] if earliest_row else None

        # FREQUENCY CALCULATION
        if weather_count > 0 and earliest is not None:
            # Make earliest timezone-aware (assuming it's stored as UTC)
            if earliest.tzinfo is None:
                earliest = earliest.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            total_minutes = (now - earliest).total_seconds() / 60

            # Average interval in minutes between each weather occurrence
            interval_per_occurrence = total_minutes / weather_count

            # Format string nicely
            if interval_per_occurrence < 60:
                freq_string = f"Appears every {round(interval_per_occurrence)} mins"
            elif interval_per_occurrence < 1440:  # less than a day
                hours = interval_per_occurrence / 60
                hours_str = int(hours) if hours.is_integer() else round(hours, 2)
                freq_string = f"Appears every {hours_str} hrs"
            else:
                days = interval_per_occurrence / 1440
                days_str = int(days) if days.is_integer() else round(days, 2)
                freq_string = f"Appears every {days_str} days"
        else:
            freq_string = "Has not appeared"

        # Philippine timezone
        ph_tz = timezone(timedelta(hours=8))

        # Get today in PH time
        today = datetime.now(ph_tz).date()
        seven_days_ago = today - timedelta(days=6)

        # APPEARANCES (count occurrences per day for last 7 days)
        cursor.execute("""
                       SELECT DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Manila') AS day, COUNT(*) AS count
                       FROM weather_snapshot
                       WHERE weather = %s
                         AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Manila') >= %s
                       GROUP BY day
                       ORDER BY day
                       """, (weather_lower, seven_days_ago))

        appearances_rows = cursor.fetchall()

        # Fill missing days with 0
        appearances_dict = {row["day"]: row["count"] for row in appearances_rows}
        appearances: List[int] = [
            appearances_dict.get(seven_days_ago + timedelta(days=i), 0) for i in range(7)
        ]

        cursor.close()

    return {
        "weather": weather,
        "last_seen": last_seen,
        "count": weather_count,
        "frequency_string": freq_string,
        "appearances": appearances
    }


@router.get("/p-vals")
async def p_vals():
    with get_db() as conn:
        cursor = conn.cursor()

        # 1. Load all snapshot data
        cursor.execute("""
                       SELECT item, shop, timestamp
                       FROM item_snapshot
                       ORDER BY timestamp
                       """)
        rows = cursor.fetchall()
        cursor.close()

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(rows, columns=["item", "shop", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    results = []

    # 2. Process per shop
    for shop, shop_df in df.groupby("shop"):
        refresh = shop_refresh_minutes.get(str(shop))
        timestamps = sorted(shop_df["timestamp"].unique())
        if len(timestamps) < 2:
            continue

        # build full timeline based on shop refresh
        full_range = pd.date_range(
            start=min(timestamps),
            end=max(timestamps),
            freq=f"{refresh}min"
        )

        # 3. Process per item in this shop
        for item, item_df in shop_df.groupby("item"):
            ts = pd.Series(0, index=full_range)
            ts[item_df["timestamp"].values] = 1

            prev = ts.shift(1)

            # Drop the first NaN to avoid artifacts
            mask_valid = prev.notna()
            ts_valid = ts[mask_valid]
            prev_valid = prev[mask_valid]

            p_yes = ts_valid[prev_valid == 1].mean()
            p_no = ts_valid[prev_valid == 0].mean()

            results.append({
                "shop": shop,
                "item": item,
                "refresh_minutes": refresh,
                "p_next_if_appeared": round(float(p_yes), 4) if p_yes == p_yes else None,
                "p_next_if_not": round(float(p_no), 4) if p_no == p_no else None,
                "difference": None if (p_yes != p_yes or p_no != p_no) else round(float(p_yes - p_no), 4)
            })

    # 4. Sort by shop then item for easier viewing
    results.sort(key=lambda x: (x["shop"], x["item"]))

    return {"dependence": results}