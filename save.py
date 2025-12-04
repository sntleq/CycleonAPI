def save_weather_snapshot(conn, weather_data):
    """
    Save weather snapshot to database

    Args:
        conn: Database connection
        weather_data: Weather data from API (dict with 'type', 'active', 'lastUpdated')

    Returns:
        int: ID of inserted row or None if failed
    """
    if not weather_data or not weather_data.get('active'):
        print("Weather is not active, skipping save")
        return None

    cursor = conn.cursor()

    try:
        weather_type = weather_data.get('type')
        duration = 0
        last_updated = weather_data.get('lastUpdated')

        cursor.execute("""
            INSERT INTO weather_snapshot (weather, duration, timestamp)
            VALUES (%s, %s, %s)
            ON CONFLICT (weather, timestamp) DO NOTHING
            RETURNING id
        """, (weather_type, duration, last_updated))

        result = cursor.fetchone()
        conn.commit()

        if result:
            print(f"Saved weather snapshot: {weather_type} at {last_updated}")
            return result[0]
        else:
            print(f"Weather snapshot already exists: {weather_type} at {last_updated}")
            return None

    except Exception as e:
        print(f"Error saving weather snapshot: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        return None


def save_seeds(conn, seeds_data):
    """
    Save seeds snapshot to database

    Args:
        conn: Database connection
        seeds_data: List of seed items from API

    Returns:
        int: Number of items saved
    """
    if not seeds_data:
        print("No seeds to save")
        return 0

    cursor = conn.cursor()
    saved_count = 0

    try:
        for item in seeds_data:
            item_name = item.get('name')
            quantity = item.get('quantity')

            if not item_name or quantity is None:
                continue

            cursor.execute("""
                INSERT INTO item_snapshot (item, shop, quantity, timestamp)
                VALUES (%s, %s, %s,
                       DATE_TRUNC('hour', CURRENT_TIMESTAMP) + 
                       INTERVAL '5 min' * FLOOR(EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) / 5))
                ON CONFLICT (item, shop, timestamp) DO NOTHING
            """, (item_name, 'seeds', quantity))

            if cursor.rowcount > 0:
                saved_count += 1

        conn.commit()
        print(f"Saved {saved_count} seeds")
        return saved_count

    except Exception as e:
        print(f"Error saving seeds: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        return None


def save_gear(conn, gear_data):
    """
    Save gear snapshot to database

    Args:
        conn: Database connection
        gear_data: List of gear items from API

    Returns:
        int: Number of items saved
    """
    if not gear_data:
        print("No gear to save")
        return 0

    cursor = conn.cursor()
    saved_count = 0

    try:
        for item in gear_data:
            item_name = item.get('name')
            quantity = item.get('quantity')

            if not item_name or quantity is None:
                continue

            cursor.execute("""
                INSERT INTO item_snapshot (item, shop, quantity, timestamp)
                VALUES (%s, %s, %s,
                       DATE_TRUNC('hour', CURRENT_TIMESTAMP) + 
                       INTERVAL '5 min' * FLOOR(EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) / 5))
                ON CONFLICT (item, shop, timestamp) DO NOTHING
            """, (item_name, 'gear', quantity))

            if cursor.rowcount > 0:
                saved_count += 1

        conn.commit()
        print(f"Saved {saved_count} gear items")
        return saved_count

    except Exception as e:
        print(f"Error saving gear: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        return None


def save_cosmetics(conn, cosmetics_data):
    """
    Save cosmetics snapshot to database

    Args:
        conn: Database connection
        cosmetics_data: List of cosmetic items from API

    Returns:
        int: Number of items saved
    """
    if not cosmetics_data:
        print("No cosmetics to save")
        return 0

    cursor = conn.cursor()
    saved_count = 0

    try:
        for item in cosmetics_data:
            item_name = item.get('name')
            quantity = item.get('quantity')

            if not item_name or quantity is None:
                continue

            cursor.execute("""
                INSERT INTO item_snapshot (item, shop, quantity, timestamp)
                VALUES (%s, %s, %s,
                       DATE_TRUNC('day', CURRENT_TIMESTAMP) + 
                       INTERVAL '4 hour' * FLOOR(EXTRACT(HOUR FROM CURRENT_TIMESTAMP) / 4))
                ON CONFLICT (item, shop, timestamp) DO NOTHING
            """, (item_name, 'cosmetics', quantity))

            if cursor.rowcount > 0:
                saved_count += 1

        conn.commit()
        print(f"Saved {saved_count} cosmetics")
        return saved_count

    except Exception as e:
        print(f"Error saving cosmetics: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        return None


def save_eggs(conn, eggs_data):
    """
    Save eggs snapshot to database

    Args:
        conn: Database connection
        eggs_data: List of egg items from API

    Returns:
        int: Number of items saved
    """
    if not eggs_data:
        print("No eggs to save")
        return 0

    cursor = conn.cursor()
    saved_count = 0

    try:
        for item in eggs_data:
            item_name = item.get('name')
            quantity = item.get('quantity')

            if not item_name or quantity is None:
                continue

            cursor.execute("""
                INSERT INTO item_snapshot (item, shop, quantity, timestamp)
                VALUES (%s, %s, %s,
                       DATE_TRUNC('hour', CURRENT_TIMESTAMP) + 
                       INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) / 30))
                ON CONFLICT (item, shop, timestamp) DO NOTHING
            """, (item_name, 'eggs', quantity))

            if cursor.rowcount > 0:
                saved_count += 1

        conn.commit()
        print(f"Saved {saved_count} eggs")
        return saved_count

    except Exception as e:
        print(f"Error saving eggs: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        return None


def save_eventshop(conn, eventshop_data):
    """
    Save event shop snapshot to database

    Args:
        conn: Database connection
        eventshop_data: List of event shop items from API

    Returns:
        int: Number of items saved
    """
    if not eventshop_data:
        print("No event shop items to save")
        return 0

    cursor = conn.cursor()
    saved_count = 0

    try:
        for item in eventshop_data:
            item_name = item.get('name')
            quantity = item.get('quantity')

            if not item_name or quantity is None:
                continue

            cursor.execute("""
                INSERT INTO item_snapshot (item, shop, quantity, timestamp)
                VALUES (%s, %s, %s,
                       DATE_TRUNC('hour', CURRENT_TIMESTAMP) + 
                       INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) / 30))
                ON CONFLICT (item, shop, timestamp) DO NOTHING
            """, (item_name, 'eventshop', quantity))

            if cursor.rowcount > 0:
                saved_count += 1

        conn.commit()
        print(f"Saved {saved_count} event shop items")
        return saved_count

    except Exception as e:
        print(f"Error saving event shop: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
        return None