"""
Generate predictions CSV from trained model - SHOP-SPECIFIC VERSION
This is just a helper function, NOT a standalone script
"""

import pandas as pd
from inference_script import ItemPredictor, WeatherPredictor


def generate_shop_predictions_csv(
    shop_name,
    model_s3_path,
    items_csv_path,
    cycle_minutes,
    output_csv=None
):
    """
    Generate predictions for ALL items in a specific shop and save to CSV

    Args:
        shop_name: Name of the shop (e.g., 'seeds', 'eggs')
        model_s3_path: S3 path to trained model
        items_csv_path: Path to items CSV (e.g., 'datasets/seeds_item_deltas.csv')
        cycle_minutes: Shop cycle time in minutes
        output_csv: Output CSV filename (defaults to {shop_name}_predictions.csv)

    Returns:
        Path to generated CSV file
    """
    if output_csv is None:
        output_csv = f"predictions/{shop_name}_predictions.csv"

    print(f"Generating predictions for {shop_name} shop...")

    # Load data
    df = pd.read_csv(items_csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Data should already be filtered by shop, but double-check
    shop_df = df[df['shop'] == shop_name]

    if len(shop_df) == 0:
        raise ValueError(f"No data found for shop: {shop_name}")

    # Initialize predictor
    predictor = ItemPredictor(model_s3_path)

    # Get all unique items in this shop
    unique_items = shop_df['item'].unique()

    all_predictions = []

    print(f"  Processing {len(unique_items)} items...")

    for item_name in unique_items:
        # 1. Next 5 timestamps
        next_occurrences = predictor.predict_next_occurrences(
            item_name, shop_name, df, n_predictions=5
        )

        if next_occurrences:
            for pred in next_occurrences:
                all_predictions.append({
                    'shop': shop_name,
                    'item': item_name,
                    'prediction_type': 'next_occurrence',
                    'occurrence_number': pred['occurrence'],
                    'predicted_timestamp': pred['predicted_time'],
                    'predicted_delta_minutes': pred['predicted_delta_minutes'],
                    'confidence': pred['confidence'],
                    'cycle_number': None,
                    'probability': None
                })

        # 2. Cycle probabilities (5 cycles)
        cycle_probs = predictor.predict_cycle_probabilities(
            item_name, shop_name, df, cycle_minutes=cycle_minutes, n_cycles=5
        )

        if cycle_probs:
            for prob in cycle_probs:
                all_predictions.append({
                    'shop': shop_name,
                    'item': item_name,
                    'prediction_type': 'cycle_probability',
                    'occurrence_number': None,
                    'predicted_timestamp': None,
                    'predicted_delta_minutes': None,
                    'confidence': None,
                    'cycle_number': prob['cycle'],
                    'probability': prob['probability']
                })

        # 3. Confidence windows (80%, 85%, 90%)
        for conf_level in [0.80, 0.85, 0.90]:
            window = predictor.predict_confidence_window(
                item_name, shop_name, df, confidence=conf_level, cycle_minutes=cycle_minutes
            )

            if window:
                all_predictions.append({
                    'shop': shop_name,
                    'item': item_name,
                    'prediction_type': 'confidence_window',
                    'occurrence_number': None,
                    'predicted_timestamp': None,
                    'predicted_delta_minutes': None,
                    'confidence': window['confidence_level'],
                    'cycle_number': window['cycles'],
                    'probability': None
                })

    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Save to CSV
    predictions_df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(predictions_df)} predictions to {output_csv}")

    return output_csv


def generate_weather_predictions_csv(weather_csv_path, output_csv='predictions/weather_predictions.csv'):
    """
    Generate predictions for ALL weather types and save to CSV

    Args:
        weather_csv_path: Path to weather.csv (e.g., 'datasets/weather_deltas.csv')
        output_csv: Output CSV filename

    Returns:
        Path to generated CSV file
    """
    print("Generating weather predictions...")

    # Load data
    df = pd.read_csv(weather_csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Initialize predictor
    predictor = WeatherPredictor(df)

    # Get all unique weather types
    unique_weather = df['weather'].unique()

    all_predictions = []

    print(f"  Processing {len(unique_weather)} weather types...")

    for weather_type in unique_weather:
        # 1. Next 5 timestamps
        next_occurrences = predictor.predict_next_occurrences(
            weather_type, n_predictions=5
        )

        if next_occurrences:
            for pred in next_occurrences:
                all_predictions.append({
                    'weather': weather_type,
                    'prediction_type': 'next_occurrence',
                    'occurrence_number': pred['occurrence'],
                    'predicted_timestamp': pred['predicted_time'],
                    'predicted_delta_minutes': pred['predicted_delta_minutes'],
                    'confidence': pred['confidence'],
                    'time_window_minutes': None,
                    'probability': None
                })

        # 2. Time window probabilities (5, 10, 15, 20, 25 min)
        time_probs = predictor.predict_time_window_probabilities(
            weather_type, time_windows=[5, 10, 15, 20, 25]
        )

        if time_probs:
            for prob in time_probs:
                all_predictions.append({
                    'weather': weather_type,
                    'prediction_type': 'time_window_probability',
                    'occurrence_number': None,
                    'predicted_timestamp': prob['predicted_time'],
                    'predicted_delta_minutes': None,
                    'confidence': None,
                    'time_window_minutes': prob['minutes'],
                    'probability': prob['probability']
                })

        # 3. Confidence windows (80%, 85%, 90%)
        conf_windows = predictor.predict_confidence_windows(
            weather_type, confidence_levels=[0.80, 0.85, 0.90]
        )

        if conf_windows:
            for window in conf_windows:
                all_predictions.append({
                    'weather': weather_type,
                    'prediction_type': 'confidence_window',
                    'occurrence_number': None,
                    'predicted_timestamp': window['predicted_time'],
                    'predicted_delta_minutes': None,
                    'confidence': window['confidence_level'],
                    'time_window_minutes': window['minutes'],
                    'probability': None
                })

    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Save to CSV
    predictions_df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(predictions_df)} predictions to {output_csv}")

    return output_csv