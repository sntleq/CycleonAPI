"""
Statistical prediction models (NO ML, NO SAGEMAKER)
Simple, fast, and actually works
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import os


class StatisticalItemPredictor:
    """Statistical model for item predictions using exponential smoothing"""

    def __init__(self, datasets_path: str = 'datasets'):
        """Load all shop item data"""
        self.datasets_path = datasets_path
        self.shop_data = {}

        # Load all shop CSV files
        for filename in os.listdir(datasets_path):
            if filename.endswith('_item_deltas.csv'):
                shop_name = filename.replace('_item_deltas.csv', '')
                filepath = os.path.join(datasets_path, filename)
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.shop_data[shop_name] = df
                print(f"  ✓ Loaded {shop_name} shop data: {len(df)} records")

    def predict_next_occurrences(
        self,
        item_name: str,
        shop_name: str,
        n_predictions: int = 1
    ) -> Optional[List[Dict]]:
        """Predict next N occurrence timestamps using weighted moving average"""

        if shop_name not in self.shop_data:
            return None

        df = self.shop_data[shop_name]
        item_data = df[df['item'] == item_name].sort_values('timestamp')

        if len(item_data) < 2:
            return None

        # Use exponentially weighted moving average (more weight on recent)
        recent_deltas = item_data['delta_minutes'].tail(10).values
        weights = np.exp(np.linspace(-1, 0, len(recent_deltas)))
        avg_delta = np.average(recent_deltas, weights=weights)

        last_timestamp = item_data['timestamp'].iloc[-1]
        predictions = []

        for i in range(1, n_predictions + 1):
            next_time = last_timestamp + timedelta(minutes=avg_delta * i)
            confidence = (1.0 / (1 + i * 0.15)) * 100  # Convert to percentage

            predictions.append({
                'occurrence': i,
                'predicted_time': next_time.isoformat(),
                'predicted_delta_minutes': round(avg_delta, 2),
                'confidence': round(confidence, 2)
            })

        return predictions

    def predict_cycle_probabilities(
        self,
        item_name: str,
        shop_name: str,
        cycle_minutes: int,
        n_cycles: int = 5
    ) -> Optional[List[Dict]]:
        """Predict probability item appears in next N cycles"""

        if shop_name not in self.shop_data:
            return None

        df = self.shop_data[shop_name]
        item_data = df[df['item'] == item_name]

        if len(item_data) < 2:
            return None

        # Calculate average appearance rate
        avg_delta = item_data['delta_minutes'].mean()
        appearances_per_cycle = cycle_minutes / avg_delta

        probabilities = []
        for i in range(1, n_cycles + 1):
            lambda_param = appearances_per_cycle * i
            prob = (1 - np.exp(-lambda_param)) * 100  # Convert to percentage

            probabilities.append({
                'cycle': i,
                'minutes_from_now': cycle_minutes * i,
                'probability': round(min(prob, 100.0), 2)
            })

        return probabilities

    def predict_confidence_windows(
        self,
        item_name: str,
        shop_name: str,
        cycle_minutes: int,
        confidence_levels: List[float] = [0.80, 0.85, 0.90]
    ) -> Optional[List[Dict]]:
        """Predict within how many cycles item will appear with given confidence"""

        if shop_name not in self.shop_data:
            return None

        df = self.shop_data[shop_name]
        item_data = df[df['item'] == item_name]

        if len(item_data) < 3:
            # Not enough data for confidence windows
            return []

        mean_delta = item_data['delta_minutes'].mean()
        std_delta = item_data['delta_minutes'].std()

        # Check for zero variance (all deltas the same)
        if std_delta == 0 or mean_delta == 0:
            return []

        cv = std_delta / mean_delta

        z_scores = {0.80: 1.28, 0.85: 1.44, 0.90: 1.645, 0.95: 1.96}

        windows = []
        for confidence in confidence_levels:
            z = z_scores.get(confidence, 1.645)
            upper_bound = mean_delta * (1 + z * cv)
            cycles_needed = int(np.ceil(upper_bound / cycle_minutes))

            windows.append({
                'confidence_level': round(confidence * 100, 2),  # Convert to percentage
                'cycles': cycles_needed,
                'minutes': cycles_needed * cycle_minutes
            })

        return windows

    def get_all_items(self) -> List[Dict]:
        """Get list of all items across all shops"""
        all_items = []
        for shop_name, df in self.shop_data.items():
            items = df['item'].unique()
            for item in items:
                all_items.append({'item': item, 'shop': shop_name})
        return all_items


class StatisticalWeatherPredictor:
    """Statistical model for weather predictions using exponential distribution"""

    def __init__(self, datasets_path: str = 'datasets'):
        """Load historical weather data"""
        filepath = os.path.join(datasets_path, 'weather_deltas.csv')
        self.df = pd.read_csv(filepath)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"  ✓ Loaded weather data: {len(self.df)} records")

    def predict_next_occurrences(
        self,
        weather_type: str,
        n_predictions: int = 1
    ) -> Optional[List[Dict]]:
        """Predict next N occurrence timestamps"""
        weather_data = self.df[self.df['weather'] == weather_type].sort_values('timestamp')

        if len(weather_data) < 2:
            return None

        mean_delta = weather_data['delta_minutes'].mean()
        last_timestamp = weather_data['timestamp'].iloc[-1]

        predictions = []
        current_time = last_timestamp

        for i in range(1, n_predictions + 1):
            next_time = current_time + timedelta(minutes=mean_delta)
            confidence = (1.0 / (1 + i * 0.2)) * 100  # Convert to percentage

            predictions.append({
                'occurrence': i,
                'predicted_time': next_time.isoformat(),
                'predicted_delta_minutes': round(mean_delta, 2),
                'confidence': round(confidence, 2)
            })

            current_time = next_time

        return predictions

    def predict_time_window_probabilities(
        self,
        weather_type: str,
        time_windows: List[int] = [5, 10, 15, 20, 25]
    ) -> Optional[List[Dict]]:
        """Predict probability weather appears in specific time windows"""
        weather_data = self.df[self.df['weather'] == weather_type]

        if len(weather_data) < 2:
            return None

        mean_delta = weather_data['delta_minutes'].mean()
        rate = 1 / mean_delta
        last_timestamp = weather_data['timestamp'].iloc[-1]

        probabilities = []
        for window in time_windows:
            prob = (1 - np.exp(-rate * window)) * 100  # Convert to percentage
            predicted_time = last_timestamp + timedelta(minutes=window)

            probabilities.append({
                'minutes': window,
                'predicted_time': predicted_time.isoformat(),
                'probability': round(prob, 2)
            })

        return probabilities

    def predict_confidence_windows(
        self,
        weather_type: str,
        confidence_levels: List[float] = [0.80, 0.85, 0.90]
    ) -> Optional[List[Dict]]:
        """Predict within how many minutes weather will appear"""
        weather_data = self.df[self.df['weather'] == weather_type]

        if len(weather_data) < 2:
            return None

        mean_delta = weather_data['delta_minutes'].mean()
        rate = 1 / mean_delta
        last_timestamp = weather_data['timestamp'].iloc[-1]

        windows = []
        for confidence in confidence_levels:
            minutes_needed = -np.log(1 - confidence) / rate
            predicted_time = last_timestamp + timedelta(minutes=minutes_needed)

            windows.append({
                'confidence_level': round(confidence * 100, 2),  # Convert to percentage
                'minutes': int(np.ceil(minutes_needed)),
                'predicted_time': predicted_time.isoformat()
            })

        return windows

    def get_all_weather_types(self) -> List[str]:
        """Get list of all weather types"""
        return self.df['weather'].unique().tolist()