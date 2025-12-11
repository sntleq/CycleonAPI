"""
Unified prediction service that switches between SageMaker and Statistical models
"""

import pandas as pd
import os
from typing import Optional, List, Dict
from config import PREDICTION_MODE, DATASETS_PATH, PREDICTIONS_PATH
from statistical_predictor import StatisticalItemPredictor, StatisticalWeatherPredictor


class PredictionService:
    """Main service that routes to correct prediction method"""

    def __init__(self):
        self.mode = PREDICTION_MODE

        if self.mode == 'statistical':
            print(f"✓ Prediction Service initialized in STATISTICAL mode")
            self.item_predictor = StatisticalItemPredictor(DATASETS_PATH)
            self.weather_predictor = StatisticalWeatherPredictor(DATASETS_PATH)

        elif self.mode == 'sagemaker':
            print(f"✓ Prediction Service initialized in SAGEMAKER mode")
            # Load pre-generated CSVs from SageMaker
            self.items_predictions = {}
            self.weather_predictions = None

            # Load shop predictions
            for filename in os.listdir(PREDICTIONS_PATH):
                if filename.endswith('_predictions.csv') and filename != 'weather_predictions.csv':
                    shop_name = filename.replace('_predictions.csv', '')
                    filepath = os.path.join(PREDICTIONS_PATH, filename)
                    self.items_predictions[shop_name] = pd.read_csv(filepath)
                    print(f"  ✓ Loaded {shop_name} predictions")

            # Load weather predictions
            weather_path = os.path.join(PREDICTIONS_PATH, 'weather_predictions.csv')
            if os.path.exists(weather_path):
                self.weather_predictions = pd.read_csv(weather_path)
                print(f"  ✓ Loaded weather predictions")

        else:
            raise ValueError(f"Invalid PREDICTION_MODE: {self.mode}")

    def get_item_predictions(
        self,
        item_name: str,
        shop_name: str,
        cycle_minutes: int
    ) -> Dict:
        """Get all predictions for an item"""
        if self.mode == 'statistical':
            return self._get_item_predictions_statistical(item_name, shop_name, cycle_minutes)
        else:
            return self._get_item_predictions_sagemaker(item_name, shop_name)

    def _get_item_predictions_statistical(
        self,
        item_name: str,
        shop_name: str,
        cycle_minutes: int
    ) -> Dict:
        """Get predictions using statistical models"""

        next_occ = self.item_predictor.predict_next_occurrences(
            item_name, shop_name, n_predictions=1  # Only 1 occurrence
        )

        cycle_probs = self.item_predictor.predict_cycle_probabilities(
            item_name, shop_name, cycle_minutes, n_cycles=5
        )

        conf_windows = self.item_predictor.predict_confidence_windows(
            item_name, shop_name, cycle_minutes, confidence_levels=[0.80, 0.85, 0.90]
        )

        return {
            'item': item_name,
            'shop': shop_name,
            'prediction_mode': 'statistical',
            'next_occurrences': next_occ or [],
            'cycle_probabilities': cycle_probs or [],
            'confidence_windows': conf_windows or []  # Will be [] if not enough data
        }

    def _get_item_predictions_sagemaker(
        self,
        item_name: str,
        shop_name: str
    ) -> Dict:
        """Get predictions from pre-generated SageMaker CSV"""

        if shop_name not in self.items_predictions:
            return {
                'item': item_name,
                'shop': shop_name,
                'prediction_mode': 'sagemaker',
                'next_occurrences': [],
                'cycle_probabilities': [],
                'confidence_windows': []
            }

        df = self.items_predictions[shop_name]
        df = df[df['item'] == item_name]

        if len(df) == 0:
            return {
                'item': item_name,
                'shop': shop_name,
                'prediction_mode': 'sagemaker',
                'next_occurrences': [],
                'cycle_probabilities': [],
                'confidence_windows': []
            }

        # Parse next occurrences
        next_occ_df = df[df['prediction_type'] == 'next_occurrence'].sort_values('occurrence_number')
        next_occurrences = []
        for _, row in next_occ_df.iterrows():
            next_occurrences.append({
                'occurrence': int(row['occurrence_number']),
                'predicted_time': row['predicted_timestamp'],
                'predicted_delta_minutes': float(row['predicted_delta_minutes']) if pd.notna(row['predicted_delta_minutes']) else None,
                'confidence': float(row['confidence']) if pd.notna(row['confidence']) else None
            })

        # Parse cycle probabilities
        cycle_df = df[df['prediction_type'] == 'cycle_probability'].sort_values('cycle_number')
        cycle_probabilities = []
        for _, row in cycle_df.iterrows():
            cycle_probabilities.append({
                'cycle': int(row['cycle_number']),
                'probability': float(row['probability']) if pd.notna(row['probability']) else None
            })

        # Parse confidence windows
        conf_df = df[df['prediction_type'] == 'confidence_window'].sort_values('confidence')
        confidence_windows = []
        for _, row in conf_df.iterrows():
            confidence_windows.append({
                'confidence_level': float(row['confidence']) if pd.notna(row['confidence']) else None,
                'cycles': int(row['cycle_number']) if pd.notna(row['cycle_number']) else None
            })

        return {
            'item': item_name,
            'shop': shop_name,
            'prediction_mode': 'sagemaker',
            'next_occurrences': next_occurrences,
            'cycle_probabilities': cycle_probabilities,
            'confidence_windows': confidence_windows
        }

    def get_weather_predictions(self, weather_type: str) -> Dict:
        """Get all predictions for weather"""
        if self.mode == 'statistical':
            return self._get_weather_predictions_statistical(weather_type)
        else:
            return self._get_weather_predictions_sagemaker(weather_type)

    def _get_weather_predictions_statistical(self, weather_type: str) -> Dict:
        """Get weather predictions using statistical models"""

        next_occ = self.weather_predictor.predict_next_occurrences(
            weather_type, n_predictions=1  # Only 1 occurrence
        )

        time_probs = self.weather_predictor.predict_time_window_probabilities(
            weather_type, time_windows=[5, 10, 15, 20, 25]
        )

        conf_windows = self.weather_predictor.predict_confidence_windows(
            weather_type, confidence_levels=[0.80, 0.85, 0.90]
        )

        return {
            'weather': weather_type,
            'prediction_mode': 'statistical',
            'next_occurrences': next_occ or [],
            'time_window_probabilities': time_probs or [],
            'confidence_windows': conf_windows or []
        }

    def _get_weather_predictions_sagemaker(self, weather_type: str) -> Dict:
        """Get weather predictions from pre-generated SageMaker CSV"""

        if self.weather_predictions is None:
            return {
                'weather': weather_type,
                'prediction_mode': 'sagemaker',
                'next_occurrences': [],
                'time_window_probabilities': [],
                'confidence_windows': []
            }

        df = self.weather_predictions[self.weather_predictions['weather'] == weather_type]

        if len(df) == 0:
            return {
                'weather': weather_type,
                'prediction_mode': 'sagemaker',
                'next_occurrences': [],
                'time_window_probabilities': [],
                'confidence_windows': []
            }

        # Parse next occurrences
        next_occ_df = df[df['prediction_type'] == 'next_occurrence'].sort_values('occurrence_number')
        next_occurrences = []
        for _, row in next_occ_df.iterrows():
            next_occurrences.append({
                'occurrence': int(row['occurrence_number']),
                'predicted_time': row['predicted_timestamp'],
                'predicted_delta_minutes': float(row['predicted_delta_minutes']) if pd.notna(row['predicted_delta_minutes']) else None,
                'confidence': float(row['confidence']) if pd.notna(row['confidence']) else None
            })

        # Parse time window probabilities
        time_df = df[df['prediction_type'] == 'time_window_probability'].sort_values('time_window_minutes')
        time_probabilities = []
        for _, row in time_df.iterrows():
            time_probabilities.append({
                'minutes': int(row['time_window_minutes']),
                'predicted_time': row['predicted_timestamp'],
                'probability': float(row['probability']) if pd.notna(row['probability']) else None
            })

        # Parse confidence windows
        conf_df = df[df['prediction_type'] == 'confidence_window'].sort_values('confidence')
        confidence_windows = []
        for _, row in conf_df.iterrows():
            confidence_windows.append({
                'confidence_level': float(row['confidence']) if pd.notna(row['confidence']) else None,
                'minutes': int(row['time_window_minutes']) if pd.notna(row['time_window_minutes']) else None,
                'predicted_time': row['predicted_timestamp']
            })

        return {
            'weather': weather_type,
            'prediction_mode': 'sagemaker',
            'next_occurrences': next_occurrences,
            'time_window_probabilities': time_probabilities,
            'confidence_windows': confidence_windows
        }

    def list_available_items(self) -> List[Dict]:
        """List all items available for predictions"""
        if self.mode == 'statistical':
            return self.item_predictor.get_all_items()
        else:
            all_items = []
            for shop_name, df in self.items_predictions.items():
                items = df['item'].unique()
                for item in items:
                    all_items.append({'item': item, 'shop': shop_name})
            return all_items

    def list_available_weather(self) -> List[str]:
        """List all weather types available for predictions"""
        if self.mode == 'statistical':
            return self.weather_predictor.get_all_weather_types()
        else:
            if self.weather_predictions is not None:
                return self.weather_predictions['weather'].unique().tolist()
            return []