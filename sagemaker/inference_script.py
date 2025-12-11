"""
Make predictions using trained LSTM model
Run this in your Railway app to get predictions
"""

import boto3
import torch
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta


class ItemPredictor:
    def __init__(self, model_s3_path, device='cpu'):
        """
        Load trained model from S3

        Args:
            model_s3_path: S3 path to model artifacts (e.g., 's3://bucket/models/model.tar.gz')
        """
        self.device = torch.device(device)
        self.download_model(model_s3_path)
        self.load_model()

    def download_model(self, s3_path):
        """Download model artifacts from S3"""
        s3 = boto3.client('s3')

        # Parse S3 path
        bucket = s3_path.split('/')[2]
        key = '/'.join(s3_path.split('/')[3:])

        # Download and extract
        import tarfile
        s3.download_file(bucket, key, 'model.tar.gz')

        with tarfile.open('model.tar.gz', 'r:gz') as tar:
            tar.extractall()

    def load_model(self):
        """Load model, scaler, and config"""
        from training_script import ItemLSTM  # Import your model class

        # Load config
        self.config = joblib.load('config.pkl')
        self.scaler = joblib.load('scaler.pkl')

        # Load model
        self.model = ItemLSTM(
            input_size=1,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers']
        )
        self.model.load_state_dict(torch.load('lstm_model.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict_next_occurrences(self, item_name, shop_name, df, n_predictions=5):
        """
        Predict next N occurrences for an item

        Args:
            item_name: Name of the item
            shop_name: Name of the shop
            df: DataFrame with historical data
            n_predictions: Number of future occurrences to predict

        Returns:
            List of predictions with timestamps and confidence
        """
        # Get item history
        item_data = df[(df['item'] == item_name) & (df['shop'] == shop_name)].sort_values('timestamp')

        if len(item_data) < self.config['seq_length']:
            return None

        # Get last sequence
        last_deltas = item_data['delta_minutes'].tail(self.config['seq_length']).values
        last_timestamp = item_data['timestamp'].iloc[-1]

        predictions = []
        current_sequence = last_deltas.copy()
        current_time = last_timestamp

        with torch.no_grad():
            for i in range(n_predictions):
                # Prepare input
                seq_scaled = self.scaler.transform(current_sequence.reshape(1, -1))
                seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(-1).to(self.device)

                # Predict next delta
                predicted_delta = self.model(seq_tensor).item()

                # Calculate next timestamp
                next_time = current_time + timedelta(minutes=predicted_delta)

                # Confidence decreases with distance
                confidence = 1.0 / (1 + i * 0.15)

                predictions.append({
                    'occurrence': i + 1,
                    'predicted_time': next_time,
                    'predicted_delta_minutes': predicted_delta,
                    'confidence': confidence
                })

                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = predicted_delta
                current_time = next_time

        return predictions

    def predict_cycle_probabilities(self, item_name, shop_name, df, cycle_minutes=30, n_cycles=5):
        """
        Predict probability of item appearing in next N shop cycles

        Args:
            cycle_minutes: Minutes per cycle (shop refresh rate)
            n_cycles: Number of cycles to predict

        Returns:
            List of probabilities for each cycle
        """
        # Get predicted next occurrence
        next_occurrences = self.predict_next_occurrences(item_name, shop_name, df, n_predictions=1)

        if not next_occurrences:
            return None

        predicted_delta = next_occurrences[0]['predicted_delta_minutes']

        probabilities = []
        for i in range(1, n_cycles + 1):
            cycle_end = cycle_minutes * i

            # Probability that predicted_delta <= cycle_end
            # Using exponential decay based on predicted delta
            prob = 1 - np.exp(-cycle_end / predicted_delta)

            probabilities.append({
                'cycle': i,
                'minutes_from_now': cycle_end,
                'probability': min(prob, 1.0),
                'cumulative': True
            })

        return probabilities

    def predict_confidence_window(self, item_name, shop_name, df, confidence=0.85, cycle_minutes=30):
        """
        Predict within how many cycles the item will appear with given confidence

        Args:
            confidence: Confidence level (0.80 = 80%, 0.90 = 90%, etc.)
            cycle_minutes: Minutes per cycle

        Returns:
            Number of cycles and minutes for given confidence level
        """
        # Get item history to estimate variance
        item_data = df[(df['item'] == item_name) & (df['shop'] == shop_name)].sort_values('timestamp')

        if len(item_data) < 3:
            return None

        # Get predicted delta and historical std deviation
        next_pred = self.predict_next_occurrences(item_name, shop_name, df, n_predictions=1)
        if not next_pred:
            return None

        predicted_delta = next_pred[0]['predicted_delta_minutes']
        historical_std = item_data['delta_minutes'].std()

        # Calculate confidence interval using predicted delta ± z*std
        # For exponential-like distributions, use log-normal approximation
        mean_delta = item_data['delta_minutes'].mean()
        cv = historical_std / mean_delta  # Coefficient of variation

        # Confidence multiplier (z-score approximations)
        z_scores = {0.80: 1.28, 0.85: 1.44, 0.90: 1.645, 0.95: 1.96}
        z = z_scores.get(confidence, 1.645)

        # Upper bound for confidence interval
        upper_bound = predicted_delta * (1 + z * cv)

        # Convert to cycles
        cycles_needed = int(np.ceil(upper_bound / cycle_minutes))
        minutes_needed = cycles_needed * cycle_minutes

        return {
            'confidence_level': confidence,
            'cycles': cycles_needed,
            'minutes': minutes_needed,
            'message': f"Item will appear within {cycles_needed} cycles ({minutes_needed} min) with {confidence * 100:.0f}% confidence"
        }


# Usage example
def main():
    # Load your data
    df = pd.read_csv('items.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Initialize predictor
    predictor = ItemPredictor(
        model_s3_path='s3://your-bucket/models/pytorch-training-2024-12-11-12-00-00-000/output/model.tar.gz'
    )

    # Predict next 5 occurrences for Bug Egg
    predictions = predictor.predict_next_occurrences('Bug Egg', 'eggs', df, n_predictions=5)

    print("Predicted next occurrences:")
    for pred in predictions:
        print(f"  {pred['occurrence']}. {pred['predicted_time']} "
              f"(in {pred['predicted_delta_minutes']:.1f} min, confidence: {pred['confidence']:.2%})")

    # Predict cycle probabilities
    cycle_probs = predictor.predict_cycle_probabilities('Bug Egg', 'eggs', df, cycle_minutes=30, n_cycles=5)

    print("\nProbability in next cycles:")
    for prob in cycle_probs:
        print(f"  Cycle {prob['cycle']} ({prob['minutes_from_now']} min): {prob['probability']:.2%}")

    # NEW: Get confidence windows for 80%, 85%, 90%
    confidence_windows = predictor.predict_confidence_window('Bug Egg', 'eggs', df, confidence=0.80, cycle_minutes=30)

    print("\nConfidence Windows (80%, 85%, 90%):")
    for conf_level in [0.80, 0.85, 0.90]:
        window = predictor.predict_confidence_window('Bug Egg', 'eggs', df, confidence=conf_level, cycle_minutes=30)
        if window:
            print(f"  {window['message']}")


class WeatherPredictor:
    """Predictor for random weather events"""

    def __init__(self, df):
        self.df = df
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def predict_next_occurrences(self, weather_type, n_predictions=5):
        """
        Predict next N timestamps for weather events

        Args:
            weather_type: Type of weather event
            n_predictions: Number of future occurrences to predict

        Returns:
            List of predicted timestamps
        """
        weather_data = self.df[self.df['weather'] == weather_type].sort_values('timestamp')

        if len(weather_data) < 2:
            return None

        # For exponential distribution, mean is the best predictor
        mean_delta = weather_data['delta_minutes'].mean()
        last_timestamp = weather_data['timestamp'].iloc[-1]

        predictions = []
        current_time = last_timestamp

        for i in range(1, n_predictions + 1):
            # For random events, each prediction adds mean_delta
            next_time = current_time + timedelta(minutes=mean_delta)

            # Confidence decreases with each prediction
            confidence = 1.0 / (1 + i * 0.2)

            predictions.append({
                'occurrence': i,
                'predicted_time': next_time,
                'predicted_delta_minutes': mean_delta,
                'confidence': confidence
            })

            current_time = next_time

        return predictions

    def predict_confidence_windows(self, weather_type, confidence_levels=[0.80, 0.85, 0.90]):
        """
        Predict within how many minutes weather will appear for multiple confidence levels

        Args:
            weather_type: Type of weather event
            confidence_levels: List of confidence levels to calculate

        Returns:
            List of confidence windows
        """
        weather_data = self.df[self.df['weather'] == weather_type]

        if len(weather_data) < 3:
            return None

        # For random events, use exponential distribution
        mean_delta = weather_data['delta_minutes'].mean()
        rate = 1 / mean_delta
        last_timestamp = weather_data['timestamp'].iloc[-1]

        windows = []
        for confidence in confidence_levels:
            # For exponential: t = -ln(1 - confidence) / λ
            minutes_needed = -np.log(1 - confidence) / rate
            predicted_time = last_timestamp + timedelta(minutes=minutes_needed)

            windows.append({
                'confidence_level': confidence,
                'minutes': int(np.ceil(minutes_needed)),
                'predicted_time': predicted_time,
                'message': f"{weather_type} will appear within {int(np.ceil(minutes_needed))} minutes with {confidence * 100:.0f}% confidence"
            })

        return windows

    def predict_time_window_probabilities(self, weather_type, time_windows=[5, 10, 15, 20, 25]):
        """
        Predict probability of weather appearing in specific time windows

        Returns:
            List of probabilities for each time window (5 probabilities for 5 windows)
        """
        weather_data = self.df[self.df['weather'] == weather_type]

        if len(weather_data) < 2:
            return None

        # Exponential distribution
        mean_delta = weather_data['delta_minutes'].mean()
        rate = 1 / mean_delta
        last_timestamp = weather_data['timestamp'].iloc[-1]

        probabilities = []
        for window in time_windows:
            # P(event within window) = 1 - e^(-λt)
            prob = 1 - np.exp(-rate * window)
            predicted_time = last_timestamp + timedelta(minutes=window)

            probabilities.append({
                'minutes': window,
                'predicted_time': predicted_time,
                'probability': prob,
                'message': f"{prob * 100:.1f}% chance within {window} minutes (by {predicted_time.strftime('%H:%M:%S')})"
            })

        return probabilities


# Weather usage example
def weather_example():
    df = pd.read_csv('weather.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    predictor = WeatherPredictor(df)

    # 1. Predict next 5 timestamps
    next_weather = predictor.predict_next_occurrences('heatwave', n_predictions=5)

    print("Weather: Predicted next 5 occurrences:")
    for pred in next_weather:
        print(f"  {pred['occurrence']}. {pred['predicted_time']} "
              f"(in {pred['predicted_delta_minutes']:.1f} min, confidence: {pred['confidence']:.2%})")

    # 2. Get confidence windows for 80%, 85%, 90%
    confidence_windows = predictor.predict_confidence_windows('heatwave', confidence_levels=[0.80, 0.85, 0.90])

    print("\nWeather Confidence Windows (80%, 85%, 90%):")
    for window in confidence_windows:
        print(f"  {window['message']} (at {window['predicted_time'].strftime('%Y-%m-%d %H:%M:%S')})")

    # 3. Get probabilities for 5 specific time windows
    probs = predictor.predict_time_window_probabilities('heatwave', time_windows=[5, 10, 15, 20, 25])

    print("\nProbabilities for next 5 time windows:")
    for prob in probs:
        print(f"  {prob['message']}")


if __name__ == '__main__':
    main()
    print("\n" + "=" * 50 + "\n")
    weather_example()