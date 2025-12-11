"""
Configuration flags for prediction system
"""

# Prediction mode: 'sagemaker' or 'statistical'
PREDICTION_MODE = 'statistical'  # Set to 'statistical' for local, 'sagemaker' for AWS

# Shop cycle times (in minutes)
SHOP_CYCLE_MINUTES = {
    'seeds': 5,
    'eggs': 30,
    'default': 30
}

# API settings
API_TITLE = "Cycleon Predictions API"
API_VERSION = "1.0.0"

# Datasets base path
DATASETS_PATH = 'datasets'
PREDICTIONS_PATH = 'predictions'