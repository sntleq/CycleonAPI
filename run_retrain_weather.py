"""
Weather retrain workflow
Run with cron: 0 */6 * * * python run_retrain_weather.py
"""

import asyncio
from datetime import datetime
from prepare_data import get_weather_csv
from tasks import save_csv, download_predictions
from sagemaker.sagemaker_training import train_weather_model, generate_weather_predictions

BUCKET_NAME = "amazon-sagemaker-248896561752-ap-southeast-2-c3asvvi6hbt3qa"


async def main():
    print(f"\n{'='*60}")
    print(f"WEATHER RETRAIN - {datetime.now()}")
    print(f"{'='*60}\n")

    csv_data = await get_weather_csv()
    csv_path = csv_data.get('filename')
    save_csv(csv_path)

    estimator = train_weather_model(
        bucket_name=BUCKET_NAME
    )

    predictions_s3_path = generate_weather_predictions(
        model_s3_path=estimator.model_data,
        bucket_name=BUCKET_NAME
    )

    download_predictions('weather')  # Download only weather

    print(f"\n{'='*60}")
    print("COMPLETE! 🎉")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())