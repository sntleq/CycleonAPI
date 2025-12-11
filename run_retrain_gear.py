import asyncio
from datetime import datetime
from prepare_data import get_gear_csv
from tasks import save_csv, download_predictions
from sagemaker.sagemaker_training import train_shop_model, generate_shop_predictions

BUCKET_NAME = "amazon-sagemaker-248896561752-ap-southeast-2-c3asvvi6hbt3qa"
CYCLE_MINUTES = 5


async def main():
    print(f"\n{'='*60}")
    print(f"GEAR RETRAIN - {datetime.now()}")
    print(f"{'='*60}\n")

    csv_file = await get_gear_csv()
    save_csv(csv_file)

    estimator = train_shop_model(
        shop_name="gear",
        bucket_name=BUCKET_NAME
    )

    generate_shop_predictions(
        shop_name="gear",
        model_s3_path=estimator.model_data,
        bucket_name=BUCKET_NAME,
        cycle_minutes=CYCLE_MINUTES
    )

    download_predictions("gear")

    print(f"\n{'='*60}")
    print("GEAR COMPLETE! 🎉")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
