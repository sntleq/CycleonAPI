import asyncio
from datetime import datetime
from prepare_data import get_eventshop_csv
from tasks import save_csv, download_predictions
from sagemaker.sagemaker_training import train_shop_model, generate_shop_predictions

BUCKET_NAME = "amazon-sagemaker-248896561752-ap-southeast-2-c3asvvi6hbt3qa"
CYCLE_MINUTES = 30


async def main():
    print(f"\n{'='*60}")
    print(f"EVENTSHOP RETRAIN - {datetime.now()}")
    print(f"{'='*60}\n")

    csv_file = await get_eventshop_csv()
    save_csv(csv_file)

    estimator = train_shop_model(
        shop_name="eventshop",
        bucket_name=BUCKET_NAME
    )

    generate_shop_predictions(
        shop_name="eventshop",
        model_s3_path=estimator.model_data,
        bucket_name=BUCKET_NAME,
        cycle_minutes=CYCLE_MINUTES
    )

    download_predictions("eventshop")

    print(f"\n{'='*60}")
    print("EVENTSHOP COMPLETE! 🎉")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
