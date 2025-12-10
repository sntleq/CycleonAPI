import asyncio
from prepare_data import get_seeds_csv
from tasks import save_csv

async def generate_and_save_csv():
    save_csv(await get_seeds_csv())

if __name__ == "__main__":
    asyncio.run(generate_and_save_csv())
    print("Generated and saved dataset CSV")