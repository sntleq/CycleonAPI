import boto3
from dotenv import load_dotenv

from fetch import fetch_seeds, fetch_gear, fetch_cosmetics, fetch_eggs, fetch_eventshop, fetch_weather
from save import save_seeds, save_gear, save_cosmetics, save_eggs, save_eventshop, save_weather_snapshot
from db import get_db


def update_seeds():
    """Fetch and save seeds data"""
    with get_db() as conn:
        data = fetch_seeds()
        if data:
            save_seeds(conn, data)

def update_gear():
    """Fetch and save gear data"""
    with get_db() as conn:
        data = fetch_gear()
        if data:
            save_gear(conn, data)

def update_cosmetics():
    """Fetch and save cosmetics data"""
    with get_db() as conn:
        data = fetch_cosmetics()
        if data:
            save_cosmetics(conn, data)

def update_eggs():
    """Fetch and save eggs data"""
    with get_db() as conn:
        data = fetch_eggs()
        if data:
            save_eggs(conn, data)

def update_eventshop():
    """Fetch and save event shop data"""
    with get_db() as conn:
        data = fetch_eventshop()
        if data:
            save_eventshop(conn, data)

def update_weather():
    """Fetch and save weather data"""
    with get_db() as conn:
        data = fetch_weather()
        if data:
            save_weather_snapshot(conn, data)


load_dotenv()
BUCKET_NAME = "amazon-sagemaker-248896561752-ap-southeast-2-c3asvvi6hbt3qa"
s3 = boto3.client("s3")

def save_csv(csv_file):
    try:
        s3.upload_file(csv_file, BUCKET_NAME, csv_file)
        print(f"Uploaded {csv_file} to s3://{BUCKET_NAME}")
    except Exception as e:
        print("Failed to upload {csv_file}:", e)