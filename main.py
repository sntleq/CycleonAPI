import os
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

from tasks import (
    update_seeds, update_gear, update_cosmetics,
    update_eggs, update_eventshop, update_weather
)
from db import init_db

# Detect Railway build phase
IS_BUILD = os.getenv("RAILWAY_BUILD", "0") == "1"

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not IS_BUILD:
        # ---------------------- #
        # Start-up (Deploy only)
        # ---------------------- #
        init_db()

        # Schedule cron jobs
        scheduler.add_job(update_seeds, 'cron', minute='*/5')
        scheduler.add_job(update_gear, 'cron', minute='*/5')
        scheduler.add_job(update_eggs, 'cron', minute='*/30')
        scheduler.add_job(update_eventshop, 'cron', minute='*/30')
        scheduler.add_job(update_cosmetics, 'cron', hour='*/4', minute='0')
        scheduler.add_job(update_weather, 'cron', minute='*/5')

        scheduler.start()

        # Run once at startup
        update_seeds()
        update_gear()
        update_cosmetics()
        update_eggs()
        update_eventshop()
        update_weather()

    yield

    if not IS_BUILD:
        # ---------------------- #
        # Shutdown (Deploy only)
        # ---------------------- #
        scheduler.shutdown()


app = FastAPI(lifespan=lifespan)
