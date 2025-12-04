import os
from fastapi import FastAPI
# from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

from tasks import (
    update_seeds, update_gear, update_cosmetics,
    update_eggs, update_eventshop, update_weather
)
from db import init_db

# Detect Railway build phase
IS_BUILD = os.getenv("RAILWAY_BUILD", "0") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not IS_BUILD:

        # Run once at startup
        update_seeds()
        update_gear()
        update_cosmetics()
        update_eggs()
        update_eventshop()
        update_weather()

    yield

    # if not IS_BUILD:
        # ---------------------- #
        # Shutdown (Deploy only)
        # ---------------------- #
        # scheduler.shutdown()


app = FastAPI(lifespan=lifespan)