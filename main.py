from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

from fetch import *
from tasks import update_seeds, update_gear, update_cosmetics, update_eggs, update_eventshop, update_weather
from db import init_db

# Initialize scheduler
scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()

    # Schedule tasks with cron (adjust minutes as needed)
    scheduler.add_job(update_seeds, 'cron', minute='*/5')
    scheduler.add_job(update_gear, 'cron', minute='*/5')
    scheduler.add_job(update_eggs, 'cron', minute='*/30')
    scheduler.add_job(update_eventshop, 'cron', minute='*/30')
    scheduler.add_job(update_cosmetics, 'cron', hour='*/4', minute='0')
    scheduler.add_job(update_weather, 'cron', minute='*/5')

    scheduler.start()

    # Run all tasks once on startup
    update_seeds()
    update_gear()
    update_cosmetics()
    update_eggs()
    update_eventshop()
    update_weather()

    yield

    # Shutdown
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)