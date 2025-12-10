from contextlib import asynccontextmanager
from fastapi import FastAPI
import stats, prepare_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Cycleon startup complete.")
    yield
    print("Cycleon is shutting down.")

app = FastAPI(lifespan=lifespan)
app.include_router(stats.router)
app.include_router(prepare_data.router)
