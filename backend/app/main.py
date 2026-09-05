from contextlib import asynccontextmanager

from fastapi import FastAPI

from .db.session import engine, init_db
from .agent.router import router as agent_router
from .ingest.router import router as ingest_router
from .search.router import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield


app = FastAPI(title="Voice Knowledge Agent", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(agent_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
