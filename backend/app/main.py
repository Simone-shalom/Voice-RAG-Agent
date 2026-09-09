from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from .core.logging import configure_logging
from .core.rate_limit import limiter
from .core.allowed_origins import get_allowed_origins
from .db.session import engine, init_db
from .agent.router import router as agent_router
from .episodes.router import router as episodes_router
from .ingest.router import router as ingest_router
from .search.router import router as search_router
from .voice.router import router as voice_router

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield


app = FastAPI(title="Voice Knowledge Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

_allowed_origins = get_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(agent_router)
app.include_router(voice_router)
app.include_router(episodes_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
