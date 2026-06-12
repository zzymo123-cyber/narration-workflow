import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes.project import router as project_router
from api.routes.tasks import router as tasks_router
from api.routes.assets import router as assets_router
from api.routes.settings import router as settings_router, get_api_key
from api import poller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_port() -> int:
    raw_port = os.environ.get("NARRATION_PORT") or os.environ.get("PORT") or "8003"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError("NARRATION_PORT/PORT must be an integer between 1 and 65535.") from exc
    if port < 1 or port > 65535:
        raise RuntimeError("NARRATION_PORT/PORT must be an integer between 1 and 65535.")
    return port


@asynccontextmanager
async def lifespan(app: FastAPI):
    await poller.start()
    yield
    await poller.stop()


app = FastAPI(lifespan=lifespan)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


@app.middleware("http")
async def no_cache_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


app.include_router(project_router, prefix="/api/project")
app.include_router(tasks_router, prefix="/api/tasks")
app.include_router(assets_router, prefix="/api")
app.include_router(settings_router, prefix="/api")

app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    })


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/llm/config")
async def llm_config():
    from api.llm import get_llm_config
    return get_llm_config()


@app.post("/api/llm/health")
async def llm_health():
    import asyncio
    from api.llm import test_llm_connection
    result = await asyncio.to_thread(test_llm_connection)
    return result


if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.environ.get("NARRATION_RELOAD") == "1"
    uvicorn.run("main:app", host="0.0.0.0", port=get_port(), reload=reload_enabled,
                timeout_graceful_shutdown=10)
