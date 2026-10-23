from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router as api_router, get_pipeline
from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Ticket Summarization DaaS service...")
    settings = get_settings()
    try:
        pipeline = get_pipeline()
        logger.info(f"Service ready with model '{pipeline.model_wrapper.model_name}'")
    except Exception as e:
        logger.error(f"Error during model startup: {str(e)}")
    yield
    logger.info("Shutting down Ticket Summarization DaaS service...")

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Ticket Summarization DaaS",
    description="Production-ready Ticket Summarization Data-as-a-Service using Student LLM Baselines.",
    version="0.1.0",
    lifespan=lifespan
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serves the interactive testing and evaluation UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Ticket Summarization DaaS API is running."}

app.include_router(api_router)
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
