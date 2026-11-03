from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(
    title="Ticket Summarization DaaS",
    description="Production-ready Ticket Summarization Distillation-as-a-Service using Student LLM Baselines.",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin testing from client company frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serves the primary Distillation-as-a-Service product website & evaluation studio."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Ticket Summarization DaaS API is running."}

@app.get("/client", include_in_schema=False)
def serve_client_ui():
    """Serves the client company customer support testing website."""
    client_file = STATIC_DIR / "client.html"
    if client_file.exists():
        return FileResponse(client_file)
    return {"message": "Client testing portal is not found."}

app.include_router(api_router)
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
