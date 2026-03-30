"""
FastAPI application for Obsidian optimization campaigns.

This module provides a REST API that exposes the orchestration layer via HTTP endpoints.
"""

from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import logging

from obsidian.api.routers import sessions, info
from obsidian.orchestration import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app (disable default docs, we'll create custom ones)
app = FastAPI(
    title="Obsidian Optimization API",
    description="REST API for Bayesian optimization campaigns using Obsidian",
    version="0.1.0",
    docs_url=None,  # Disable default
    redoc_url=None,  # Disable default
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Add CORS middleware (for web UIs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize SessionManager on startup
@app.on_event("startup")
async def startup_event():
    """Initialize SessionManager singleton on startup."""
    logger.info("Initializing SessionManager...")
    SessionManager.get_instance()
    logger.info("SessionManager initialized successfully")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "error_type": "internal_server_error"},
    )


# KeyError handler (for missing sessions)
@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    """Handle KeyError (typically missing session ID)."""
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc), "error_type": "not_found"})


# ValueError handler (for invalid input)
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError (typically invalid configuration)."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc), "error_type": "bad_request"}
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Obsidian API is running"}


# Favicon endpoint with Obsidian logo
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve Obsidian logo as favicon."""
    favicon_path = Path(__file__).parent / "static" / "obsidian_logo.svg"
    return FileResponse(favicon_path)


# Custom Swagger UI with Obsidian logo
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with Obsidian branding."""
    html = get_swagger_ui_html(
        openapi_url=str(app.openapi_url),
        title=f"{app.title} - Swagger UI",
        swagger_favicon_url="/static/obsidian_logo.svg",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "docExpansion": "list",
        },
    )

    # Inject CSS to replace Swagger logo
    custom_css = """
    <style>
        .topbar-wrapper img[alt="Swagger UI"] {
            content: url('/static/obsidian_logo.svg');
            height: 35px;
            width: auto;
        }
    </style>
    """

    # Insert CSS before closing head tag
    html_content = html.body.decode('utf-8')
    html_content = html_content.replace('</head>', f'{custom_css}</head>')

    return HTMLResponse(content=html_content)


# Custom ReDoc with Obsidian logo
@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    """Custom ReDoc with Obsidian branding."""
    return get_redoc_html(
        openapi_url=str(app.openapi_url),
        title=f"{app.title} - ReDoc",
        redoc_favicon_url="/static/obsidian_logo.svg",
    )


# Include routers
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
app.include_router(info.router, prefix="/api/v1", tags=["Information"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
