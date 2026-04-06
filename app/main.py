import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # app/
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Suicidal Ideation Detection API",
    description=(
        "A tiered NLP system for detecting and classifying suicidal ideation "
        "in social media posts. Uses a keyword prefilter, a unified Claude "
        "analysis agent, escalation for ambiguous/high-risk cases, and a "
        "motivation layer."
    ),
    version="1.0.0",
    contact={
        "name": "Mental Health NLP Research",
        "url": "https://github.com/your-org/suicidal-ideation-detection",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS - allow all origins in development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files & templates
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.warning("Static directory not found at %s. /static will not be served.", STATIC_DIR)

templates: Jinja2Templates | None = None
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
else:
    logger.warning(
        "Templates directory not found at %s. The root route will return a plain response.",
        TEMPLATES_DIR,
    )

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(router, prefix="/api")

# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    primary_provider = os.environ.get("LLM_PRIMARY_PROVIDER", "openrouter")
    fallback_provider = os.environ.get("LLM_FALLBACK_PROVIDER", "openrouter")
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        logger.info("ANTHROPIC_API_KEY detected (%s). Anthropic override is available.", masked)
    else:
        logger.info("ANTHROPIC_API_KEY not set. This is fine when using OpenRouter-only mode.")

    if openrouter_key:
        masked_or = openrouter_key[:8] + "..." + openrouter_key[-4:] if len(openrouter_key) > 12 else "***"
        logger.info("OPENROUTER_API_KEY detected (%s). OpenRouter website mode is ready.", masked_or)
    else:
        logger.warning(
            "OPENROUTER_API_KEY not set. The website's default LLM path will not work. "
            "Set OPENROUTER_API_KEY or use use_ml_fallback=true."
        )

    logger.info("LLM routing: primary=%s fallback=%s", primary_provider, fallback_provider)

    logger.info("Suicidal Ideation Detection API v1.0.0 is starting up.")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Frontend"], include_in_schema=False)
async def root(request: Request) -> HTMLResponse:
    """Serve the main frontend page."""
    html_file = TEMPLATES_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file), media_type="text/html")

    # Fallback if templates directory is missing
    return HTMLResponse(
        content=(
            "<html><body>"
            "<h1>MindGuard | AI Mental Wellness Analyser</h1>"
            "<p>Templates directory not found. "
            "Visit <a href='/docs'>/docs</a> for the interactive API documentation.</p>"
            "</body></html>"
        ),
        status_code=200,
    )


@app.get("/api/info", tags=["Meta"])
async def api_info() -> dict:
    """Return project metadata and endpoint overview."""
    return {
        "project": "Suicidal Ideation Detection in Social Media",
        "version": "1.0.0",
        "description": (
            "Tiered NLP pipeline using normalization, rule-based signal analysis, "
            "and routed LLM calls to classify self-harm and harm-to-others risk, "
            "explain linguistic markers, recommend interventions, and generate "
            "support content."
        ),
        "llm_routing": {
            "primary_provider": os.environ.get("LLM_PRIMARY_PROVIDER", "openrouter"),
            "fallback_provider": os.environ.get("LLM_FALLBACK_PROVIDER", "openrouter"),
            "openrouter_fast_model": os.environ.get("OPENROUTER_FAST_MODEL", "openrouter/free"),
            "openrouter_smart_model": os.environ.get("OPENROUTER_SMART_MODEL", "openrouter/free"),
        },
        "agents": {
            "normalization_agent": {
                "description": "Repairs merged words, slang, and generation-specific phrasing before analysis.",
                "model": "local rules",
            },
            "keyword_prefilter": {
                "description": "Zero-token rules for obvious high/moderate/low-risk phrases.",
                "model": "local rules",
            },
            "unified_analysis": {
                "description": "Classifies risk, explains markers, and recommends interventions.",
                "model": "claude-haiku / claude-sonnet",
            },
            "motivation": {
                "description": "Adds supportive coping strategies and motivational guidance.",
                "model": "claude-haiku",
            },
        },
        "endpoints": {
            "POST /api/analyze": "Analyse a single post.",
            "POST /api/analyze/batch": "Analyse up to 10 posts.",
            "GET /api/health": "Health check.",
            "GET /api/stats": "In-memory usage statistics.",
            "POST /api/feedback": "Submit label corrections.",
            "GET /api/info": "This endpoint.",
            "GET /docs": "Interactive Swagger UI.",
            "GET /redoc": "ReDoc documentation.",
        },
        "crisis_resources": {
            "988_lifeline": "Call or text 988 (US)",
            "crisis_text_line": "Text HOME to 741741",
            "iasp": "https://www.iasp.info/resources/Crisis_Centres/",
        },
    }
