import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

# Load .env file from project root or current directory
# .env.local takes precedence over .env
env_paths = [
    Path(__file__).parent.parent / ".env.local",  # fish-speech/.env.local (local overrides)
    Path(__file__).parent.parent / ".env",  # fish-speech/.env
    Path.cwd() / ".env.local",  # current directory
    Path.cwd() / ".env",  # current directory
    Path("/workspace/fish-speech/.env.local"),  # RunPod
    Path("/workspace/fish-speech/.env"),  # RunPod default
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

import pyrootutils
import uvicorn
from pydantic import ValidationError
from kui.asgi import (
    Depends,
    FactoryClass,
    HTTPException,
    HttpRoute,
    Kui,
    OpenAPI,
    Routes,
    request,
)
from kui.cors import CORSConfig
from kui.openapi.specification import Info
from kui.security import bearer_auth
from loguru import logger
from typing_extensions import Annotated

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.server.api_utils import MsgPackRequest, parse_args
from tools.server.exception_handler import ExceptionHandler
from tools.server.model_manager import ModelManager
from tools.server.views import routes

# Configure structured logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "text"

# Remove default logger and configure based on format
logger.remove()
if LOG_FORMAT == "json":
    logger.add(
        sys.stderr,
        format="{message}",
        level=LOG_LEVEL,
        serialize=True,  # JSON output
    )
else:
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        colorize=True,
    )

# Initialize Sentry if DSN is provided
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[
                StarletteIntegration(transaction_style="endpoint"),
                LoggingIntegration(level=None, event_level="ERROR"),
            ],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            send_default_pii=False,
        )
        logger.info("Sentry initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Sentry: {e}")


# Rate limiting configuration (per hour)
RATE_LIMIT_PER_KEY = int(os.getenv("RATE_LIMIT_PER_KEY", "2000"))  # requests per hour
RATE_LIMIT_PER_IP = int(os.getenv("RATE_LIMIT_PER_IP", "200"))  # requests per hour for unauthenticated
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Check if request is allowed for given key."""
        with self.lock:
            now = time.time()
            # Remove old requests outside window
            self.requests[key] = [t for t in self.requests[key] if now - t < window_seconds]

            if len(self.requests[key]) >= limit:
                return False

            self.requests[key].append(now)
            return True

    def get_remaining(self, key: str, limit: int, window_seconds: int = 60) -> int:
        """Get remaining requests for key."""
        with self.lock:
            now = time.time()
            self.requests[key] = [t for t in self.requests[key] if now - t < window_seconds]
            return max(0, limit - len(self.requests[key]))


rate_limiter = RateLimiter()


class API(ExceptionHandler):
    def __init__(self):
        self.args = parse_args()

        def api_auth(endpoint):
            async def verify(token: Annotated[str, Depends(bearer_auth)]):
                if token != self.args.api_key:
                    raise HTTPException(401, None, "Invalid token")

                # Rate limiting for authenticated requests (per hour)
                if RATE_LIMIT_ENABLED:
                    key = f"api_key:{token}"
                    if not rate_limiter.is_allowed(key, RATE_LIMIT_PER_KEY, RATE_LIMIT_WINDOW):
                        from tools.server.metrics import RATE_LIMIT_HITS
                        RATE_LIMIT_HITS.labels(key_type="api_key").inc()
                        raise HTTPException(429, None, "Rate limit exceeded")

                return await endpoint()

            async def passthrough():
                # Rate limiting for unauthenticated requests (by IP, per hour)
                if RATE_LIMIT_ENABLED:
                    # Get client IP from request
                    client_ip = "unknown"
                    try:
                        if hasattr(request, "client") and request.client:
                            client_ip = request.client.host
                    except Exception:
                        pass

                    key = f"ip:{client_ip}"
                    if not rate_limiter.is_allowed(key, RATE_LIMIT_PER_IP, RATE_LIMIT_WINDOW):
                        from tools.server.metrics import RATE_LIMIT_HITS
                        RATE_LIMIT_HITS.labels(key_type="ip").inc()
                        raise HTTPException(429, None, "Rate limit exceeded")

                return await endpoint()

            if self.args.api_key is not None:
                return verify
            else:
                return passthrough

        self.routes = Routes(
            routes,  # keep existing routes
            http_middlewares=[api_auth],  # apply api_auth middleware
        )

        # OpenAPIの設定
        self.openapi = OpenAPI(
            Info(
                {
                    "title": "Fish Speech API",
                    "version": "1.5.0",
                }
            ),
        ).routes

        # Initialize the app
        self.app = Kui(
            routes=self.routes + self.openapi[1:],  # Remove the default route
            exception_handlers={
                HTTPException: self.http_exception_handler,
                ValidationError: self.validation_exception_handler,
                Exception: self.other_exception_handler,
            },
            factory_class=FactoryClass(http=MsgPackRequest),
            cors_config=CORSConfig(),
        )

        # Add the state variables
        self.app.state.lock = Lock()
        self.app.state.device = self.args.device
        self.app.state.max_text_length = self.args.max_text_length

        # Associate the app with the model manager
        self.app.on_startup(self.initialize_app)

    async def initialize_app(self, app: Kui):
        # Make the ModelManager available to the views
        app.state.model_manager = ModelManager(
            mode=self.args.mode,
            device=self.args.device,
            half=self.args.half,
            compile=self.args.compile,
            llama_checkpoint_path=self.args.llama_checkpoint_path,
            decoder_checkpoint_path=self.args.decoder_checkpoint_path,
            decoder_config_name=self.args.decoder_config_name,
        )

        logger.info(f"Startup done, listening server at http://{self.args.listen}")


# Each worker process created by Uvicorn has its own memory space,
# meaning that models and variables are not shared between processes.
# Therefore, any variables (like `llama_queue` or `decoder_model`)
# will not be shared across workers.

# Multi-threading for deep learning can cause issues, such as inconsistent
# outputs if multiple threads access the same buffers simultaneously.
# Instead, it's better to use multiprocessing or independent models per thread.

if __name__ == "__main__":
    api = API()

    # IPv6 address format is [xxxx:xxxx::xxxx]:port
    match = re.search(r"\[([^\]]+)\]:(\d+)$", api.args.listen)
    if match:
        host, port = match.groups()  # IPv6
    else:
        host, port = api.args.listen.split(":")  # IPv4

    uvicorn.run(
        api.app,
        host=host,
        port=int(port),
        workers=api.args.workers,
        log_level="info",
    )
