import traceback
from http import HTTPStatus

from kui.asgi import HTTPException, JSONResponse
from loguru import logger
from pydantic import ValidationError


class ExceptionHandler:

    async def http_exception_handler(self, exc: HTTPException):
        logger.warning(f"[HTTP {exc.status_code}] {exc.content}")
        return JSONResponse(
            dict(
                statusCode=exc.status_code,
                message=exc.content,
                error=HTTPStatus(exc.status_code).phrase,
            ),
            exc.status_code,
            exc.headers,
        )

    async def validation_exception_handler(self, exc: ValidationError):
        errors = exc.errors()
        logger.error(f"[VALIDATION ERROR] {errors}")
        return JSONResponse(
            dict(
                statusCode=422,
                message=errors,
                error="Unprocessable Entity",
            ),
            422,
        )

    async def other_exception_handler(self, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        traceback.print_exc()

        status = HTTPStatus.INTERNAL_SERVER_ERROR
        return JSONResponse(
            dict(statusCode=status, message=str(exc), error=status.phrase),
            status,
        )
