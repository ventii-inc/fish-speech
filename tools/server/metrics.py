"""Prometheus metrics for Fish Speech API server."""

import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Request metrics
REQUEST_COUNT = Counter(
    "tts_requests_total",
    "Total number of TTS requests",
    ["endpoint", "method", "status", "format"],
)

REQUEST_DURATION = Histogram(
    "tts_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

REQUEST_IN_PROGRESS = Gauge(
    "tts_requests_in_progress",
    "Number of requests currently being processed",
    ["endpoint"],
)

# Audio metrics
AUDIO_BYTES_TOTAL = Counter(
    "tts_audio_bytes_total",
    "Total bytes of audio generated",
    ["format"],
)

AUDIO_DURATION_SECONDS = Counter(
    "tts_audio_duration_seconds_total",
    "Total duration of audio generated in seconds",
)

# Model metrics
MODEL_LOADED = Gauge(
    "tts_model_loaded",
    "Whether the TTS model is loaded (1) or not (0)",
)

MODEL_LOAD_TIME = Gauge(
    "tts_model_load_time_seconds",
    "Time taken to load the model in seconds",
)

# GPU metrics
GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory currently in use",
    ["device"],
)

GPU_MEMORY_TOTAL = Gauge(
    "gpu_memory_total_bytes",
    "Total GPU memory available",
    ["device"],
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percentage",
    ["device"],
)

# Rate limiting metrics
RATE_LIMIT_HITS = Counter(
    "tts_rate_limit_hits_total",
    "Number of requests rate limited",
    ["key_type"],  # "api_key" or "ip"
)


def update_gpu_metrics():
    """Update GPU memory metrics. Call periodically."""
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = f"cuda:{i}"
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory

                GPU_MEMORY_USED.labels(device=device).set(memory_allocated)
                GPU_MEMORY_TOTAL.labels(device=device).set(memory_total)
    except Exception:
        pass  # GPU metrics are best-effort


def get_metrics_response():
    """Generate Prometheus metrics response."""
    update_gpu_metrics()
    return generate_latest(), CONTENT_TYPE_LATEST


def track_request(endpoint: str):
    """Decorator to track request metrics."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            REQUEST_IN_PROGRESS.labels(endpoint=endpoint).inc()
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start_time
                REQUEST_IN_PROGRESS.labels(endpoint=endpoint).dec()
                REQUEST_DURATION.labels(endpoint=endpoint, method="POST").observe(
                    duration
                )

        return wrapper

    return decorator
