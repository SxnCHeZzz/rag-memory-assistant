"""Robust async Ollama client for FastAPI (Windows-safe, proxy-safe, timeout-safe)."""

import logging
from urllib.parse import urlparse

import httpx

from app.config import settings

logger = logging.getLogger("ollama")


def _fix_windows_host(host: str) -> str:
    parsed = urlparse(host)
    hostname = parsed.hostname or ""

    if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
        netloc = f"127.0.0.1:{parsed.port}" if parsed.port else "127.0.0.1"
        return parsed._replace(netloc=netloc).geturl()

    return host.rstrip("/")


class OllamaUnavailableError(Exception):
    """LLM недоступен или вернул ошибку."""
    pass


class OllamaClient:
    """
    Robust async Ollama client.
    Windows-safe, proxy-safe, timeout-safe.
    """

    def __init__(
        self,
        host: str = settings.ollama_host,
        model: str = settings.ollama_model,
        timeout: float = settings.ollama_timeout,
        num_ctx: int = settings.ollama_num_ctx,
    ):
        self.host = _fix_windows_host(host)
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        logger.info("OllamaClient initialized: host=%s model=%s", self.host, self.model)

   
    # PUBLIC API
    async def health(self) -> bool:
        """Проверяет доступность Ollama и наличие модели."""
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                r = await client.get(f"{self.host}/api/tags")
                if r.status_code != 200:
                    logger.warning(
                        "Ollama /api/tags returned %d from %s",
                        r.status_code, self.host
                    )
                    return False

                data = r.json()
                models = [m.get("name", "") for m in data.get("models", [])]

                model_found = any(self.model in m for m in models)
                if not model_found:
                    logger.warning(
                        "Model %s not found. Available: %s",
                        self.model, models
                    )
                    return False

                logger.info("Ollama health OK: model=%s available", self.model)
                return True

        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama at %s: %s", self.host, e)
            return False
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Генерирует ответ через Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                trust_env=False,
            ) as client:
                resp = await client.post(
                    f"{self.host}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")

        except httpx.TimeoutException as e:
            logger.error("Ollama timeout after %.1fs: %s", self.timeout, e)
            raise OllamaUnavailableError(f"LLM timeout after {self.timeout}s")

        except httpx.HTTPStatusError as e:
            logger.error(
                "Ollama HTTP error %s: body=%r",
                e.response.status_code,
                e.response.text[:300],
            )
            raise OllamaUnavailableError(f"HTTP {e.response.status_code}")

        except httpx.ConnectError as e:
            logger.error("Ollama connection error: %s", e)
            raise OllamaUnavailableError("LLM unreachable")

        except Exception as e:
            logger.exception("Unexpected Ollama error")
            raise OllamaUnavailableError(str(e))
