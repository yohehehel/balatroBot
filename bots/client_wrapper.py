"""JSON-RPC 2.0 client wrapper for the Balatro API.

Provides a synchronous HTTP client with configurable delay between actions,
retry logic, and timeout handling. Points to http://localhost:12346/ by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Error codes from docs/api.md
# ---------------------------------------------------------------------------
INTERNAL_ERROR = -32000
BAD_REQUEST = -32001
INVALID_STATE = -32002
NOT_ALLOWED = -32003


class RPCError(Exception):
    """Raised when the JSON-RPC server returns an error response."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(f"[{code}] {message}")


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------

@dataclass
class ClientConfig:
    """Configuration for the RPC client."""

    base_url: str = "http://localhost:12346"
    delay: float = 0.8  # seconds between consecutive API calls
    timeout: float = 30.0  # HTTP request timeout
    max_retries: int = 3
    retry_backoff: float = 1.0  # seconds, multiplied by attempt number


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------

class BalatroClient:
    """Synchronous JSON-RPC 2.0 client for the Balatro API.

    Handles:
    - Auto-incrementing request IDs
    - Configurable inter-action delay (``--delay``)
    - Retry on transient connection errors
    - Structured error raising via :class:`RPCError`
    """

    def __init__(self, config: ClientConfig | None = None):
        self._config = config or ClientConfig()
        self._request_id: int = 0
        self._last_call_time: float = 0.0
        self._client: httpx.Client = httpx.Client(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.timeout),
        )

    # -- public API ----------------------------------------------------------

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and return the ``result`` dict.

        Raises:
            RPCError: If the response contains a JSON-RPC ``error`` object.
            httpx.HTTPError: On transport-level failures after exhausting retries.
        """
        self._wait_delay()
        self._request_id += 1

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._request_id,
        }
        if params:
            payload["params"] = params

        response_data = self._send_with_retry(payload)

        if "error" in response_data:
            err = response_data["error"]
            raise RPCError(
                code=err.get("code", INTERNAL_ERROR),
                message=err.get("message", "Unknown error"),
                data=err.get("data"),
            )

        return response_data.get("result", {})

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "BalatroClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals -----------------------------------------------------------

    def _wait_delay(self) -> None:
        """Respect the configured delay between consecutive API calls."""
        if self._config.delay <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < self._config.delay:
            time.sleep(self._config.delay - elapsed)
        self._last_call_time = time.monotonic()

    def _send_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST the *payload* with retry on connection errors."""
        last_exc: Exception | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = self._client.post("/", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._config.max_retries:
                    time.sleep(self._config.retry_backoff * attempt)
        raise last_exc  # type: ignore[misc]
