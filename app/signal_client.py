"""Thin client for the signal-cli-rest-api with retry logic."""

import base64
import time
import httpx
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]  # seconds between retries


class SignalClient:
    def __init__(self, base_url: str, number: str):
        self.base_url = base_url.rstrip("/")
        self.number = number
        self._http = httpx.Client(base_url=self.base_url, timeout=60)

    def _retry(self, operation: str, func, *args, **kwargs):
        """Execute a function with retries on failure."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        "Signal %s failed (attempt %d/%d): %s — retrying in %ds",
                        operation, attempt + 1, MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Signal %s failed after %d attempts: %s",
                        operation, MAX_RETRIES, exc,
                    )
        return last_exc

    # ── Receive ──────────────────────────────────────────────
    def receive(self) -> list[dict]:
        """Poll for new incoming messages via the REST API."""
        try:
            resp = self._http.get(f"/v1/receive/{self.number}?timeout=1")
            if resp.status_code == 200:
                return resp.json() or []
            return []
        except Exception as exc:
            logger.error("receive failed: %s", exc)
            return []

    # ── Group ID resolution ───────────────────────────────────
    def _resolve_group_id(self, group_id: str) -> str:
        """Convert an internal group ID to the group.XXX= form needed by /v2/send."""
        if group_id.startswith("group."):
            return group_id
        try:
            resp = self._http.get(f"/v1/groups/{self.number}")
            resp.raise_for_status()
            for g in resp.json():
                if g.get("internal_id") == group_id:
                    return g["id"]
        except Exception:
            pass
        return group_id

    # ── React ─────────────────────────────────────────────────
    def react(self, recipient: str, target_author: str, timestamp: int, emoji: str = "🤖") -> bool:
        """Send an emoji reaction to a specific message."""
        try:
            resp = self._http.post(
                f"/v1/reactions/{self.number}",
                json={
                    "reaction": emoji,
                    "recipient": self._resolve_group_id(recipient),
                    "target_author": target_author,
                    "timestamp": timestamp,
                },
            )
            return resp.status_code in (200, 204)
        except Exception as exc:
            logger.error("React failed: %s", exc)
            return False

    # ── Send ─────────────────────────────────────────────────
    def send(self, recipient: str, message: str) -> bool:
        """Send a text message to a single recipient with retry.

        Long messages are chunked at 2000 chars. Each chunk is retried
        independently on failure.
        """
        chunks = [message[i : i + 2000] for i in range(0, len(message), 2000)]
        for i, chunk in enumerate(chunks):
            def _do(text=chunk):
                resp = self._http.post(
                    "/v2/send",
                    json={
                        "message": text,
                        "number": self.number,
                        "recipients": [self._resolve_group_id(recipient)],
                    },
                )
                resp.raise_for_status()
                return True

            result = self._retry(f"send (chunk {i+1}/{len(chunks)})", _do)
            if isinstance(result, Exception):
                return False
        return True

    # ── Send voice ───────────────────────────────────────────
    def send_voice(self, recipient: str, ogg_bytes: bytes) -> bool:
        """Send an OGG/Opus voice note attachment."""
        encoded = base64.standard_b64encode(ogg_bytes).decode()

        def _do():
            resp = self._http.post(
                "/v2/send",
                json={
                    "message": "",
                    "number": self.number,
                    "recipients": [self._resolve_group_id(recipient)],
                    "base64_attachments": [f"data:audio/ogg;filename=voice.ogg;base64,{encoded}"],
                },
            )
            resp.raise_for_status()
            return True

        result = self._retry("send_voice", _do)
        return not isinstance(result, Exception)

    # ── Health ───────────────────────────────────────────────
    def is_healthy(self) -> bool:
        try:
            resp = self._http.get("/v1/about")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
