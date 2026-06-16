"""
Push notification module - PushPlus and WeCom webhook support.
Instruction #5: Push stability rules
  - Global rate limiter caps pushes per minute
  - Single send, no batching; retry only once
  - Push runs in isolated background thread (PushDispatcher)
  - Minimal logging: only success/failure per push
"""
import requests
import json
import time
import logging
import threading
from collections import deque
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Session that bypasses system proxy settings
_no_proxy_session = requests.Session()
_no_proxy_session.trust_env = False


class PushRateLimiter:
    """Sliding-window rate limiter: caps total pushes per minute globally."""

    def __init__(self, max_per_minute: int = 10):
        self._max_per_minute = max_per_minute
        self._window_secs = 60.0
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Try to acquire a send slot. Returns 0.0 if OK, or wait-seconds if throttled."""
        if self._max_per_minute <= 0:
            return 0.0
        now = time.monotonic()
        with self._lock:
            # Drop expired entries outside the window
            cutoff = now - self._window_secs
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) < self._max_per_minute:
                self._timestamps.append(now)
                return 0.0
            else:
                # Wait until the oldest slot expires
                wait = self._timestamps[0] - cutoff
                return max(wait, 0.0)


class WeChatNotifier:
    """WeChat notifier via PushPlus and WeCom webhooks."""

    def __init__(self, pushplus_token: str = "", wecom_webhook: str = ""):
        self.pushplus_token = pushplus_token
        self.wecom_webhook = wecom_webhook
        self._send_lock = threading.Lock()

    def send_via_pushplus(self, title: str, content: str) -> bool:
        if not self.pushplus_token:
            return False
        url = "https://www.pushplus.plus/send"
        payload = {
            "token": self.pushplus_token,
            "title": title,
            "content": content,
            "template": "txt",
        }
        try:
            resp = _no_proxy_session.post(url, json=payload, timeout=30)
            result = resp.json()
            return result.get("code") == 200
        except Exception:
            return False

    def send_via_wecom(self, content: str) -> bool:
        if not self.wecom_webhook:
            return False
        try:
            resp = _no_proxy_session.post(
                self.wecom_webhook,
                json={"msgtype": "markdown", "markdown": {"content": content}},
                timeout=30,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def send_single(self, title: str, content: str, retry_count: int = 1) -> bool:
        """Send one alert. Attempt once; if it fails, retry up to `retry_count` times.
        Returns True if any attempt succeeded."""
        with self._send_lock:
            # First attempt
            ok = self.send_via_pushplus(title, content)
            if ok:
                return True
            if self.wecom_webhook:
                ok = self.send_via_wecom(content)
                if ok:
                    return True

            # Retry (only if first attempt completely failed)
            for attempt in range(retry_count):
                ok = self.send_via_pushplus(title, content)
                if ok:
                    return True
                if self.wecom_webhook:
                    ok = self.send_via_wecom(content)
                    if ok:
                        return True
            return False


class PushDispatcher:
    """Background push worker: isolated thread with its own queue and rate limiter.
    One alert = one push. Failures don't affect data collection/computation."""

    def __init__(
        self,
        notifier: WeChatNotifier,
        max_per_minute: int = 10,
        retry_count: int = 1,
    ):
        self._notifier = notifier
        self._rate_limiter = PushRateLimiter(max_per_minute)
        self._retry_count = retry_count
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Launch the background push worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker, name="push-dispatcher", daemon=True
        )
        self._worker_thread.start()
        logger.info("PushDispatcher started (isolated thread)")

    def stop(self, drain: bool = True, timeout: float = 10.0):
        """Stop the dispatcher, optionally draining the queue first."""
        with self._lock:
            self._running = False
            self._cond.notify_all()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def enqueue(self, title: str, content: str):
        """Queue a single alert for delivery. Non-blocking, thread-safe."""
        with self._lock:
            self._queue.append((title, content))
            self._cond.notify()

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    # ----------------------------------------------------------------
    # Worker loop (runs in background daemon thread)
    # ----------------------------------------------------------------
    def _worker(self):
        while True:
            item = self._dequeue()
            if item is None:
                # Stopped with empty queue
                return

            title, content = item

            # Rate-limit: wait if throttled
            wait = self._rate_limiter.acquire()
            if wait > 0:
                time.sleep(wait)

            # Send single push with retry
            ok = self._notifier.send_single(title, content, self._retry_count)

            # Minimal logging: only success/failure per push
            if ok:
                logger.info(f"Push OK: {title}")
            else:
                logger.error(f"Push FAILED: {title}")

    def _dequeue(self):
        """Wait for the next item or stop signal. Returns None when stopped."""
        with self._cond:
            while self._running and not self._queue:
                self._cond.wait(timeout=5.0)
            if not self._running:
                # Drain remaining items on stop if any
                if self._queue:
                    return self._queue.popleft()
                return None
            return self._queue.popleft()
