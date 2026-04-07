from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

from app.config import GMAIL_POLL_INTERVAL_MINUTES, GMAIL_QUERY
from services.gmail_polling import poll_gmail_once
from storage.db import get_state, set_state


class GmailPollingWorker:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        logger = logging.getLogger(__name__)
        while not self._stop_event.is_set():
            last_checked = get_state("last_checked_ts")
            if last_checked:
                query = f"{GMAIL_QUERY} after:{last_checked}"
            else:
                query = f"{GMAIL_QUERY} newer_than:1d"

            logger.info("Polling Gmail with query: %s", query)
            try:
                poll_gmail_once(query)
                set_state("last_checked_ts", str(int(time.time())))
            except Exception as exc:
                logger.exception("Gmail polling failed: %s", exc)

            interval = max(1, int(GMAIL_POLL_INTERVAL_MINUTES))
            for _ in range(interval * 60):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

