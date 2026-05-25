"""
db.py — SQLite job tracker + dashboard sync.

Ensures the agent never applies to the same job twice across runs.
After every write, optionally syncs the record to the Vercel dashboard
via a fire-and-forget async POST (requires httpx).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from agents.base_agent import ApplicationResult, ApplicationStatus, JobListing
from config import settings

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    platform    TEXT    NOT NULL,
    job_key     TEXT    NOT NULL,
    title       TEXT,
    company     TEXT,
    url         TEXT,
    status      TEXT    NOT NULL,
    applied_at  TEXT,
    notes       TEXT,
    UNIQUE(platform, job_key)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_platform_key ON jobs (platform, job_key);
"""


class JobDB:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or settings.db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("JobDB opened: %s", self._path)

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_INDEX)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def is_already_seen(self, platform: str, job_key: str) -> bool:
        row = self._conn.execute(
            "SELECT id FROM jobs WHERE platform = ? AND job_key = ?",
            (platform, job_key),
        ).fetchone()
        return row is not None

    def get_stats(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def count_applied_today(self) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = ? AND applied_at LIKE ?",
            (ApplicationStatus.APPLIED.value, f"{today}%"),
        ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_result(self, result: ApplicationResult) -> None:
        job = result.job
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO jobs
                    (platform, job_key, title, company, url, status, applied_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.platform,
                    job.job_key,
                    job.title,
                    job.company,
                    job.url,
                    result.status.value,
                    now if result.status == ApplicationStatus.APPLIED else None,
                    result.notes[:500] if result.notes else None,
                ),
            )
        logger.debug(
            "Recorded [%s] %s @ %s → %s",
            job.platform, job.title, job.company, result.status.value,
        )
        # Fire-and-forget sync to dashboard (non-blocking)
        asyncio.ensure_future(self._sync_to_dashboard(result))

    async def _sync_to_dashboard(self, result: ApplicationResult) -> None:
        """Push a single result to the Vercel dashboard API. Silently swallows errors."""
        if not _HTTPX_AVAILABLE or not settings.dashboard_api_url or not settings.dashboard_api_key:
            return

        job = result.job
        payload = {
            "platform": job.platform,
            "job_key": job.job_key,
            "title": job.title,
            "company": job.company,
            "url": job.url,
            "status": result.status.value,
            "applied_at": (
                datetime.now().isoformat()
                if result.status == ApplicationStatus.APPLIED
                else None
            ),
            "notes": result.notes[:500] if result.notes else None,
        }

        url = settings.dashboard_api_url.rstrip("/") + "/api/jobs"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"x-api-key": settings.dashboard_api_key},
                )
                if resp.status_code not in (200, 201):
                    logger.warning("Dashboard sync failed: %s %s", resp.status_code, resp.text[:200])
                else:
                    logger.debug("Dashboard sync OK for %s", job.title)
        except Exception as exc:
            logger.debug("Dashboard sync error (non-fatal): %s", exc)

    def record_seen(self, job: JobListing) -> None:
        """Mark a job as seen (without a full result) to avoid re-processing."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO jobs
                    (platform, job_key, title, company, url, status, applied_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (job.platform, job.job_key, job.title, job.company, job.url, "seen"),
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
        logger.info("JobDB closed")

    def __enter__(self) -> "JobDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
