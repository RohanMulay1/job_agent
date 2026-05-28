"""
main.py — Orchestrator entry point.

Usage:
    python main.py

Flow:
  1. Load config + profile
  2. Launch persistent browser (reuses session from setup_auth.py)
  3. All enabled platforms run IN PARALLEL — each gets its own browser tab
  4. Print combined run summary
  5. Graceful shutdown on Ctrl+C
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import re
import sys
from pathlib import Path

from agents.base_agent import ApplicationResult, ApplicationStatus, JobListing
from agents.indeed_agent import IndeedAgent
from agents.naukri_agent import NaukriAgent
from agents.wellfound_agent import WellfoundAgent
from agents.yc_agent import YCAgent
from browser_engine import BrowserEngine
from config import settings
from db import JobDB
from llm_brain import evaluate_job_fit
from profile_schema import CandidateProfile, load_profile

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    fh = logging.handlers.RotatingFileHandler(
        settings.log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform registry
# ---------------------------------------------------------------------------

_AGENT_MAP = {
    "indeed": IndeedAgent,
    "naukri": NaukriAgent,
    "wellfound": WellfoundAgent,
    "yc": YCAgent,
}

# ---------------------------------------------------------------------------
# Title relevance filter
# Short tokens (ai, ml, sde, n8n) use \b word boundaries to avoid matching
# inside longer words like "trainee"→ai, "international"→intern.
# ---------------------------------------------------------------------------

_RELEVANT_TITLE_KEYWORDS = [
    r"\bai\b", r"\bml\b", r"\bsde\b", r"\bn8n\b", r"\bnlp\b", r"\badas\b", r"\bllm\b",
    "machine learning", "deep learning", "natural language",
    "computer vision", "large language", "data science", "data scientist",
    "artificial intelligence", "neural network",
    "python developer", "python engineer",
    "software engineer", "software developer",
    "backend engineer", "backend developer",
    "full stack", "fullstack",
    "ai agent", "autonomous", "perception engineer",
    "automation engineer", "ai automation", "workflow automation",
    "founding engineer", "founding ai",
    "software intern", "engineer intern", "developer intern",
    "research intern", "ai intern", "ml intern", "sde intern",
    "data science intern", "deep learning intern",
]

_TITLE_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in _RELEVANT_TITLE_KEYWORDS]


def _has_relevant_title(job: JobListing) -> bool:
    return any(p.search(job.title) for p in _TITLE_PATTERNS)


def _is_eligible_by_location(job: JobListing) -> bool:
    india_keywords = settings.india_locations
    searchable = " ".join([job.title or "", job.company or "", job.url or "", job.description or ""]).lower()
    if any(kw in searchable for kw in india_keywords):
        return True
    return "remote" in searchable


def _make_result(job: JobListing, status: ApplicationStatus, notes: str) -> ApplicationResult:
    return ApplicationResult(job=job, status=status, notes=notes)


# ---------------------------------------------------------------------------
# Per-platform runner (runs concurrently)
# ---------------------------------------------------------------------------


async def _run_platform(
    platform: str,
    engine: BrowserEngine,
    db: JobDB,
    profile: CandidateProfile,
    counters: dict[str, int],
    applied_lock: asyncio.Lock,
    applied_ref: list[int],  # mutable single-element list as shared counter
) -> None:
    agent_cls = _AGENT_MAP.get(platform)
    if not agent_cls:
        logger.warning("Unknown platform '%s' — skipping", platform)
        return

    agent = agent_cls(engine, profile)
    apply_page = await engine.new_page()
    logger.info("--- Running %s agent ---", platform.capitalize())

    try:
        async for job in agent.search_jobs():
            async with applied_lock:
                if applied_ref[0] >= settings.max_applications_per_run:
                    logger.info("[%s] Reached max applications — stopping", platform)
                    break

            # Deduplication
            if db.is_already_seen(job.platform, job.job_key):
                logger.debug("Already seen: %s @ %s", job.title, job.company)
                async with applied_lock:
                    counters["skipped_seen"] = counters.get("skipped_seen", 0) + 1
                continue

            # Title relevance pre-filter (non-YC only)
            if platform != "yc" and not _has_relevant_title(job):
                logger.info("[%s] Skipped (irrelevant title): %s", platform, job.title)
                db.record_seen(job)
                continue

            if platform != "yc":
                if not job.easy_apply:
                    db.record_seen(job)
                    continue
                # Wellfound searches are already URL-scoped to specific locations
                # so skip the text-based location filter there
                if platform != "wellfound" and not _is_eligible_by_location(job):
                    logger.info("[%s] Skipped (location): %s @ %s", platform, job.title, job.company)
                    db.record_seen(job)
                    continue

            # LLM fit screening (skip for YC)
            if platform != "yc":
                logger.info("[%s] Evaluating fit: %s @ %s", platform, job.title, job.company)
                fit = await evaluate_job_fit(job.title, job.description or job.title, profile)
                if not fit.match:
                    logger.info("[%s] Skipped (fit %.2f): %s", platform, fit.score, job.title)
                    db.record_result(_make_result(job, ApplicationStatus.SKIPPED_FIT, fit.reason))
                    async with applied_lock:
                        counters["skipped_fit"] = counters.get("skipped_fit", 0) + 1
                    continue

            # Apply
            result = await agent.apply_to_job(apply_page, job)
            db.record_result(result)

            status_key = result.status.value
            async with applied_lock:
                counters[status_key] = counters.get(status_key, 0) + 1
                if result.status == ApplicationStatus.APPLIED:
                    applied_ref[0] += 1
                    logger.info(
                        "[✓] Applied (%d/%d): %s @ %s",
                        applied_ref[0], settings.max_applications_per_run,
                        job.title, job.company,
                    )
                elif result.status == ApplicationStatus.NEEDS_HUMAN:
                    logger.warning("[!] Needs human: %s @ %s — %s", job.title, job.company, result.notes)

            await BrowserEngine.human_delay(2000, 4000)

    finally:
        await apply_page.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run() -> None:
    _setup_logging()
    logger.info("=== Job Agent starting ===")
    logger.info("Platforms: %s", settings.enabled_platforms)
    logger.info("Max applications this run: %d", settings.max_applications_per_run)

    profile_path = settings.sample_profile_path
    if not profile_path.exists():
        logger.error("Profile not found: %s — run setup first", profile_path)
        sys.exit(1)

    profile: CandidateProfile = load_profile(profile_path)
    logger.info("Profile loaded: %s", profile.personal_info.full_name)

    counters: dict[str, int] = {}
    applied_lock = asyncio.Lock()
    applied_ref = [0]  # shared applied counter

    with JobDB() as db:
        async with BrowserEngine() as engine:
            # All platforms launch simultaneously; return_exceptions=True so one crash
            # doesn't cancel the other platforms.
            results = await asyncio.gather(
                *[
                    _run_platform(platform, engine, db, profile, counters, applied_lock, applied_ref)
                    for platform in settings.enabled_platforms
                    if platform in _AGENT_MAP
                ],
                return_exceptions=True,
            )
            for platform, res in zip(
                [p for p in settings.enabled_platforms if p in _AGENT_MAP], results
            ):
                if isinstance(res, Exception):
                    logger.error("[%s] Platform crashed: %s", platform, res)

    _print_summary(counters, db_path=settings.db_path)


def _print_summary(counters: dict[str, int], db_path: Path) -> None:
    total = sum(counters.values())
    print("\n" + "=" * 55)
    print("  Job Agent Run Complete")
    print("=" * 55)
    print(f"  {'Applied:':<25} {counters.get('applied', 0)}")
    print(f"  {'Skipped (external ATS):':<25} {counters.get('skipped_external', 0)}")
    print(f"  {'Skipped (poor fit):':<25} {counters.get('skipped_fit', 0)}")
    print(f"  {'Skipped (already seen):':<25} {counters.get('skipped_seen', 0)}")
    print(f"  {'Needs human review:':<25} {counters.get('needs_human', 0)}")
    print(f"  {'Failed:':<25} {counters.get('failed', 0)}")
    print(f"  {'Total processed:':<25} {total}")
    print("=" * 55)
    print(f"  Full log: {settings.log_path}")
    print(f"  Database: {db_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[Interrupted] Shutting down gracefully...")
        sys.exit(0)
