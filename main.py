"""
main.py — Orchestrator entry point.

Usage:
    python main.py

Flow:
  1. Load config + profile
  2. Launch persistent browser (reuses session from setup_auth.py)
  3. For each enabled platform:
       search_jobs() → filter seen/fit → apply_to_job() → record result
  4. Print run summary
  5. Graceful shutdown on Ctrl+C
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import sys
from pathlib import Path

from agents.base_agent import ApplicationStatus, JobListing
from agents.indeed_agent import IndeedAgent
from agents.naukri_agent import NaukriAgent
from agents.wellfound_agent import WellfoundAgent
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

    # Rotating file handler — 5 MB × 3 backups
    fh = logging.handlers.RotatingFileHandler(
        settings.log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
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
}

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def run() -> None:
    _setup_logging()
    logger.info("=== Job Agent starting ===")
    logger.info("Platforms: %s", settings.enabled_platforms)
    logger.info("Titles: %s", settings.target_titles)
    logger.info("Locations: %s", settings.target_locations)
    logger.info("Max applications this run: %d", settings.max_applications_per_run)

    # Load candidate profile
    profile_path = settings.sample_profile_path
    if not profile_path.exists():
        logger.error("Profile not found: %s — run setup first", profile_path)
        sys.exit(1)

    profile: CandidateProfile = load_profile(profile_path)
    logger.info("Profile loaded: %s", profile.personal_info.full_name)

    counters = {
        "applied": 0,
        "skipped_external": 0,
        "skipped_fit": 0,
        "skipped_seen": 0,
        "failed": 0,
        "needs_human": 0,
    }
    applied_this_run = 0

    with JobDB() as db:
        async with BrowserEngine() as engine:
            apply_page = await engine.new_page()

            for platform in settings.enabled_platforms:
                agent_cls = _AGENT_MAP.get(platform)
                if not agent_cls:
                    logger.warning("Unknown platform '%s' — skipping", platform)
                    continue

                agent = agent_cls(engine, profile)
                logger.info("--- Running %s agent ---", platform.capitalize())

                async for job in agent.search_jobs():
                    if applied_this_run >= settings.max_applications_per_run:
                        logger.info("Reached max applications (%d) — stopping", settings.max_applications_per_run)
                        break

                    # Deduplication
                    if db.is_already_seen(job.platform, job.job_key):
                        logger.debug("Already seen: %s @ %s", job.title, job.company)
                        counters["skipped_seen"] += 1
                        continue

                    # Skip non-easy-apply listings found despite the filter
                    if not job.easy_apply:
                        logger.debug("Not Easy Apply, skipping: %s", job.title)
                        db.record_seen(job)
                        continue

                    # Location filter:
                    # India roles → apply to everything
                    # Non-India roles → only apply if the listing is remote
                    if not _is_eligible_by_location(job):
                        logger.info(
                            "Skipped (non-India, not remote): %s @ %s", job.title, job.company
                        )
                        db.record_seen(job)
                        continue

                    # LLM job-fit screening
                    logger.info("Evaluating fit: %s @ %s", job.title, job.company)
                    fit = await evaluate_job_fit(job.title, job.description or job.title, profile)

                    if not fit.match:
                        logger.info(
                            "Skipped (fit %.2f): %s — %s", fit.score, job.title, fit.reason
                        )
                        db.record_result(
                            _make_result(job, ApplicationStatus.SKIPPED_FIT, fit.reason)
                        )
                        counters["skipped_fit"] += 1
                        continue

                    # Apply
                    result = await agent.apply_to_job(apply_page, job)
                    db.record_result(result)

                    status_key = result.status.value
                    counters[status_key] = counters.get(status_key, 0) + 1

                    if result.status == ApplicationStatus.APPLIED:
                        applied_this_run += 1
                        logger.info(
                            "[✓] Applied (%d/%d): %s @ %s",
                            applied_this_run, settings.max_applications_per_run,
                            job.title, job.company,
                        )
                    elif result.status == ApplicationStatus.NEEDS_HUMAN:
                        logger.warning(
                            "[!] Needs human: %s @ %s — %s", job.title, job.company, result.notes
                        )

                    # Brief pause between applications
                    await BrowserEngine.human_delay(3000, 6000)

            await apply_page.close()

    # Summary
    _print_summary(counters, db_path=settings.db_path)


def _is_eligible_by_location(job: JobListing) -> bool:
    """
    Apply to ALL India-based roles unconditionally.
    For roles outside India (or unknown location), only apply if
    the title/company/description contains 'remote'.
    """
    india_keywords = settings.india_locations  # lowercase list from config

    searchable = " ".join([
        job.title or "",
        job.company or "",
        job.url or "",
        job.description or "",
    ]).lower()

    # If any India keyword appears in the job's metadata → eligible
    if any(kw in searchable for kw in india_keywords):
        return True

    # Otherwise require the word "remote" to appear
    return "remote" in searchable


def _make_result(job: JobListing, status: ApplicationStatus, notes: str):
    from agents.base_agent import ApplicationResult
    return ApplicationResult(job=job, status=status, notes=notes)


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
