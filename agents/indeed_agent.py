"""
agents/indeed_agent.py — Indeed Easy Apply agent.

Search strategy:
  - Builds URL with Easy Apply filter (DSQF7 attribute key)
  - Paginates via &start= offset (10 results per page)
  - Extracts job key from data-jk attribute for dedup
  - Opens right-panel detail, triggers "Easy Apply" button
  - Hands off to BaseAgent._execute_actions() for LLM form fill
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import AsyncIterator

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from browser_engine import BrowserEngine
from config import settings
from profile_schema import CandidateProfile

from .base_agent import ApplicationResult, ApplicationStatus, BaseAgent, JobListing

logger = logging.getLogger(__name__)

_INDEED_BASE = "https://www.indeed.com"
_RESULTS_PER_PAGE = 10

# DSQF7 = Indeed's internal key for "Easy Apply only" filter
_EASY_APPLY_FILTER = "attr(DSQF7)"


class IndeedAgent(BaseAgent):
    def __init__(self, engine: BrowserEngine, profile: CandidateProfile) -> None:
        super().__init__(engine, profile)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_jobs(self) -> AsyncIterator[JobListing]:
        page = await self.engine.new_page()
        try:
            for title in settings.target_titles:
                for location in settings.target_locations:
                    async for job in self._search_query(page, title, location):
                        yield job
        finally:
            await page.close()

    async def _search_query(
        self, page: Page, title: str, location: str
    ) -> AsyncIterator[JobListing]:
        for page_num in range(settings.max_pages_per_search):
            start = page_num * _RESULTS_PER_PAGE
            url = self._build_search_url(title, location, start)

            logger.info("[Indeed] Searching: %s in %s (page %d)", title, location, page_num + 1)

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                await BrowserEngine.human_delay(1000, 2500)
            except PlaywrightTimeout:
                logger.warning("[Indeed] Page load timeout for %s", url)
                return

            # Check for CAPTCHA / bot detection
            if await self._is_blocked(page):
                logger.error("[Indeed] Bot detection triggered — stopping Indeed search")
                return

            jobs = await self._extract_job_cards(page)
            if not jobs:
                logger.info("[Indeed] No more results for '%s' in '%s'", title, location)
                return

            for job in jobs:
                yield job

            # Check if "Next" page exists
            next_btn = page.locator("a[aria-label='Next Page']")
            if not await next_btn.count():
                break

    def _build_search_url(self, title: str, location: str, start: int = 0) -> str:
        params = {
            "q": title,
            "l": location,
            "fromage": "7",          # posted within last 7 days
            "sc": f"0kf:{_EASY_APPLY_FILTER}",
            "start": str(start),
        }
        return f"{_INDEED_BASE}/jobs?{urllib.parse.urlencode(params)}"

    async def _extract_job_cards(self, page: Page) -> list[JobListing]:
        jobs: list[JobListing] = []
        try:
            cards = await page.locator("div.job_seen_beacon").all()
            if not cards:
                cards = await page.locator("[data-jk]").all()
        except Exception:
            return jobs

        for card in cards:
            try:
                job_key = await card.get_attribute("data-jk") or ""
                if not job_key:
                    job_key = await card.locator("[data-jk]").first.get_attribute("data-jk") or ""

                title_el = card.locator("h2.jobTitle span[title], h2.jobTitle a span").first
                title = (await title_el.text_content() or "").strip()

                company_el = card.locator("[data-testid='company-name'], .companyName").first
                company = (await company_el.text_content() or "").strip()

                job_url = f"{_INDEED_BASE}/rc/clk?jk={job_key}" if job_key else ""

                # Confirm Easy Apply badge is present on this card
                easy_apply = bool(
                    await card.locator("span:has-text('Easily apply'), .iaLabel").count()
                )

                if job_key and title:
                    jobs.append(
                        JobListing(
                            platform="indeed",
                            job_key=job_key,
                            title=title,
                            company=company,
                            url=job_url,
                            easy_apply=easy_apply,
                        )
                    )
            except Exception as exc:
                logger.debug("[Indeed] Card parse error: %s", exc)

        return jobs

    # ------------------------------------------------------------------
    # Apply flow
    # ------------------------------------------------------------------

    async def _open_apply_flow(self, page: Page, job: JobListing) -> bool:
        """
        Navigate to job, wait for right-panel details, click Easy Apply.
        Returns True if the apply modal is open and ready for form fill.
        """
        try:
            await page.goto(
                f"{_INDEED_BASE}/viewjob?jk={job.job_key}",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            await BrowserEngine.human_delay(1000, 2000)
        except PlaywrightTimeout:
            logger.warning("[Indeed] Timeout loading job page: %s", job.job_key)
            return False

        if self._is_external_redirect(page.url):
            return False

        # Extract description for job-fit context (stored in job object)
        try:
            desc_el = page.locator("#jobDescriptionText, .jobsearch-jobDescriptionText").first
            job.description = (await desc_el.text_content() or "")[:3000]
        except Exception:
            pass

        # Locate Easy Apply button (multiple possible selectors across Indeed versions)
        apply_btn = page.locator(
            "button.ia-IndeedApplyButton, "
            "button[data-testid='indeedApplyButton'], "
            "button:has-text('Apply now'), "
            "span.indeed-apply-widget button"
        ).first

        if not await apply_btn.count():
            logger.info("[Indeed] No Easy Apply button on %s — skipping", job.title)
            return False

        try:
            await apply_btn.wait_for(state="visible", timeout=8000)
            await BrowserEngine.human_delay(300, 700)
            await apply_btn.click()
            # Wait for the modal/iframe to appear
            await page.wait_for_selector(
                "[role='dialog'], .ia-BasePage, #ia-container",
                timeout=15_000,
            )
            return True
        except PlaywrightTimeout:
            logger.warning("[Indeed] Apply modal did not appear for %s", job.title)
            return False

    # ------------------------------------------------------------------
    # Bot detection check
    # ------------------------------------------------------------------

    @staticmethod
    async def _is_blocked(page: Page) -> bool:
        title = (await page.title()).lower()
        url = page.url.lower()
        blocked_signals = ["captcha", "robot", "blocked", "unusual traffic", "verify"]
        return any(s in title or s in url for s in blocked_signals)
