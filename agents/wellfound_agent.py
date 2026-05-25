"""
agents/wellfound_agent.py — Wellfound (AngelList Talent) native apply agent.

Search strategy:
  - URL: /jobs?q={title}&l={location}&jobType=full-time
  - Skip any job whose apply button says "Apply on company site" (external ATS)
  - Only proceed with native Wellfound applications
  - Pagination via scroll-to-load (infinite scroll) or &page= param
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import AsyncIterator

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from browser_engine import BrowserEngine
from config import settings
from profile_schema import CandidateProfile

from .base_agent import BaseAgent, JobListing

logger = logging.getLogger(__name__)

_WELLFOUND_BASE = "https://wellfound.com"


class WellfoundAgent(BaseAgent):
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
        url = self._build_search_url(title, location)
        logger.info("[Wellfound] Searching: %s in %s", title, location)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await BrowserEngine.human_delay(2000, 3500)
        except PlaywrightTimeout:
            logger.warning("[Wellfound] Page load timeout: %s", url)
            return

        if await self._is_blocked(page):
            logger.error("[Wellfound] Bot detection triggered")
            return

        for page_num in range(settings.max_pages_per_search):
            jobs = await self._extract_job_cards(page)
            if not jobs:
                logger.info("[Wellfound] No results found for '%s' in '%s'", title, location)
                return

            for job in jobs:
                yield job

            # Wellfound uses infinite scroll — scroll down to load more
            loaded = await self._scroll_for_more(page)
            if not loaded:
                break

    def _build_search_url(self, title: str, location: str) -> str:
        params = {
            "q": title,
            "jobType": "full-time",
        }
        # Wellfound handles "Remote" as a special location
        loc_lower = location.lower()
        if "remote" in loc_lower:
            params["remote"] = "true"
        else:
            params["l"] = location

        return f"{_WELLFOUND_BASE}/jobs?{urllib.parse.urlencode(params)}"

    async def _extract_job_cards(self, page: Page) -> list[JobListing]:
        jobs: list[JobListing] = []

        try:
            cards = await page.locator(
                "[data-test='JobListing'], "
                "div[class*='styles_component'][class*='JobListing'], "
                "div[class*='job-listing']"
            ).all()

            if not cards:
                # Broader fallback
                cards = await page.locator("a[href*='/jobs/']").all()
        except Exception:
            return jobs

        seen_keys: set[str] = set()

        for card in cards:
            try:
                # Extract job URL from the card link
                link = card.locator("a[href*='/jobs/']").first
                href = await link.get_attribute("href") or ""
                if not href:
                    href = await card.get_attribute("href") or ""

                if not href:
                    continue

                full_url = href if href.startswith("http") else _WELLFOUND_BASE + href
                job_key = href.split("/jobs/")[-1].split("?")[0].strip("/")

                if job_key in seen_keys:
                    continue
                seen_keys.add(job_key)

                # Title
                title_el = card.locator(
                    "h2, h3, [class*='title'], [data-test='job-title']"
                ).first
                title = (await title_el.text_content() or "").strip()

                # Company
                company_el = card.locator(
                    "[class*='company'], [data-test='company-name'], a[href*='/company/']"
                ).first
                company = (await company_el.text_content() or "").strip()

                # Native apply check — look for "Apply on company site" text
                external_signal = card.locator(
                    "span:has-text('Apply on company site'), "
                    "button:has-text('Apply on company site'), "
                    "a:has-text('Apply on company site')"
                )
                is_external = bool(await external_signal.count())

                if title and not is_external:
                    jobs.append(
                        JobListing(
                            platform="wellfound",
                            job_key=job_key,
                            title=title,
                            company=company,
                            url=full_url,
                            easy_apply=True,
                        )
                    )
            except Exception as exc:
                logger.debug("[Wellfound] Card parse error: %s", exc)

        return jobs

    # ------------------------------------------------------------------
    # Apply flow
    # ------------------------------------------------------------------

    async def _open_apply_flow(self, page: Page, job: JobListing) -> bool:
        if not job.url:
            return False

        try:
            await page.goto(job.url, wait_until="domcontentloaded", timeout=30_000)
            await BrowserEngine.human_delay(1500, 3000)
        except PlaywrightTimeout:
            logger.warning("[Wellfound] Timeout loading job: %s", job.url)
            return False

        if self._is_external_redirect(page.url):
            return False

        # Grab description
        try:
            desc_el = page.locator(
                "[class*='description'], [data-test='job-description'], .prose"
            ).first
            job.description = (await desc_el.text_content() or "")[:3000]
        except Exception:
            pass

        # Hard check — if the only apply option is external, skip
        external_btn = page.locator(
            "a:has-text('Apply on company site'), "
            "button:has-text('Apply on company site')"
        )
        if await external_btn.count():
            logger.info("[Wellfound] External-only apply for %s — skipping", job.title)
            return False

        # Find native apply button
        apply_btn = page.locator(
            "button:has-text('Apply'), "
            "button[data-test='apply-button'], "
            "a:has-text('Apply now')"
        ).first

        if not await apply_btn.count():
            logger.info("[Wellfound] No native apply button for %s", job.title)
            return False

        try:
            await apply_btn.wait_for(state="visible", timeout=8000)
            await BrowserEngine.human_delay(400, 800)
            await apply_btn.click()

            await page.wait_for_selector(
                "[role='dialog'], form[class*='apply'], [class*='ApplicationModal']",
                timeout=15_000,
            )
            return True
        except PlaywrightTimeout:
            logger.warning("[Wellfound] Apply modal did not appear for %s", job.title)
            return False

    # ------------------------------------------------------------------
    # Infinite scroll helper
    # ------------------------------------------------------------------

    async def _scroll_for_more(self, page: Page) -> bool:
        """Scroll to the bottom; return True if new cards loaded."""
        prev_count = await page.locator("[data-test='JobListing']").count()
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await BrowserEngine.human_delay(2000, 3500)
        new_count = await page.locator("[data-test='JobListing']").count()
        return new_count > prev_count

    @staticmethod
    async def _is_blocked(page: Page) -> bool:
        title = (await page.title()).lower()
        return any(s in title for s in ["captcha", "robot", "blocked", "sign in"])
