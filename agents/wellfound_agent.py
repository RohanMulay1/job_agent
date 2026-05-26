"""
agents/wellfound_agent.py — Wellfound (AngelList Talent) native apply agent.

Search strategy:
  - URL: /jobs?q={title}  (Wellfound search with React hydration wait)
  - Job cards: div.styles_joblistingContainer__ZxslN containers
  - Title link: a.styles_joblistingTitleAnchor__DFCkK with href /jobs/{id}-{slug}
  - Company: a.styles_company__w5lec
  - Infinite scroll for pagination

Apply flow:
  - Navigate to job URL, find Apply button, force click
  - Wait for apply modal ([role='dialog']) — then LLM form loop via BaseAgent
  - DataDome blocking: detected via captcha-delivery.com iframe in page source
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

_WELLFOUND_BASE = "https://wellfound.com"
_TITLE_LINK_SEL = "a.styles_joblistingTitleAnchor__DFCkK"
_CONTAINER_SEL = "div.styles_joblistingContainer__ZxslN"


class WellfoundAgent(BaseAgent):
    def __init__(self, engine: BrowserEngine, profile: CandidateProfile) -> None:
        super().__init__(engine, profile)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_jobs(self) -> AsyncIterator[JobListing]:
        page = await self.engine.new_page()
        yielded_keys: set[str] = set()
        try:
            for title in settings.target_titles:
                async for job in self._search_query(page, title):
                    if job.job_key not in yielded_keys:
                        yielded_keys.add(job.job_key)
                        yield job
        finally:
            await page.close()

    async def _search_query(self, page: Page, title: str) -> AsyncIterator[JobListing]:
        url = f"{_WELLFOUND_BASE}/jobs?{urllib.parse.urlencode({'q': title})}"
        logger.info("[Wellfound] Searching: %s", title)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except PlaywrightTimeout:
            logger.warning("[Wellfound] Page load timeout: %s", url)
            return

        # Wellfound is a Next.js app — wait explicitly for card anchor to render
        try:
            await page.wait_for_selector(_TITLE_LINK_SEL, timeout=15_000)
        except PlaywrightTimeout:
            logger.info("[Wellfound] No job cards found for '%s' after 15s", title)
            return

        if await self._is_blocked(page):
            logger.error("[Wellfound] Bot detection triggered on search page")
            return

        for _page_num in range(settings.max_pages_per_search):
            jobs = await self._extract_job_cards(page)
            if not jobs:
                logger.info("[Wellfound] No results for '%s'", title)
                return

            for job in jobs:
                yield job

            # Wellfound uses infinite scroll
            if not await self._scroll_for_more(page):
                break

    async def _extract_job_cards(self, page: Page) -> list[JobListing]:
        jobs: list[JobListing] = []
        seen_keys: set[str] = set()

        try:
            title_links = await page.locator(_TITLE_LINK_SEL).all()
        except Exception:
            return jobs

        for link in title_links:
            try:
                href = await link.get_attribute("href") or ""
                if not href or "/jobs/" not in href:
                    continue

                full_url = href if href.startswith("http") else _WELLFOUND_BASE + href
                # href looks like /jobs/4261027-ai-agent-engineer
                job_key = href.split("/jobs/")[-1].split("?")[0].strip("/")

                if not job_key or job_key in seen_keys:
                    continue
                seen_keys.add(job_key)

                title = (await link.text_content() or "").strip()

                # Find the containing card, then grab the company link within it
                container = page.locator(_CONTAINER_SEL).filter(has=page.locator(f"a[href='{href}']"))
                company_el = container.locator("a.styles_company__w5lec").first
                company = ""
                if await company_el.count():
                    company = (await company_el.text_content() or "").strip()

                if title:
                    jobs.append(JobListing(
                        platform="wellfound",
                        job_key=job_key,
                        title=title,
                        company=company,
                        url=full_url,
                        easy_apply=True,
                    ))
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
            await BrowserEngine.human_delay(2000, 3500)
        except PlaywrightTimeout:
            logger.warning("[Wellfound] Timeout loading job: %s", job.url)
            return False

        if await self._is_blocked(page):
            return False

        if self._is_external_redirect(page.url):
            return False

        # Grab description
        try:
            desc_el = page.locator("[class*='description'], .prose, [class*='jobDescription']").first
            job.description = (await desc_el.text_content() or "")[:3000]
        except Exception:
            pass

        # Skip jobs that only offer external apply
        external_btn = page.locator(
            "a:has-text('Apply on company site'), button:has-text('Apply on company site')"
        )
        if await external_btn.count():
            logger.info("[Wellfound] External-only apply — skipping %s", job.title)
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
            await BrowserEngine.human_delay(400, 900)
            # force=True bypasses visibility checks — button may be in a fixed header
            await apply_btn.click(force=True, timeout=8000)

            # Wait for the apply modal / inline form
            try:
                await page.wait_for_selector(
                    "[role='dialog'], [class*='ApplicationModal'], [class*='applyModal']",
                    timeout=12_000,
                )
                return True
            except PlaywrightTimeout:
                pass

            # Fallback: maybe an inline form appeared on the page
            await BrowserEngine.human_delay(1500, 2500)
            if await page.locator("form").count():
                return True

            logger.warning("[Wellfound] Apply modal did not appear for %s", job.title)
            return False

        except Exception as exc:
            logger.warning("[Wellfound] Apply click failed for %s: %s", job.title, exc)
            return False

    # ------------------------------------------------------------------
    # Infinite scroll helper
    # ------------------------------------------------------------------

    async def _scroll_for_more(self, page: Page) -> bool:
        prev_count = await page.locator(_TITLE_LINK_SEL).count()
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await BrowserEngine.human_delay(2500, 4000)
        new_count = await page.locator(_TITLE_LINK_SEL).count()
        return new_count > prev_count

    # ------------------------------------------------------------------
    # Block detection
    # ------------------------------------------------------------------

    @staticmethod
    async def _is_blocked(page: Page) -> bool:
        title = (await page.title()).lower()
        if any(s in title for s in ["captcha", "robot", "blocked"]):
            return True
        # DataDome CAPTCHA challenge embeds an iframe pointing at captcha-delivery.com
        try:
            frame_sources = await page.evaluate(
                "Array.from(document.querySelectorAll('iframe')).map(f => f.src)"
            )
            return any("captcha-delivery.com" in src for src in frame_sources)
        except Exception:
            return False
