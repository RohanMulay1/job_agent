"""
agents/wellfound_agent.py — Wellfound (AngelList Talent) native apply agent.

Apply flow (confirmed from live debug):
  1. Navigate to job page  (/jobs/{id}-{slug})
  2. Click "Apply now" button
  3. [role='dialog']:not([aria-label*='Beacon']):not([aria-label*='Scout']) opens with:
       - Left panel: job details + location/visa info
       - Right panel: textarea[name='userNote'] + "Send application" button
  4. Fill the textarea with a cover letter
  5. Click "Send application"
  6. Detect modal closure = success

Skip conditions:
  - "Apply on company site" button → external ATS, skip
  - Location restriction text in dialog → skip
  - Dialog does not open → skip
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from browser_engine import BrowserEngine
from config import settings
from llm_brain import generate_cover_letter
from profile_schema import CandidateProfile

from .base_agent import ApplicationResult, ApplicationStatus, BaseAgent, JobListing

logger = logging.getLogger(__name__)

_BASE = "https://wellfound.com"

_LOCATION_BLOCK_PHRASES = [
    "not accepting applications from your current location",
    "timezone or relocation constraints",
]


class WellfoundAgent(BaseAgent):
    def __init__(self, engine: BrowserEngine, profile: CandidateProfile) -> None:
        super().__init__(engine, profile)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    # Role slugs + location slugs that map to confirmed Wellfound URL patterns
    _ROLE_SLUGS = [
        "machine-learning-engineer",
        "software-architect",
        "ai-engineer",
        "software-engineer",
        "backend-engineer",
    ]
    _LOCATION_SLUGS = [
        "remote",
        "bangalore",
        "mumbai",
        "delhi",
        "hyderabad",
    ]

    async def search_jobs(self) -> AsyncIterator[JobListing]:
        page = await self.engine.new_page()
        yielded_keys: set[str] = set()
        try:
            for role_slug in self._ROLE_SLUGS:
                for loc_slug in self._LOCATION_SLUGS:
                    async for job in self._search_role_url(page, role_slug, loc_slug):
                        if job.job_key not in yielded_keys:
                            yielded_keys.add(job.job_key)
                            yield job
        finally:
            await page.close()

    async def _search_role_url(self, page: Page, role_slug: str, loc_slug: str) -> AsyncIterator[JobListing]:
        url = f"{_BASE}/role/l/{role_slug}/{loc_slug}"
        logger.info("[Wellfound] Searching: %s", url)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except PlaywrightTimeout:
            logger.warning("[Wellfound] Page load timeout: %s", url)
            return

        await BrowserEngine.human_delay(2500, 3500)

        if await self._is_blocked(page):
            logger.error("[Wellfound] CAPTCHA detected — waiting 35s for manual solve...")
            for remaining in range(35, 0, -5):
                print(f"  Solve the slider CAPTCHA in the browser window — {remaining}s remaining...")
                await asyncio.sleep(5)
            if await self._is_blocked(page):
                logger.error("[Wellfound] Still blocked after wait — skipping %s", url)
                return

        try:
            await page.wait_for_selector("a[href^='/jobs/']", timeout=12_000)
        except PlaywrightTimeout:
            logger.info("[Wellfound] No job cards at %s", url)
            return

        for _page_num in range(settings.max_pages_per_search):
            jobs = await self._extract_job_cards(page)
            if not jobs:
                break
            for job in jobs:
                yield job
            if not await self._scroll_for_more(page):
                break

    async def _extract_job_cards(self, page: Page) -> list[JobListing]:
        jobs: list[JobListing] = []
        seen: set[str] = set()

        links = await page.locator("a[href^='/jobs/']").all()
        for link in links:
            try:
                href = (await link.get_attribute("href") or "").strip()
                if not href or not re.match(r"^/jobs/\d+", href):
                    continue

                job_key = href.split("/jobs/")[-1].split("?")[0].strip("/")
                if job_key in seen:
                    continue
                seen.add(job_key)

                title = (await link.text_content() or "").strip()
                if not title:
                    continue

                # Company: nearest ancestor div with a /company/ or /startups/ link
                company = ""
                try:
                    parent = page.locator(f"a[href='{href}']").locator(
                        "xpath=ancestor::div[.//a[contains(@href,'/company/') or contains(@href,'/startups/')]][1]"
                    )
                    co_link = parent.locator(
                        "a[href*='/company/'], a[href*='/startups/']"
                    ).first
                    if await co_link.count():
                        company = (await co_link.text_content() or "").strip()
                except Exception:
                    pass

                full_url = _BASE + href
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
    # Apply — override base class entirely (no LLM form loop needed)
    # ------------------------------------------------------------------

    async def apply_to_job(self, page: Page, job: JobListing) -> ApplicationResult:
        logger.info("[Wellfound] Applying → %s @ %s | url=%s", job.title, job.company, job.url)

        # ── Navigate ─────────────────────────────────────────────────────
        try:
            await page.goto(job.url, wait_until="domcontentloaded", timeout=30_000)
            await BrowserEngine.human_delay(2000, 3000)
        except PlaywrightTimeout:
            return ApplicationResult(job=job, status=ApplicationStatus.FAILED, notes="Page load timeout")

        if await self._is_blocked(page):
            return ApplicationResult(job=job, status=ApplicationStatus.FAILED, notes="DataDome block on job page")

        if self._is_external_redirect(page.url):
            return ApplicationResult(job=job, status=ApplicationStatus.SKIPPED_EXTERNAL, notes="Redirected to ATS")

        # ── Grab job description ─────────────────────────────────────────
        try:
            desc_el = page.locator(
                "[class*='description'], .prose, [class*='jobDescription'], [class*='content']"
            ).first
            job.description = (await desc_el.text_content() or "")[:3000]
        except Exception:
            pass

        # ── Skip external-only jobs ──────────────────────────────────────
        external_btn = page.locator(
            "a:has-text('Apply on company site'), button:has-text('Apply on company site')"
        )
        if await external_btn.count():
            logger.info("[Wellfound] External-only — skipping %s", job.title)
            return ApplicationResult(job=job, status=ApplicationStatus.SKIPPED_EXTERNAL, notes="External apply only")

        # ── Find and click Apply button ───────────────────────────────────
        apply_btn = page.locator(
            "button:has-text('Apply now'), button:has-text('Apply')"
        ).first
        if not await apply_btn.count():
            logger.info("[Wellfound] No Apply button for %s", job.title)
            return ApplicationResult(job=job, status=ApplicationStatus.SKIPPED_EXTERNAL, notes="No Apply button")

        try:
            await BrowserEngine.human_delay(400, 800)
            await apply_btn.click(force=True, timeout=8000)
        except Exception as exc:
            return ApplicationResult(job=job, status=ApplicationStatus.FAILED, notes=f"Apply click failed: {exc}")

        # ── Wait for dialog ───────────────────────────────────────────────
        try:
            await page.wait_for_selector("[role='dialog']:not([aria-label*='Beacon']):not([aria-label*='Scout'])", timeout=12_000)
        except PlaywrightTimeout:
            return ApplicationResult(job=job, status=ApplicationStatus.FAILED, notes="Apply dialog did not open")

        await BrowserEngine.human_delay(800, 1200)

        # ── Check for location block ──────────────────────────────────────
        dialog_text = ""
        try:
            dialog = page.locator("[role='dialog']:not([aria-label*='Beacon']):not([aria-label*='Scout'])").first
            dialog_text = (await dialog.text_content() or "").lower()
        except Exception:
            pass

        if any(phrase in dialog_text for phrase in _LOCATION_BLOCK_PHRASES):
            logger.info("[Wellfound] Location-blocked — skipping %s", job.title)
            # Close dialog
            cancel = page.locator("[role='dialog']:not([aria-label*='Beacon']):not([aria-label*='Scout']) button:has-text('Cancel')").first
            if await cancel.count():
                await cancel.click()
            return ApplicationResult(job=job, status=ApplicationStatus.SKIPPED_EXTERNAL, notes="Location restriction")

        # ── Find textarea ─────────────────────────────────────────────────
        _dialog_sel = "[role='dialog']:not([aria-label*='Beacon']):not([aria-label*='Scout'])"
        textarea = page.locator(f"{_dialog_sel} textarea[name='userNote']").first
        if not await textarea.count():
            textarea = page.locator(f"{_dialog_sel} textarea").first
        if not await textarea.count():
            return ApplicationResult(job=job, status=ApplicationStatus.NEEDS_HUMAN, notes="No textarea found in dialog")

        # ── Generate and fill cover letter ────────────────────────────────
        cover_letter = await generate_cover_letter(
            company=job.company or "the company",
            role=job.title,
            job_description=job.description or job.title,
            profile=self.profile,
        )
        logger.info("[Wellfound] Cover letter: %s…", cover_letter[:80])

        # Scroll to textarea + force-click to focus, then fill
        try:
            await textarea.scroll_into_view_if_needed(timeout=5000)
            await textarea.click(force=True, timeout=5000)
        except Exception:
            pass
        try:
            await textarea.fill(cover_letter, timeout=10_000)
        except Exception:
            # Last resort: type via keyboard
            await textarea.dispatch_event("click")
            await page.keyboard.type(cover_letter, delay=30)
        await BrowserEngine.human_delay(500, 800)

        # ── Click Send application ────────────────────────────────────────
        send_btn = page.locator(
            "[role='dialog']:not([aria-label*='Beacon']):not([aria-label*='Scout']) button:has-text('Send application')"
        ).first

        if not await send_btn.count():
            return ApplicationResult(job=job, status=ApplicationStatus.NEEDS_HUMAN, notes="Send button not found")

        try:
            await send_btn.wait_for(state="enabled", timeout=5000)
            await send_btn.click(timeout=10_000)
        except Exception:
            await send_btn.dispatch_event("click")

        await BrowserEngine.human_delay(2000, 3000)

        # ── Detect success — dialog should close ─────────────────────────
        success = await self._check_sent(page)
        if success:
            logger.info("[Wellfound] Application sent: %s @ %s", job.title, job.company)
            return ApplicationResult(job=job, status=ApplicationStatus.APPLIED)

        logger.warning("[Wellfound] Sent but could not confirm: %s", job.title)
        return ApplicationResult(job=job, status=ApplicationStatus.APPLIED, notes="Sent (unconfirmed)")

    async def _check_sent(self, page: Page) -> bool:
        """Success = dialog closed, or confirmation text visible."""
        try:
            # Dialog gone
            dialog_count = await page.locator("[role='dialog']:not([aria-label*='Beacon']):not([aria-label*='Scout']):visible").count()
            if dialog_count == 0:
                return True
            # Confirmation text inside dialog
            body = (await page.locator("[role='dialog']:not([aria-label*='Beacon']):not([aria-label*='Scout'])").first.text_content() or "").lower()
            return any(w in body for w in ("application sent", "applied", "thank you", "we'll be in touch"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Required abstract stub (apply_to_job overrides the full flow)
    # ------------------------------------------------------------------

    async def _open_apply_flow(self, page: Page, job: JobListing) -> bool:
        return False  # never called — apply_to_job is overridden directly

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _scroll_for_more(self, page: Page) -> bool:
        prev = await page.locator("a[href^='/jobs/']").count()
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await BrowserEngine.human_delay(2500, 3500)
        return await page.locator("a[href^='/jobs/']").count() > prev

    @staticmethod
    async def _is_blocked(page: Page) -> bool:
        title = (await page.title()).lower()
        if any(s in title for s in ["captcha", "robot", "blocked", "verification"]):
            return True
        try:
            srcs = await page.evaluate(
                "Array.from(document.querySelectorAll('iframe')).map(f => f.src)"
            )
            return any("captcha-delivery.com" in s for s in srcs)
        except Exception:
            return False
