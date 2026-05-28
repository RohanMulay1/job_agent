"""
agents/naukri_agent.py — Naukri.com Easy Apply agent.

Search strategy:
  - URL pattern: /jobs-in-{location}?k={title}&jobAge=7
  - Easy Apply filter applied via query param (easyApply=true) + badge check
  - Pagination via &pageNo= parameter
  - Job keys extracted from data-job-id attributes
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import AsyncIterator

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from browser_engine import BrowserEngine
from config import settings
from llm_brain import parse_form_and_decide_actions
from profile_schema import CandidateProfile

from .base_agent import ApplicationResult, ApplicationStatus, BaseAgent, JobListing

logger = logging.getLogger(__name__)

_NAUKRI_BASE = "https://www.naukri.com"


class NaukriAgent(BaseAgent):
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
        for page_num in range(1, settings.max_pages_per_search + 1):
            url = self._build_search_url(title, location, page_num)
            logger.info("[Naukri] Searching: %s in %s (page %d)", title, location, page_num)

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                # Wait for job cards to render (Next.js hydration needs time)
                try:
                    await page.wait_for_selector("div[data-job-id]", timeout=10_000)
                except PlaywrightTimeout:
                    pass  # No cards on this page — handled below
                await BrowserEngine.human_delay(800, 1500)
            except PlaywrightTimeout:
                logger.warning("[Naukri] Page load timeout: %s", url)
                return

            if await self._is_blocked(page):
                logger.error("[Naukri] Bot detection triggered")
                return

            jobs = await self._extract_job_cards(page)
            if not jobs:
                logger.info("[Naukri] No more results for '%s' in '%s'", title, location)
                return

            for job in jobs:
                yield job

    def _build_search_url(self, title: str, location: str, page_num: int = 1) -> str:
        title_slug = title.lower().replace(" ", "-").replace("/", "-")
        loc_lower = location.lower()

        # Naukri calls remote work "work-from-home", not "remote"
        if "remote" in loc_lower:
            path = f"/{title_slug}-work-from-home-jobs"
        else:
            location_slug = loc_lower.replace(" ", "-")
            path = f"/{title_slug}-jobs-in-{location_slug}"

        params: dict[str, str] = {
            "jobAge": "7",
            "experience": "0,3",
            "applyType": "1",   # 1 = Apply on Naukri (native), filters out external ATS
        }
        if page_num > 1:
            params["pageNo"] = str(page_num)

        return f"{_NAUKRI_BASE}{path}?{urllib.parse.urlencode(params)}"

    async def _extract_job_cards(self, page: Page) -> list[JobListing]:
        jobs: list[JobListing] = []

        try:
            cards = await page.locator("div[data-job-id]").all()
        except Exception:
            return jobs

        for card in cards:
            try:
                job_key = await card.get_attribute("data-job-id") or ""

                title_el = card.locator("a.title").first
                title = (await title_el.text_content() or "").strip()
                job_url = await title_el.get_attribute("href") or ""

                company_el = card.locator("a.comp-name").first
                company = (await company_el.text_content() or "").strip()

                # Easy Apply badge doesn't reliably appear in search results;
                # mark all as eligible and let _open_apply_flow verify on the job page.
                easy_apply = True

                if not job_key:
                    job_key = job_url.split("/")[-1].split("?")[0] or title[:30]

                if title:
                    jobs.append(
                        JobListing(
                            platform="naukri",
                            job_key=job_key,
                            title=title,
                            company=company,
                            url=job_url,
                            easy_apply=easy_apply,
                        )
                    )
            except Exception as exc:
                logger.debug("[Naukri] Card parse error: %s", exc)

        return jobs

    # ------------------------------------------------------------------
    # Apply flow
    # ------------------------------------------------------------------

    async def apply_to_job(self, page: Page, job: JobListing) -> ApplicationResult:
        """Override to handle Naukri's 1-click (navigation-based) apply."""
        logger.info("[naukri] Applying → %s @ %s", job.title, job.company)

        try:
            opened = await self._open_apply_flow(page, job)
        except Exception as exc:
            logger.error("Failed to open apply flow for %s: %s", job.title, exc)
            return ApplicationResult(job=job, status=ApplicationStatus.FAILED, notes=str(exc))

        if not opened:
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.SKIPPED_EXTERNAL,
                notes="No native apply button or external redirect",
            )

        # Naukri 1-click: browser navigated to /myapply/showAcp
        # multiApplyResp value of 202 = already submitted
        if "/myapply/" in page.url:
            if "202" in page.url:
                logger.info("[naukri] 1-click applied: %s @ %s", job.title, job.company)
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.APPLIED,
                    notes=f"Naukri 1-click apply confirmed (multiApplyResp 202): {page.url}",
                )
            # ACP page loaded but no 202 — may have a multi-step form; fall through to LLM loop

        if self._is_external_redirect(page.url):
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.SKIPPED_EXTERNAL,
                notes=f"External ATS redirect: {page.url}",
            )

        # LLM-driven form fill loop for multi-step ACP forms
        for step in range(15):
            await BrowserEngine.human_delay(800, 1800)

            if self._is_external_redirect(page.url):
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.SKIPPED_EXTERNAL,
                    notes=f"External ATS at step {step}: {page.url}",
                )

            snapshot = await self._extract_dom_snapshot(page)
            page_hint = f"Step {step + 1} — {job.title} at {job.company}"
            form_response = await parse_form_and_decide_actions(snapshot, self.profile, page_hint)

            if form_response.needs_human:
                logger.warning("Human needed at step %d: %s", step, form_response.notes)
                return ApplicationResult(
                    job=job, status=ApplicationStatus.NEEDS_HUMAN, notes=form_response.notes
                )

            if not form_response.actions:
                logger.warning("No LLM actions at step %d — aborting", step)
                return ApplicationResult(
                    job=job, status=ApplicationStatus.FAILED, notes=f"No actions at step {step}"
                )

            await self._execute_actions(page, form_response)

            if form_response.is_complete:
                logger.info("[naukri] Application submitted: %s", job.title)
                return ApplicationResult(job=job, status=ApplicationStatus.APPLIED)

        return ApplicationResult(
            job=job, status=ApplicationStatus.FAILED, notes="Exceeded max form steps (15)"
        )

    async def _open_apply_flow(self, page: Page, job: JobListing) -> bool:
        if not job.url:
            return False

        try:
            await page.goto(job.url, wait_until="domcontentloaded", timeout=30_000)
            await BrowserEngine.human_delay(1500, 2500)
        except PlaywrightTimeout:
            logger.warning("[Naukri] Timeout loading job: %s", job.url)
            return False

        if self._is_external_redirect(page.url):
            return False

        if "login" in page.url.lower() or "signup" in page.url.lower():
            logger.warning("[Naukri] Session expired — login redirect for %s", job.title)
            return False

        # Grab description
        try:
            desc_el = page.locator(
                "section.job-desc, div[class*='job-description'], .dang-inner-html"
            ).first
            job.description = (await desc_el.text_content() or "")[:3000]
        except Exception:
            pass

        # Scroll down a bit to trigger Naukri's sticky header (which has a visible apply button)
        await page.evaluate("window.scrollBy(0, 400)")
        await BrowserEngine.human_delay(800, 1200)

        # Find a VISIBLE apply button — Naukri has multiple apply buttons; only some are visible.
        # Prefer Easy Apply, then any visible Apply button.
        apply_btn = await self._find_visible_apply_btn(page)

        if apply_btn is None:
            logger.info("[Naukri] No visible Apply button found for %s", job.title)
            return False

        await BrowserEngine.human_delay(300, 700)

        # Click the visible button; also try JS as final fallback
        clicked = False
        for attempt in ("normal", "js"):
            try:
                if attempt == "normal":
                    await apply_btn.click(timeout=5000)
                else:
                    handle = await apply_btn.element_handle()
                    if handle:
                        await page.evaluate("el => el.click()", handle)
                    else:
                        continue
                clicked = True
                break
            except Exception as exc:
                logger.debug("[Naukri] Apply click attempt '%s' failed: %s", attempt, exc)

        if not clicked:
            # Last resort: click ANY apply-button via pure JS on the page
            try:
                await page.evaluate(
                    "() => { const btn = document.querySelector('button.apply-button, "
                    "a.apply-button, #apply-button'); if(btn) btn.click(); }"
                )
                clicked = True
                logger.debug("[Naukri] Fell back to page-level JS click for %s", job.title)
            except Exception as exc:
                logger.warning("[Naukri] All apply click attempts failed for %s: %s", job.title, exc)
                return False

        await BrowserEngine.human_delay(800, 1200)

        # Check for login modal — session expired means nothing will work
        try:
            login_visible = await page.locator(
                "[class*='login-layer'], [class*='loginModal'], "
                "input[placeholder*='Email'], input[type='email']"
            ).first.is_visible(timeout=2000)
            if login_visible:
                logger.warning("[Naukri] Login modal appeared — session expired for %s", job.title)
                return False
        except Exception:
            pass

        # Naukri 1-click navigates to /myapply/
        try:
            await page.wait_for_url("**/myapply/**", timeout=6_000)
            return True
        except PlaywrightTimeout:
            pass

        # Some jobs show a modal/drawer
        try:
            await page.wait_for_selector(
                "[role='dialog'], .apply-modal, form[class*='apply']",
                timeout=5_000,
            )
            return True
        except PlaywrightTimeout:
            logger.warning("[Naukri] No navigation or modal after Apply click for %s", job.title)
            return False

    @staticmethod
    async def _find_visible_apply_btn(page: Page):
        """Return the first visible Apply button. Prefers Easy Apply."""
        # Try Easy Apply first
        for sel in [
            "button:has-text('Easy Apply')",
            "a:has-text('Easy Apply')",
        ]:
            try:
                el = page.locator(sel).first
                if await el.count() and await el.is_visible(timeout=2000):
                    return el
            except Exception:
                pass

        # Find ALL apply buttons and return the first visible one
        candidates = page.locator(
            "button:has-text('Apply'), a.apply-button, button.apply-button, "
            "button[id*='apply'], a[id*='apply']"
        )
        count = await candidates.count()
        for i in range(count):
            try:
                btn = candidates.nth(i)
                href = await btn.get_attribute("href") or ""
                # Skip external-redirect buttons
                if "greenhouse" in href or "lever.co" in href or "ashbyhq" in href:
                    continue
                if await btn.is_visible(timeout=1000):
                    return btn
            except Exception:
                pass
        return None

    @staticmethod
    async def _is_blocked(page: Page) -> bool:
        url = page.url.lower()
        if "login" in url or "signup" in url:
            return True
        title = (await page.title()).lower()
        return any(s in title for s in ["captcha", "robot", "blocked", "verify you're human", "login", "sign in"])
