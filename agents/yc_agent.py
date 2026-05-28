"""
agents/yc_agent.py — Work at a Startup (WATS) job agent.

Everything runs on workatastartup.com — no ycombinator.com dependency.

Scrape flow:
  1. Navigate to WATS directory page to establish session/cookies
  2. Use page.request.get() with Accept: application/json to call Rails JSON API
  3. Parse companies + jobs from JSON response (bypasses React bot detection)
  4. Paginate via ?page= param in the API

Apply flow per job:
  1. Navigate to WATS job page
  2. Extract job description (for cover letter)
  3. Generate cover letter with LLM
  4. Click "Reach out to the team" button
  5. Wait for modal — clear pre-filled text — type cover letter
  6. Click "Send"
  7. Verify confirmation
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator

from playwright.async_api import Page

from agents.base_agent import ApplicationResult, ApplicationStatus, BaseAgent, JobListing
from browser_engine import BrowserEngine
from llm_brain import generate_cover_letter

logger = logging.getLogger(__name__)

_WATS_BASE = "https://www.workatastartup.com"
# Compact list layout — Rails serves JSON when Accept: application/json
_JOBS_URL = (
    "https://www.workatastartup.com/companies"
    "?demographic=any&hasEquity=any&hasSalary=any&industry=any"
    "&interviewProcess=any&jobType=fulltime&layout=list-compact"
    "&remote=only&role=eng&sortBy=created_desc&tab=any&usVisaNotRequired=any"
)
_JOBS_URL_JSON = (
    "https://www.workatastartup.com/companies.json"
    "?demographic=any&hasEquity=any&hasSalary=any&industry=any"
    "&interviewProcess=any&jobType=fulltime&layout=list-compact"
    "&remote=only&role=eng&sortBy=created_desc&tab=any&usVisaNotRequired=any"
)
# Job links look like /companies/{slug}/jobs/{id}-{title-slug}
_JOB_LINK_SEL = "a[href*='/companies/'][href*='/jobs/']"

_JSON_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}


class YCAgent(BaseAgent):
    """Applies to all WATS startup jobs via direct JSON API calls."""

    async def search_jobs(self) -> AsyncIterator[JobListing]:
        page = await self.engine.new_page()
        try:
            async for job in self._scrape_all_pages(page):
                yield job
        finally:
            await page.close()

    async def _scrape_all_pages(self, page: Page) -> AsyncIterator[JobListing]:
        # Capture ALL WATS requests (including cached) and JSON responses
        captured_api: list[dict] = []
        all_requests: list[str] = []
        console_msgs: list[str] = []

        async def _on_response(response) -> None:
            url = response.url
            if "workatastartup.com" not in url:
                return
            ct = response.headers.get("content-type", "")
            all_requests.append(f"{response.status} {ct[:50]} {url[:120]}")
            if "json" in ct and response.status == 200:
                try:
                    data = await response.json()
                    captured_api.append({"url": url, "data": data})
                    keys = list(data.keys()) if isinstance(data, dict) else f"list[{len(data)}]"
                    logger.info("[YC] JSON: %s → %s", url, keys)
                except Exception:
                    pass

        page.on("response", lambda r: asyncio.ensure_future(_on_response(r)))
        page.on("console", lambda m: console_msgs.append(f"[{m.type}] {m.text[:200]}"))
        page.on("pageerror", lambda e: console_msgs.append(f"[pageerror] {str(e)[:200]}"))

        logger.info("[YC] Loading WATS directory: %s", _JOBS_URL)
        try:
            await page.goto(_JOBS_URL, wait_until="networkidle", timeout=60_000)
            logger.info("[YC] networkidle reached")
        except Exception as exc:
            logger.warning("[YC] goto timed out or errored (%s), continuing anyway", exc)

        # Scroll to load all companies — WATS uses infinite scroll (20 per batch)
        prev_api_count = len(captured_api)
        for scroll_round in range(25):  # 25 × 20 = 500 companies max
            body_text = await page.evaluate("document.body.innerText")
            m = re.search(r"Showing (\d+) of (\d+)", body_text)
            if m:
                shown, total = int(m.group(1)), int(m.group(2))
                logger.info("[YC] Scroll %d: Showing %d of %d startups", scroll_round, shown, total)
                if shown >= total:
                    logger.info("[YC] All %d startups loaded", total)
                    break
            elif scroll_round > 0:
                break  # page never showed a count — stop

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await BrowserEngine.human_delay(2500, 3500)

            if len(captured_api) == prev_api_count:
                # No new batch arrived — try once more then give up
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await BrowserEngine.human_delay(2000, 3000)
                if len(captured_api) == prev_api_count:
                    logger.info("[YC] No new batch after scroll %d, stopping", scroll_round)
                    break
            prev_api_count = len(captured_api)

        page_title = await page.title()
        body_len = await page.evaluate("document.body.innerText.length")
        body_preview = await page.evaluate("document.body.innerText.slice(0, 300)")
        logger.info("[YC] Page: title=%s body_len=%d api_calls=%d requests=%d",
                    page_title, body_len, len(captured_api), len(all_requests))
        logger.info("[YC] Body preview: %r", body_preview)

        if all_requests:
            logger.info("[YC] WATS requests (%d):", len(all_requests))
            for r in all_requests[:20]:
                logger.info("[YC]   %s", r)

        if console_msgs:
            logger.info("[YC] Console (%d msgs):", len(console_msgs))
            for m in console_msgs[:20]:
                logger.info("[YC]   %s", m)

        if "sign in" in page_title.lower() or "account.ycombinator.com" in page.url:
            logger.error("[YC] Login wall — run: python setup_auth.py")
            return

        # If React made JSON API calls, use intercepted data
        if captured_api:
            seen_keys: set[str] = set()
            for entry in captured_api:
                for job in _parse_jobs_from_api(entry["data"]):
                    if job.job_key not in seen_keys:
                        seen_keys.add(job.job_key)
                        yield job
            logger.info("[YC] Yielded %d jobs from intercepted API calls", len(seen_keys))
            return

        # Fallback: DOM scrape (works if React rendered)
        if body_len > 200:
            async for job in self._dom_fallback(page, set()):
                yield job
            return

        logger.error("[YC] React did not render and no API calls captured. "
                     "The session in job_agent_profile may be expired. "
                     "Run: python setup_auth.py to re-login.")

    async def _dom_fallback(self, page: Page, seen_keys: set[str]) -> AsyncIterator[JobListing]:
        """Fallback: try to scrape job links from DOM (works if React mounted)."""
        try:
            await page.wait_for_selector(_JOB_LINK_SEL, timeout=15_000)
        except Exception:
            logger.warning("[YC] DOM fallback: no job links found")
            return

        links = await page.eval_on_selector_all(
            _JOB_LINK_SEL,
            "els => els.map(el => ({href: el.href, text: el.textContent.trim()}))",
        )
        for link in links:
            href: str = link.get("href", "")
            if "/jobs/" not in href:
                continue
            job = _make_job_from_link(href, link.get("text", ""))
            if job and job.job_key not in seen_keys:
                seen_keys.add(job.job_key)
                yield job


    # -------------------------------------------------------------------------
    # apply_to_job — fully overrides base
    # -------------------------------------------------------------------------

    async def apply_to_job(self, page: Page, job: JobListing) -> ApplicationResult:
        logger.info("[YC] Applying → %s @ %s | url=%s", job.title, job.company, job.url)

        try:
            # Navigate to the WATS job page — use networkidle to let React render
            try:
                await page.goto(job.url, wait_until="networkidle", timeout=30_000)
            except Exception:
                await page.goto(job.url, wait_until="domcontentloaded", timeout=30_000)
                await BrowserEngine.human_delay(3000, 4000)

            page_title = await page.title()
            body_text = await page.evaluate("document.body.innerText")
            body_len = len(body_text)
            logger.info("[YC] Job page: title=%r body_len=%d url=%s", page_title, body_len, page.url)

            # Check for login wall
            if "sign in" in page_title.lower() or "account.ycombinator.com" in page.url:
                logger.warning("[YC] Login wall on job page for %s", job.title)
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.NEEDS_HUMAN,
                    notes="WATS login wall",
                )

            # Extract description now that we're on the page
            description = await _extract_jd(page)
            if description:
                job.description = description

            # Generate cover letter
            cover_letter = await generate_cover_letter(
                company=job.company,
                role=job.title,
                job_description=job.description or job.title,
                profile=self.profile,
            )
            logger.info("[YC] Cover letter: %s…", cover_letter[:80])

            # Find and click "Reach out" / "Apply" button
            reach_btn = await _find_reach_out_button(page)
            if not reach_btn:
                # Log visible buttons to debug what's actually on the page
                try:
                    btns = await page.evaluate(
                        "() => [...document.querySelectorAll('button,a')].slice(0,20)"
                        ".map(e => e.textContent.trim().slice(0,50))"
                    )
                    logger.info("[YC] Buttons/links on page: %s", btns)
                except Exception:
                    pass
                logger.warning("[YC] No Reach out button for %s @ %s", job.title, job.company)
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.FAILED,
                    notes="Reach out button not found",
                )

            # Click — WATS opens a modal in the same tab (no new tab)
            await reach_btn.click()
            await BrowserEngine.human_delay(1500, 2500)

            # Fill and submit the modal
            return await self._fill_and_send(page, job, cover_letter)

        except Exception as exc:
            logger.error("[YC] Apply error for %s: %s", job.title, exc, exc_info=True)
            return ApplicationResult(job=job, status=ApplicationStatus.FAILED, notes=str(exc))

    async def _fill_and_send(
        self, page: Page, job: JobListing, cover_letter: str
    ) -> ApplicationResult:
        """Fill the WATS 'Reach out' modal textarea and click Send."""
        # Wait for the modal / textarea to appear
        try:
            await page.wait_for_selector("textarea", timeout=15_000)
        except Exception:
            logger.warning("[YC] Textarea never appeared for %s — url: %s", job.title, page.url)
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.NEEDS_HUMAN,
                notes=f"Textarea not found after clicking Reach out (url: {page.url})",
            )

        textarea = page.locator("textarea").first
        if not await textarea.is_visible(timeout=5000):
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.NEEDS_HUMAN,
                notes="Textarea not visible",
            )

        # Use fill() to reliably set the textarea value and trigger React's onChange
        await textarea.fill(cover_letter)
        await BrowserEngine.human_delay(500, 800)
        logger.info("[YC] Textarea filled for %s @ %s", job.title, job.company)

        # Find "Send" button
        send_btn = await _find_send_button(page)
        if not send_btn:
            logger.warning("[YC] Send button not found for %s", job.title)
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.NEEDS_HUMAN,
                notes="Send button not found",
            )

        await send_btn.scroll_into_view_if_needed()
        await BrowserEngine.human_delay(400, 800)
        # Wait up to 5s for Send button to become enabled after fill()
        try:
            await send_btn.wait_for(state="enabled", timeout=5000)
        except Exception:
            pass
        try:
            await send_btn.click(timeout=10_000)
        except Exception:
            # Fallback: dispatch click directly even if disabled
            await send_btn.dispatch_event("click")
        await BrowserEngine.human_delay(2000, 3000)

        # Check confirmation
        if await _check_sent(page):
            logger.info("[YC] ✓ Applied: %s @ %s", job.title, job.company)
            return ApplicationResult(job=job, status=ApplicationStatus.APPLIED)

        # No explicit confirmation but we clicked Send — count it
        logger.warning("[YC] Sent but no confirmation text for %s", job.title)
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.APPLIED,
            notes="Sent — no confirmation text detected",
        )

    async def _open_apply_flow(self, page: Page, job: JobListing) -> bool:
        return False  # Not used — apply_to_job fully overridden


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_href(href: str) -> str:
    return re.sub(r"[?#].*$", "", href.rstrip("/"))


def _make_job_from_link(href: str, link_text: str) -> JobListing | None:
    """Build a minimal JobListing from a WATS job link."""
    match = re.search(r"/companies/([^/]+)/jobs/([^/?#]+)", href)
    if not match:
        return None
    company_slug = match.group(1)
    job_id = match.group(2)
    job_key = f"{company_slug}__{job_id}"
    company = company_slug.replace("-", " ").title()
    title = link_text.strip() or job_id.replace("-", " ").title()
    full_url = href if href.startswith("http") else _WATS_BASE + href
    return JobListing(
        platform="yc",
        job_key=job_key,
        title=title,
        company=company,
        url=full_url,
        easy_apply=True,
        description="",
    )


async def _fetch_json(page: Page, html_url: str, json_url: str) -> dict | list | None:
    """Fetch WATS directory as JSON using the browser's cookies.

    Tries three strategies:
    1. .json format URL
    2. Accept: application/json on the HTML URL
    3. XMLHttpRequest header variant
    """
    for url, extra_headers in [
        (json_url, {}),
        (html_url, _JSON_HEADERS),
        (json_url, _JSON_HEADERS),
    ]:
        try:
            resp = await page.request.get(url, headers=extra_headers, timeout=30_000)
            ct = resp.headers.get("content-type", "")
            logger.info("[YC] API request: %s → status=%d ct=%s", url, resp.status, ct)
            if resp.status == 200 and "json" in ct:
                data = await resp.json()
                logger.info(
                    "[YC] JSON response: type=%s keys=%s",
                    type(data).__name__,
                    list(data.keys()) if isinstance(data, dict) else f"list[{len(data)}]",
                )
                return data
        except Exception as exc:
            logger.debug("[YC] API fetch failed (%s): %s", url, exc)

    return None


def _parse_jobs_from_api(data: dict | list) -> list[JobListing]:
    """Extract JobListings from WATS API JSON response.

    WATS returns either:
      {"companies": [...], ...}  — each company has "jobs": [...]
      [...] — array of companies directly
    Each job has: id, title, slug; parent company has: slug, name
    """
    jobs: list[JobListing] = []

    # Normalise to a list of company objects
    if isinstance(data, dict):
        companies = data.get("companies", data.get("results", []))
        if not isinstance(companies, list):
            # Might be a flat jobs list: {"jobs": [...]}
            raw_jobs = data.get("jobs", [])
            for j in raw_jobs:
                job = _job_from_api_obj(j, company_name=j.get("company", {}).get("name", ""),
                                        company_slug=j.get("company", {}).get("slug", ""))
                if job:
                    jobs.append(job)
            return jobs
    else:
        companies = data

    for company in companies:
        if not isinstance(company, dict):
            continue
        company_slug: str = company.get("slug", "")
        company_name: str = company.get("name", company_slug.replace("-", " ").title())
        raw_jobs = company.get("jobs", [])
        for j in raw_jobs:
            job = _job_from_api_obj(j, company_name=company_name, company_slug=company_slug)
            if job:
                jobs.append(job)

    return jobs


def _job_from_api_obj(j: dict, company_name: str, company_slug: str) -> JobListing | None:
    """Build a JobListing from a single WATS job API object."""
    job_id = str(j.get("id", ""))
    title = j.get("title", "") or j.get("role", "")

    if not job_id:
        return None

    # Only process engineering roles (the URL filter returns all jobs for matching companies)
    pretty_role = (j.get("pretty_role") or "").lower()
    if pretty_role and pretty_role not in ("engineering", ""):
        return None

    # show_path is the canonical job URL (e.g. /jobs/95323)
    show_path = j.get("show_path", "")
    if show_path:
        url = show_path if show_path.startswith("http") else _WATS_BASE + show_path
    else:
        url = f"{_WATS_BASE}/jobs/{job_id}"

    job_key = f"{company_slug}__{job_id}" if company_slug else job_id

    return JobListing(
        platform="yc",
        job_key=job_key,
        title=title or f"Job {job_id}",
        company=company_name,
        url=url,
        easy_apply=True,
        description=j.get("description", ""),
    )


def _has_next_page(data: dict | list, current_page: int) -> bool:
    """Check if the API response signals more pages exist."""
    if not isinstance(data, dict):
        return False
    # WATS may include: total_pages, page, next_page, has_more
    total_pages = data.get("total_pages") or data.get("totalPages")
    if total_pages is not None:
        return current_page < int(total_pages)
    has_more = data.get("has_more") or data.get("hasMore")
    if has_more is not None:
        return bool(has_more)
    next_page = data.get("next_page") or data.get("nextPage")
    if next_page is not None:
        return next_page is not None
    # If no pagination metadata, assume single page
    return False


async def _extract_jd(page: Page) -> str:
    """Extract the job description text from a WATS job page."""
    selectors = [
        ".prose", "[class*='prose']", "[class*='jobDescription']",
        "[class*='description']", "article", "main",
    ]
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if await el.count() > 0:
                text = (await el.inner_text()).strip()
                if len(text) > 100:
                    return text[:5000]
        except Exception:
            pass
    try:
        text = await page.evaluate(
            "() => { const m = document.querySelector('main') || document.body; return m.innerText; }"
        )
        return (text or "").strip()[:5000]
    except Exception:
        return ""


async def _find_reach_out_button(page: Page):
    """Find the 'Reach out' or 'Apply' button on a WATS job page."""
    # WATS uses "Reach out to the team at X" — match on "Reach out" prefix
    for pattern in [
        r"Reach out",
        r"Apply",
    ]:
        for role in ("button", "link"):
            try:
                btn = page.get_by_role(role, name=re.compile(pattern, re.IGNORECASE))
                if await btn.count() > 0 and await btn.first.is_visible(timeout=2000):
                    return btn.first
            except Exception:
                pass

    # Fallback: any visible element containing "Reach out" or "Apply"
    try:
        btn = page.locator(
            "button:has-text('Reach out'), a:has-text('Reach out'), "
            "button:has-text('Apply'), a:has-text('Apply')"
        ).first
        if await btn.count() > 0 and await btn.is_visible(timeout=2000):
            return btn
    except Exception:
        pass

    return None


async def _find_send_button(page: Page):
    """Find the Send button inside the WATS apply modal."""
    # Exact "Send" button
    for pattern in [r"^Send$", r"^Submit"]:
        try:
            btn = page.get_by_role("button", name=re.compile(pattern, re.IGNORECASE))
            if await btn.count() > 0 and await btn.first.is_visible(timeout=2000):
                return btn.first
        except Exception:
            pass
    # Fallback: any submit-type button that isn't "Close"
    try:
        buttons = await page.locator("button").all()
        for btn in buttons:
            txt = (await btn.text_content() or "").strip().lower()
            if txt in ("send", "submit", "send application") and await btn.is_visible(timeout=1000):
                return btn
    except Exception:
        pass
    return None


async def _check_sent(page: Page) -> bool:
    """Check if WATS application was sent.

    WATS closes the modal after sending — no explicit confirmation text.
    We check: modal/textarea is gone (primary) or confirmation keywords (fallback).
    """
    try:
        body = (await page.evaluate("document.body.innerText")).lower()
        # Explicit confirmation text
        if any(k in body for k in [
            "message sent", "application sent", "sent!", "we've received",
            "thank you", "your message", "we'll be in touch",
        ]):
            return True
        # WATS typically closes the modal — if textarea is gone, assume sent
        textarea_count = await page.locator("textarea:visible").count()
        return textarea_count == 0
    except Exception:
        return False
