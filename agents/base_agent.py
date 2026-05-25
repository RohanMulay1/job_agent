"""
agents/base_agent.py — Abstract base for all platform agents.

Subclasses implement:
  search_jobs()   → yields JobListing objects
  apply_to_job()  → returns ApplicationResult

The base class owns:
  _extract_dom_snapshot()   — strips page to labels+inputs (<4 KB)
  _is_external_redirect()   — detects ATS redirects (Workday, Greenhouse, etc.)
  _execute_actions()        — runs LLM-generated action list on the live page
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from playwright.async_api import Page

from browser_engine import BrowserEngine
from llm_brain import FormResponse, parse_form_and_decide_actions
from profile_schema import CandidateProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known external ATS domains — skip any job that lands here
# ---------------------------------------------------------------------------
_EXTERNAL_ATS_PATTERNS = re.compile(
    r"(workday\.com|greenhouse\.io|lever\.co|ashbyhq\.com|"
    r"smartrecruiters\.com|taleo\.net|icims\.com|jobvite\.com|"
    r"bamboohr\.com|recruitee\.com|myworkdayjobs\.com|"
    r"apply\.workable\.com|jobs\.lever\.co|boards\.greenhouse\.io)",
    re.IGNORECASE,
)

# Tags whose content we keep when building the DOM snapshot
_KEEP_TAGS = {"form", "label", "input", "select", "textarea", "button", "option", "legend", "fieldset"}

# Attributes to keep on input/select elements (all others stripped)
_KEEP_ATTRS = {"type", "name", "id", "placeholder", "value", "required", "aria-label", "aria-labelledby", "for", "checked", "selected"}

# Maximum snapshot size in characters passed to the LLM
_MAX_SNAPSHOT_CHARS = 4000

# Maximum form-fill loop iterations per application (safety valve)
_MAX_FORM_STEPS = 15


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class JobListing:
    platform: str
    job_key: str       # Unique ID on that platform (used for dedup)
    title: str
    company: str
    url: str
    easy_apply: bool = True
    description: str = ""


class ApplicationStatus(str, Enum):
    APPLIED = "applied"
    SKIPPED_EXTERNAL = "skipped_external"
    SKIPPED_FIT = "skipped_fit"
    SKIPPED_SEEN = "skipped_seen"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"


@dataclass
class ApplicationResult:
    job: JobListing
    status: ApplicationStatus
    notes: str = ""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    def __init__(self, engine: BrowserEngine, profile: CandidateProfile) -> None:
        self.engine = engine
        self.profile = profile

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    async def search_jobs(self) -> AsyncIterator[JobListing]:
        """Yield JobListing objects for the configured search queries."""
        ...  # pragma: no cover

    @abstractmethod
    async def _open_apply_flow(self, page: Page, job: JobListing) -> bool:
        """
        Navigate to the job listing and trigger the Easy Apply button.
        Returns True if the apply modal/page is now open, False otherwise.
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # Shared apply orchestration
    # ------------------------------------------------------------------

    async def apply_to_job(self, page: Page, job: JobListing) -> ApplicationResult:
        """
        Full application flow driven by the LLM brain.
        Subclasses open the apply modal; this method handles the form loop.
        """
        logger.info("[%s] Applying → %s @ %s", job.platform, job.title, job.company)

        try:
            opened = await self._open_apply_flow(page, job)
        except Exception as exc:
            logger.error("Failed to open apply flow for %s: %s", job.title, exc)
            return ApplicationResult(job=job, status=ApplicationStatus.FAILED, notes=str(exc))

        if not opened:
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.SKIPPED_EXTERNAL,
                notes="Apply button not found or external redirect detected before modal open",
            )

        # Safety check — did we land on an external ATS?
        if self._is_external_redirect(page.url):
            logger.warning("External ATS redirect detected: %s", page.url)
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.SKIPPED_EXTERNAL,
                notes=f"Redirected to external ATS: {page.url}",
            )

        # LLM-driven form fill loop
        for step in range(_MAX_FORM_STEPS):
            await BrowserEngine.human_delay(800, 1800)

            # Re-check for external redirect on every step
            if self._is_external_redirect(page.url):
                logger.warning("Mid-flow external redirect at step %d: %s", step, page.url)
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.SKIPPED_EXTERNAL,
                    notes=f"External ATS redirect at step {step}: {page.url}",
                )

            snapshot = await self._extract_dom_snapshot(page)
            page_hint = f"Step {step + 1} — {job.title} at {job.company}"
            form_response: FormResponse = await parse_form_and_decide_actions(
                snapshot, self.profile, page_hint
            )

            if form_response.needs_human:
                logger.warning("Human intervention required at step %d: %s", step, form_response.notes)
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.NEEDS_HUMAN,
                    notes=form_response.notes,
                )

            if not form_response.actions:
                logger.warning("No actions returned at step %d — aborting", step)
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.FAILED,
                    notes=f"LLM returned no actions at step {step}",
                )

            await self._execute_actions(page, form_response)

            if form_response.is_complete:
                logger.info("[%s] Application submitted: %s", job.platform, job.title)
                return ApplicationResult(job=job, status=ApplicationStatus.APPLIED)

        # Reached max steps without is_complete
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.FAILED,
            notes=f"Exceeded max form steps ({_MAX_FORM_STEPS})",
        )

    # ------------------------------------------------------------------
    # DOM snapshot builder
    # ------------------------------------------------------------------

    async def _extract_dom_snapshot(self, page: Page) -> str:
        """
        Extract a lightweight HTML snapshot of the active form.
        Keeps only form-relevant tags and a small set of attributes.
        Output is capped at _MAX_SNAPSHOT_CHARS to stay within LLM context limits.
        """
        raw_html: str = await page.evaluate("""() => {
            function simplify(node) {
                if (node.nodeType === Node.TEXT_NODE) {
                    const t = node.textContent.trim();
                    return t ? t + ' ' : '';
                }
                if (node.nodeType !== Node.ELEMENT_NODE) return '';

                const tag = node.tagName.toLowerCase();
                const keepTags = new Set([
                    'form','label','input','select','textarea',
                    'button','option','legend','fieldset','div','span','p','h1','h2','h3','h4'
                ]);
                if (!keepTags.has(tag)) {
                    return Array.from(node.childNodes).map(simplify).join('');
                }

                const keepAttrs = new Set([
                    'type','name','id','placeholder','value','required',
                    'aria-label','aria-labelledby','for','checked','selected'
                ]);
                let attrs = '';
                for (const attr of node.attributes) {
                    if (keepAttrs.has(attr.name)) {
                        attrs += ` ${attr.name}="${attr.value}"`;
                    }
                }
                const children = Array.from(node.childNodes).map(simplify).join('');
                // Skip empty divs/spans with no attributes to reduce noise
                if ((tag === 'div' || tag === 'span') && !attrs && !children.trim()) return '';
                return `<${tag}${attrs}>${children}</${tag}>`;
            }

            // Try to find the active modal/form first, fall back to body
            const modal = (
                document.querySelector('[role="dialog"]') ||
                document.querySelector('.ia-BasePage-content') ||
                document.querySelector('[data-testid="modal-window"]') ||
                document.querySelector('form') ||
                document.body
            );
            return simplify(modal);
        }""")

        snapshot = raw_html.strip()
        if len(snapshot) > _MAX_SNAPSHOT_CHARS:
            snapshot = snapshot[:_MAX_SNAPSHOT_CHARS] + "\n<!-- [TRUNCATED] -->"
        return snapshot

    # ------------------------------------------------------------------
    # External ATS detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_external_redirect(url: str) -> bool:
        return bool(_EXTERNAL_ATS_PATTERNS.search(url))

    # ------------------------------------------------------------------
    # Action executor
    # ------------------------------------------------------------------

    async def _execute_actions(self, page: Page, form_response: FormResponse) -> None:
        """Execute the LLM-generated action list sequentially on the live page."""
        for action in form_response.actions:
            try:
                await BrowserEngine.human_delay(150, 500)
                locator = page.locator(action.selector).first

                if action.type == "fill":
                    await locator.wait_for(state="visible", timeout=8000)
                    await locator.clear()
                    await BrowserEngine.human_type_locator(locator, action.value or "")

                elif action.type == "select":
                    await locator.wait_for(state="visible", timeout=8000)
                    await locator.select_option(label=action.value)

                elif action.type == "click":
                    await locator.wait_for(state="visible", timeout=8000)
                    await locator.scroll_into_view_if_needed()
                    await BrowserEngine.human_delay(200, 600)
                    await locator.click()

                elif action.type == "check":
                    await locator.wait_for(state="visible", timeout=8000)
                    if not await locator.is_checked():
                        await locator.check()

                elif action.type == "upload":
                    # File upload — set input files directly
                    await locator.set_input_files(action.value or "")

                elif action.type == "skip":
                    logger.debug("Skipping field '%s': %s", action.selector, action.value)

                else:
                    logger.warning("Unknown action type: %s", action.type)

            except Exception as exc:
                logger.warning(
                    "Action failed [%s] selector=%s value=%s: %s",
                    action.type, action.selector, action.value, exc,
                )
                # Continue with remaining actions — don't abort the whole form
