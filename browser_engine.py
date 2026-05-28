from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from types import TracebackType
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from config import settings

logger = logging.getLogger(__name__)

# Realistic Chromium user-agent matching a common desktop Chrome version
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Viewport that looks like a normal 1080p screen
_VIEWPORT = {"width": 1366, "height": 768}


class BrowserEngine:
    """Async Playwright wrapper with stealth, persistent sessions, and human-like behaviour."""

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._headless: bool = settings.headless
        self._user_data_dir: Path = settings.user_data_dir

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def launch(self, headless: Optional[bool] = None) -> BrowserContext:
        """Launch a persistent browser context from the saved profile directory."""
        if headless is not None:
            self._headless = headless

        self._user_data_dir.mkdir(parents=True, exist_ok=True)

        # Prefer system Chrome over Playwright's bundled Chromium for better fingerprinting.
        _CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        import os
        executable = _CHROME if os.path.exists(_CHROME) else None

        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self._user_data_dir),
            headless=self._headless,
            executable_path=executable,  # None → Playwright's bundled Chromium
            user_agent=_USER_AGENT,
            viewport=_VIEWPORT,
            locale="en-US",
            timezone_id="Asia/Kolkata",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-service-autorun",
                "--password-store=basic",
            ],
        )
        if executable:
            logger.info("Using system browser: %s", executable)

        logger.info("Browser launched (headless=%s, profile=%s)", self._headless, self._user_data_dir)
        return self._context

    async def new_page(self) -> Page:
        if self._context is None:
            raise RuntimeError("BrowserEngine not launched — call launch() first")
        return await self._context.new_page()

    async def close(self) -> None:
        if self._context:
            await self._context.close()
            self._context = None
            logger.info("Browser context closed")
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BrowserEngine":
        await self.launch()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Human-like helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def human_delay(min_ms: int = 500, max_ms: int = 1500) -> None:
        """Sleep for a random duration in [min_ms, max_ms] milliseconds."""
        delay = random.randint(min_ms, max_ms) / 1000.0
        await asyncio.sleep(delay)

    @staticmethod
    async def human_type(page: Page, selector: str, text: str) -> None:
        """Type text into a field with per-keystroke random delay mimicking human typing."""
        locator = page.locator(selector).first
        await locator.click()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        for char in text:
            await locator.press(char)
            await asyncio.sleep(random.uniform(0.03, 0.12))

    @staticmethod
    async def human_type_locator(locator, text: str) -> None:
        """Same as human_type but accepts a Playwright Locator directly."""
        await locator.click()
        await asyncio.sleep(random.uniform(0.1, 0.3))
        for char in text:
            await locator.press(char)
            await asyncio.sleep(random.uniform(0.03, 0.12))

    @staticmethod
    async def scroll_into_view(page: Page, selector: str) -> None:
        """Scroll element into view before interacting with it."""
        await page.locator(selector).first.scroll_into_view_if_needed()
        await asyncio.sleep(random.uniform(0.1, 0.4))

    @staticmethod
    async def safe_click(page: Page, selector: str, timeout: int = 10_000) -> None:
        """Click with a short human delay and explicit wait."""
        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=timeout)
        await BrowserEngine.human_delay(200, 600)
        await locator.click()

    @staticmethod
    def get_context_from_engine(engine: "BrowserEngine") -> BrowserContext:
        if engine._context is None:
            raise RuntimeError("BrowserEngine not launched")
        return engine._context
