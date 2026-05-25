"""
setup_auth.py — One-time manual authentication seeder.

Run this script ONCE before using the agent:
    python setup_auth.py

A visible browser window will open with tabs for Indeed, Naukri, and Wellfound.
Log into each site manually, solve any CAPTCHAs, and close the browser window when done.
The session (cookies, local storage) will be saved to USER_DATA_DIR automatically.
All subsequent agent runs will reuse this saved session silently.
"""

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

# Resolve config relative to this file so the script works from any cwd
sys.path.insert(0, str(Path(__file__).parent))
from config import settings


PLATFORM_URLS = {
    "Indeed": "https://www.indeed.com/account/login",
    "Naukri": "https://www.naukri.com/nlogin/login",
    "Wellfound": "https://wellfound.com/login",
}

INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║           MANUAL LOGIN REQUIRED — Please read carefully             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  A browser window has opened with tabs for:                          ║
║    • Indeed                                                          ║
║    • Naukri                                                          ║
║    • Wellfound                                                       ║
║                                                                      ║
║  For EACH tab:                                                       ║
║    1. Log in with your email and password                            ║
║    2. Complete any CAPTCHA or 2FA challenges                         ║
║    3. Make sure you reach the logged-in home/dashboard page          ║
║                                                                      ║
║  When you have logged into ALL three sites, CLOSE the browser.      ║
║  Your session will be saved automatically.                           ║
║                                                                      ║
║  Profile directory: {user_data_dir}                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""


async def run() -> None:
    print(INSTRUCTIONS.format(user_data_dir=settings.user_data_dir))

    settings.user_data_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(settings.user_data_dir),
            headless=False,  # ALWAYS visible — user must interact
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
            args=["--disable-blink-features=AutomationControlled"],
        )

        # Open a tab for each platform
        pages = []
        platform_items = list(PLATFORM_URLS.items())

        # Reuse the default blank page for the first URL
        first_page = context.pages[0] if context.pages else await context.new_page()
        await first_page.goto(platform_items[0][1], wait_until="domcontentloaded")
        print(f"  Opened tab for {platform_items[0][0]}")
        pages.append(first_page)

        for name, url in platform_items[1:]:
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            print(f"  Opened tab for {name}")
            pages.append(page)

        print("\nWaiting for you to log in and close the browser...")

        # Wait until ALL pages are closed (user closes the browser window)
        try:
            while any(not p.is_closed() for p in pages):
                await asyncio.sleep(1)
        except Exception:
            pass

        await context.close()

    print("\n✓ Session saved successfully.")
    print(f"  Profile stored at: {settings.user_data_dir}")
    print("\nYou can now run the agent:")
    print("  python main.py")


if __name__ == "__main__":
    asyncio.run(run())
