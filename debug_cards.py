"""Print inner HTML of the first Naukri job card to find correct selectors."""
import asyncio
from playwright.async_api import async_playwright
from config import settings


async def main():
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(settings.user_data_dir),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        url = "https://www.naukri.com/ai-engineer-jobs?jobAge=7&experience=0,3"
        print(f"Loading: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(4)

        # Get inner HTML of first card
        card = page.locator("div[data-job-id]").first
        if await card.count():
            html = await card.inner_html()
            print("\n=== FIRST CARD HTML ===")
            print(html[:3000])
        else:
            print("No cards found — page title:", await page.title())
            print(await page.content()[:2000])

        await ctx.close()

asyncio.run(main())
