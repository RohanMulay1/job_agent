"""Quick debug: screenshot + HTML snapshot of Naukri and Wellfound search pages."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from config import settings

OUT = Path(__file__).parent / "logs"
OUT.mkdir(exist_ok=True)


async def snap(name: str, url: str) -> None:
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(settings.user_data_dir),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        print(f"\n[{name}] Loading: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(4)

        shot = OUT / f"debug_{name}.png"
        await page.screenshot(path=str(shot), full_page=False)
        print(f"[{name}] Screenshot saved: {shot}")

        html = await page.content()
        html_file = OUT / f"debug_{name}.html"
        html_file.write_text(html[:50_000], encoding="utf-8")
        print(f"[{name}] HTML (first 50k) saved: {html_file}")

        # Print job-card-like elements found
        for sel in [
            "article[data-job-id]", "div[data-job-id]", "article.jobTuple",
            "div[class*='JobTuple']", "div[class*='job-tuple']",
            "li[class*='JobTuple']", "div[data-test='JobListing']",
            "div[class*='jobListing']", "div[class*='job-listing']",
        ]:
            count = await page.locator(sel).count()
            if count:
                print(f"  FOUND {count} × '{sel}'")

        await ctx.close()


async def main():
    await snap("naukri", "https://www.naukri.com/ai-engineer-jobs?jobAge=7&experience=0,3")
    await snap("wellfound", "https://wellfound.com/jobs?q=AI+Engineer&remote=true")

asyncio.run(main())
