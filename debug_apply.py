"""Screenshot what Naukri's job page looks like before and after clicking Apply."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PwTimeout
from config import settings

OUT = Path("logs")
OUT.mkdir(exist_ok=True)

# A real Naukri job URL — paste one from your last run's log here
JOB_URL = "https://www.naukri.com/job-listings-ai-engineer-m-tech-iit-mandi-ihub-and-hci-foundation-new-delhi-mandi-0-to-3-years-210526502647"

async def main():
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(settings.user_data_dir),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(JOB_URL, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(3)

        await page.screenshot(path=str(OUT / "apply_before.png"))
        print("Saved apply_before.png")

        # Print all buttons visible
        btns = await page.locator("button").all()
        print(f"\nFound {len(btns)} buttons:")
        for b in btns[:20]:
            txt = (await b.text_content() or "").strip()
            cls = await b.get_attribute("class") or ""
            if txt:
                print(f"  '{txt}' class='{cls[:60]}'")

        # Try clicking Apply
        apply_btn = page.locator(
            "button:has-text('Apply'), button:has-text('Easy Apply')"
        ).first
        if await apply_btn.count():
            print("\nClicking Apply button...")
            await apply_btn.click()
            await asyncio.sleep(4)
            await page.screenshot(path=str(OUT / "apply_after.png"))
            print("Saved apply_after.png")

            # Print what appeared
            print("Page title:", await page.title())
            print("URL after click:", page.url)

            # Check for dialogs/panels
            for sel in ["[role='dialog']", ".apply-modal", "form", "[class*='apply']",
                        "[class*='Apply']", "[class*='modal']", "[class*='drawer']",
                        "[class*='panel']", "[class*='sidebar']"]:
                count = await page.locator(sel).count()
                if count:
                    print(f"  FOUND: {count} × '{sel}'")
        else:
            print("\nNo Apply button found on page")

        await ctx.close()

asyncio.run(main())
