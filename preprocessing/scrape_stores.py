#!/usr/bin/env python3
"""
scrape_stores.py
----------------
Scrapes store locations from the official Le Labo store locator.

The locator renders with JavaScript, so headless Chromium loads the page and
every JSON response it fetches is captured. Any record carrying latitude and
longitude is treated as a store. Raw responses, the endpoint list, and the
rendered page are kept in data/raw/scrape/ for inspection.

In a notebook, call `await scrape(...)` instead of running this file, since
Jupyter already runs an event loop.

Example usage:
    python preprocessing/scrape_stores.py \
        --output-data-file ./data/raw/stores_scraped.csv
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from core.config import RAW_DATA_DIR, SCRAPE_DIR
from core.constants import lat_keys, locator_url, lon_keys, stores_scraped, user_agent
from core.functions import find_store_records

app = typer.Typer(add_completion=False)


def _has_records(body) -> bool:
    hits: list = []
    find_store_records(body, hits)
    return bool(hits)


async def scrape(
    output_data_file: Path = RAW_DATA_DIR / stores_scraped,
    debug_dir: Path = SCRAPE_DIR,
    headless: bool = True,
) -> Optional[pd.DataFrame]:
    """Scrape the locator. Returns the store DataFrame, or None if none found."""
    from playwright.async_api import async_playwright

    debug_dir.mkdir(parents=True, exist_ok=True)
    captured: list = []

    ############################################################################
    # Step 1. Capture every JSON response the locator loads
    ############################################################################

    async def on_response(resp):
        ctype = resp.headers.get("content-type", "")
        if "json" not in ctype and not resp.url.endswith(".json"):
            return
        try:
            captured.append((resp.url, await resp.json()))
        except Exception:
            pass

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page(user_agent=user_agent)
        page.on("response", on_response)
        await page.goto(locator_url, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(3000)

        ########################################################################
        # Step 2. Some locators load stores only after a country is selected
        ########################################################################
        if not any(_has_records(b) for _, b in captured):
            for sel in await page.query_selector_all("select"):
                try:
                    await sel.select_option(label=re.compile("United States", re.I))
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(2000)
                except Exception:
                    continue

        (debug_dir / "page.html").write_text(await page.content(), encoding="utf-8")
        await browser.close()

    (debug_dir / "endpoints.txt").write_text(
        "\n".join(u for u, _ in captured), encoding="utf-8"
    )

    ############################################################################
    # Step 3. Keep records that carry coordinates
    ############################################################################
    records = []
    for url, body in captured:
        hits: list = []
        find_store_records(body, hits)
        if hits:
            print(f"{len(hits):>4} store-like records from {url}")
            (debug_dir / "raw_response.json").write_text(
                json.dumps(body, indent=2), encoding="utf-8"
            )
            for h in hits:
                h["_source_url"] = url
            records.extend(hits)

    if not records:
        print("No store records found. Endpoints seen:")
        for u, _ in captured:
            print("  ", u)
        print(f"Inspect {debug_dir / 'page.html'} or Chrome DevTools > Network > Fetch/XHR.")
        return None

    ############################################################################
    # Step 4. Save
    ############################################################################
    df = pd.json_normalize(records)
    coord_cols = [c for c in df.columns if c.split(".")[-1].lower() in lat_keys | lon_keys]
    df = df.drop_duplicates(subset=coord_cols)
    output_data_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_data_file, index=False)
    print(f"Scraped {len(df)} store records -> {output_data_file}")
    return df


@app.command()
def main(
    output_data_file: Path = typer.Option(
        RAW_DATA_DIR / stores_scraped,
        "--output-data-file",
        help="Where to save the scraped stores.",
    ),
    debug_dir: Path = typer.Option(
        SCRAPE_DIR, "--debug-dir", help="Raw responses, endpoints, page HTML."
    ),
    headless: bool = typer.Option(True, "--headless/--headed"),
) -> None:
    """Scrape the Le Labo store locator."""
    df = asyncio.run(scrape(output_data_file, debug_dir, headless))
    if df is None:
        typer.secho(
            "Scrape found no stores; later steps will use the fallback list.",
            fg="yellow",
        )


if __name__ == "__main__":
    app()
