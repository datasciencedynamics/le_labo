"""Scrape Le Labo store locations from the official store locator.

The locator renders with JavaScript, so a headless browser loads the page and
every JSON response it fetches is captured. Any record with latitude/longitude
is treated as a store.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pandas as pd

URL = "https://www.lelabofragrances.com/store-search.html"
LAT_KEYS = {"lat", "latitude", "geo_lat", "lat_coord"}
LON_KEYS = {"lng", "lon", "long", "longitude", "geo_lng", "lng_coord"}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def _find_records(obj, found: list) -> None:
    if isinstance(obj, dict):
        keys = {k.lower() for k in obj}
        if keys & LAT_KEYS and keys & LON_KEYS:
            found.append(obj)
        for v in obj.values():
            _find_records(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _find_records(v, found)


def _has_records(body) -> bool:
    hits: list = []
    _find_records(body, hits)
    return bool(hits)


async def scrape_stores(out_dir: str | Path = "output", headless: bool = True) -> pd.DataFrame | None:
    """Scrape all stores from the locator. Returns a DataFrame or None.

    In Jupyter/Colab:   df = await scrape_stores()
    From a script:      asyncio.run(scrape_stores())
    """
    from playwright.async_api import async_playwright

    out = Path(out_dir) / "scrape"
    out.mkdir(parents=True, exist_ok=True)
    captured: list[tuple[str, object]] = []

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
        page = await browser.new_page(user_agent=UA)
        page.on("response", on_response)
        await page.goto(URL, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(3000)

        # Some locators load stores only after a country is selected.
        if not any(_has_records(b) for _, b in captured):
            for sel in await page.query_selector_all("select"):
                try:
                    await sel.select_option(label=re.compile("United States", re.I))
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(2000)
                except Exception:
                    continue

        (out / "page.html").write_text(await page.content(), encoding="utf-8")
        await browser.close()

    (out / "endpoints.txt").write_text("\n".join(u for u, _ in captured), encoding="utf-8")

    records = []
    for url, body in captured:
        hits: list = []
        _find_records(body, hits)
        if hits:
            print(f"{len(hits):>4} store-like records from {url}")
            (out / "raw_response.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
            for h in hits:
                h["_source_url"] = url
            records.extend(hits)

    if not records:
        print("No store records found. Endpoints seen:")
        for u, _ in captured:
            print("  ", u)
        print(f"Inspect {out/'page.html'} or Chrome DevTools > Network > Fetch/XHR.")
        return None

    df = pd.json_normalize(records)
    df = df.drop_duplicates(subset=[c for c in df.columns if c.lower() in LAT_KEYS | LON_KEYS])
    df.to_csv(out / "stores_raw.csv", index=False)
    print(f"Scraped {len(df)} store records -> {out/'stores_raw.csv'}")
    return df


def scrape_stores_sync(out_dir: str | Path = "output", headless: bool = True):
    """Script-friendly wrapper. Do not call inside Jupyter; use `await scrape_stores()`."""
    return asyncio.run(scrape_stores(out_dir, headless))


def coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with numeric `lat` and `lon` columns, detected from any naming."""
    lat = next(c for c in df.columns if c.split(".")[-1].lower() in LAT_KEYS)
    lon = next(c for c in df.columns if c.split(".")[-1].lower() in LON_KEYS)
    out = df.copy()
    out["lat"] = pd.to_numeric(out[lat], errors="coerce")
    out["lon"] = pd.to_numeric(out[lon], errors="coerce")
    return out.dropna(subset=["lat", "lon"])
