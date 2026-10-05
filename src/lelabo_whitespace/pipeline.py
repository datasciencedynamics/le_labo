"""End-to-end pipeline: stores -> counties -> model -> CSV outputs + HTML map."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from . import data, model, report, scrape


def _prepare_stores(stores, geo, query: str | None) -> tuple[pd.DataFrame, str]:
    if stores is None:
        df = data.fallback_stores()
        note = (f"{len(df)} standalone US boutiques compiled from Yelp listings in "
                "October 2026 (bundled fallback list). Department-store counters "
                "excluded; a few boutiques may be missing.")
        return df, note
    if isinstance(stores, (str, Path)):
        stores = pd.read_csv(stores)
    df = stores.copy()
    if query:
        df = df.query(query)
    if "county_fips" not in df.columns:
        df = data.assign_counties(scrape.coordinates(df), geo)
    note = (f"{len(df)} US locations from the official Le Labo store locator"
            f"{' (filtered: ' + query + ')' if query else ''}, assigned to counties by "
            "point-in-polygon.")
    return df, note


def run(stores=None, *, out_dir: str | Path = "output", query: str | None = None,
        radius_mi: float = 60, repeats: int = 20) -> model.Result:
    """Run the model and write outputs.

    stores: None (bundled list), a CSV path, or a DataFrame. Scraped data needs
            lat/lon columns; a `county_fips` column skips geocoding.
    query:  optional pandas query to keep only boutiques, e.g. "type == 'Boutique'".
    """
    out = Path(out_dir)
    cache = out / "cache"
    geo = data.load_geojson(cache)
    counties = data.load_counties(cache, geo)
    st, note = _prepare_stores(stores, geo, query)

    res = model.fit(counties, st.county_fips.value_counts(), radius_mi=radius_mi, repeats=repeats)

    out.mkdir(parents=True, exist_ok=True)
    st.to_csv(out / "stores_used.csv", index=False)
    keep = ["fips", "label", "total_population", "median_hh_inc", "college_pct",
            "stores", "p", "rank", "mi_to_store", "segment"]
    res.scored[keep].to_csv(out / "county_scores.csv", index=False)
    res.scored[res.scored.segment == "open"][keep].to_csv(out / "whitespace_open.csv", index=False)
    res.scored[res.scored.segment == "fill"][keep].to_csv(out / "whitespace_fill.csv", index=False)
    res.coef.rename("coef").to_csv(out / "coefficients.csv")
    html = report.render(res, geo, out / "lelabo_whitespace.html",
                         store_note=note, radius_mi=radius_mi)

    print(f"\nStores: {int(res.scored.stores.sum())} in {int((res.scored.stores > 0).sum())} counties")
    print(f"CV AUC {res.auc:.3f} | AP {res.ap:.3f} | base rate {res.base_rate:.4f}")
    print("\nTop open markets:")
    print(res.scored[res.scored.segment == "open"].head(10)[["label", "p", "mi_to_store"]]
          .round(3).to_string(index=False))
    print(f"\nWrote outputs to {out.resolve()}\nMap: {html.resolve()}")
    return res


async def run_with_scrape(*, out_dir: str | Path = "output", query: str | None = None,
                          **kw) -> model.Result:
    """Notebook entry point: scrape the official locator, then run.
        res = await run_with_scrape()
    Falls back to the bundled list if the scrape finds nothing."""
    df = await scrape.scrape_stores(out_dir)
    if df is None:
        print("Scrape returned nothing; using the bundled store list.")
    return run(df, out_dir=out_dir, query=query, **kw)


def run_with_scrape_sync(**kw) -> model.Result:
    return asyncio.run(run_with_scrape(**kw))
