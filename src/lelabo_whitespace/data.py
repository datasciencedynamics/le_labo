"""County demographics, county boundaries, and store-to-county assignment."""

from __future__ import annotations

import json
import urllib.request
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

DEMOG_URL = ("https://raw.githubusercontent.com/MEDSL/2018-elections-unoffical/"
             "master/election-context-2018.csv")
GEO_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

DEMOG_COLS = ["state", "county", "fips", "total_population", "foreignborn_pct",
              "age29andunder_pct", "age65andolder_pct", "median_hh_inc",
              "lesscollege_pct", "rural_pct"]

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_geojson(cache: Path) -> dict:
    return json.loads(_download(GEO_URL, cache / "counties.geojson").read_text())


def load_counties(cache: Path, geo: dict) -> pd.DataFrame:
    """One row per county: demographics, land area, centroid, engineered features."""
    d = pd.read_csv(_download(DEMOG_URL, cache / "county_demographics.csv"))
    d = d[DEMOG_COLS].dropna()

    rows = []
    for f in geo["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        pts = np.vstack([np.asarray(p[0]) for p in polys])
        rows.append((int(f["id"]), f["properties"]["CENSUSAREA"], pts[:, 0].mean(), pts[:, 1].mean()))
    area = pd.DataFrame(rows, columns=["fips", "area_sqmi", "lon", "lat"])

    d = d.merge(area, on="fips")
    d = d[d.area_sqmi > 0].copy()
    d["log_pop"] = np.log(d.total_population)
    d["log_density"] = np.log(d.total_population / d.area_sqmi)
    d["log_income"] = np.log(d.median_hh_inc)
    d["college_pct"] = 100 - d.lesscollege_pct
    abbr = d.state.map(STATE_ABBR).fillna(d.state)
    d["label"] = np.where(d.county == d.state, d.county, d.county + ", " + abbr)
    return d.reset_index(drop=True)


def fallback_stores() -> pd.DataFrame:
    """Web-compiled list of US standalone boutiques (Oct 2026), already county-coded."""
    with resources.files("lelabo_whitespace").joinpath("assets/stores_fallback.csv").open() as fh:
        return pd.read_csv(fh)


def assign_counties(stores: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """Add `county_fips` to stores that have `lat`/`lon`, by point-in-polygon.
    Stores outside the US (non-matching) are dropped."""
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    geoms = [shape(f["geometry"]) for f in geo["features"]]
    ids = [int(f["id"]) for f in geo["features"]]
    tree = STRtree(geoms)

    fips = []
    for lat, lon in zip(stores.lat, stores.lon):
        pt = Point(lon, lat)
        hit = None
        for i in tree.query(pt):
            if geoms[i].contains(pt):
                hit = ids[i]
                break
        fips.append(hit)
    out = stores.copy()
    out["county_fips"] = fips
    dropped = out.county_fips.isna().sum()
    if dropped:
        print(f"{dropped} stores fell outside US counties and were dropped (non-US locations).")
    return out.dropna(subset=["county_fips"]).astype({"county_fips": int})
