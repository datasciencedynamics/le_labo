################################################################################
######################### Import Requisite Libraries ###########################
################################################################################

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from core.config import cv_folds, make_model, min_store_counties, rstate
from core.constants import lat_keys, lon_keys, store_count, target_outcome, var_index

################################################################################
############################## Data Download ###################################
################################################################################


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download `url` to `dest` unless it already exists."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"Exists, skipping download: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def load_geojson(path: Path) -> dict:
    return json.loads(Path(path).read_text())


################################################################################
############################## County Geometry #################################
################################################################################


def county_geometry_table(geo: dict) -> pd.DataFrame:
    """Land area and an approximate centroid (mean of exterior ring vertices)."""
    rows = []
    for f in geo["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        pts = np.vstack([np.asarray(p[0]) for p in polys])
        rows.append(
            (int(f["id"]), f["properties"]["CENSUSAREA"], pts[:, 0].mean(), pts[:, 1].mean())
        )
    return pd.DataFrame(rows, columns=[var_index, "area_sqmi", "lon", "lat"])


def haversine_miles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise great-circle miles between rows of a and b, each [lat, lon]."""
    la1, lo1 = np.radians(a[:, None, 0]), np.radians(a[:, None, 1])
    la2, lo2 = np.radians(b[None, :, 0]), np.radians(b[None, :, 1])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(h))


################################################################################
############################## Store Records ###################################
################################################################################


def find_store_records(obj, found: list) -> None:
    """Walk any JSON structure and collect dicts that carry coordinates."""
    if isinstance(obj, dict):
        keys = {k.lower() for k in obj}
        if keys & lat_keys and keys & lon_keys:
            found.append(obj)
        for v in obj.values():
            find_store_records(v, found)
    elif isinstance(obj, list):
        for v in obj:
            find_store_records(v, found)


def extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with numeric `lat`/`lon`, detected from any common naming."""
    lat = next((c for c in df.columns if c.split(".")[-1].lower() in lat_keys), None)
    lon = next((c for c in df.columns if c.split(".")[-1].lower() in lon_keys), None)
    if lat is None or lon is None:
        raise KeyError(
            "Store file needs latitude/longitude columns or a county_fips column. "
            f"Columns found: {list(df.columns)}"
        )
    out = df.copy()
    out["lat"] = pd.to_numeric(out[lat], errors="coerce")
    out["lon"] = pd.to_numeric(out[lon], errors="coerce")
    return out.dropna(subset=["lat", "lon"])


def assign_counties(stores: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """Point-in-polygon county assignment. Non-US stores are dropped."""
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    geoms = [shape(f["geometry"]) for f in geo["features"]]
    ids = [int(f["id"]) for f in geo["features"]]
    tree = STRtree(geoms)

    fips = []
    for lat, lon in zip(stores.lat, stores.lon):
        pt = Point(lon, lat)
        fips.append(next((ids[i] for i in tree.query(pt) if geoms[i].contains(pt)), None))

    out = stores.copy()
    out["county_fips"] = fips
    dropped = int(out.county_fips.isna().sum())
    if dropped:
        print(f"{dropped} stores fell outside US counties and were dropped (non-US).")
    return out.dropna(subset=["county_fips"]).astype({"county_fips": int})


################################################################################
########################### Out-of-Fold Scoring ################################
################################################################################


def oof_scores(X: pd.DataFrame, y: pd.Series, repeats: int, seed: int = rstate) -> np.ndarray:
    """Out-of-fold probabilities averaged over `repeats` stratified k-fold runs.

    No county is ever scored by a model that saw its own label.
    """
    if int(y.sum()) < min_store_counties:
        raise ValueError(
            f"Need at least {min_store_counties} store counties to fit; got {int(y.sum())}."
        )
    folds = min(cv_folds, int(y.sum()))
    preds = [
        cross_val_predict(
            make_model(),
            X.values,
            y.values,
            method="predict_proba",
            cv=StratifiedKFold(folds, shuffle=True, random_state=seed + r),
        )[:, 1]
        for r in range(repeats)
    ]
    return np.mean(preds, axis=0)


def fit_metrics(y: pd.Series, p: np.ndarray) -> dict:
    return {
        "cv_auc": float(roc_auc_score(y, p)),
        "cv_average_precision": float(average_precision_score(y, p)),
        "base_rate": float(y.mean()),
        "n_counties": int(len(y)),
        "n_store_counties": int(y.sum()),
    }


################################################################################
############################ Segmentation ######################################
################################################################################


def segment_counties(
    scores: pd.Series, labels: pd.DataFrame, meta: pd.DataFrame, radius: float
) -> pd.DataFrame:
    """Join scores, labels and meta; add distance to nearest store county,
    segment (store / fill / open) and overall rank."""
    d = meta.join(labels).join(scores.rename("p"))
    store_pts = d.loc[d[target_outcome] == 1, ["lat", "lon"]].values
    d["mi_to_store"] = haversine_miles(d[["lat", "lon"]].values, store_pts).min(axis=1)
    d["segment"] = np.select(
        [d[target_outcome] == 1, d.mi_to_store < radius], ["store", "fill"], "open"
    )
    d["rank"] = d.p.rank(ascending=False, method="first").astype(int)
    d[store_count] = d[store_count].astype(int)
    return d.sort_values("p", ascending=False)


################################################################################
############################ HTML Report Geometry ##############################
################################################################################


def _round_coords(c):
    if isinstance(c[0], (float, int)):
        return [round(c[0], 3), round(c[1], 3)]
    return [_round_coords(x) for x in c]


def slim_geojson(geo: dict, scored: pd.DataFrame, tolerance: float = 0.015) -> dict:
    """Simplified county shapes carrying only the fields the map needs."""
    from shapely.geometry import mapping, shape

    feats = []
    for f in geo["features"]:
        fid = int(f["id"])
        if fid not in scored.index:
            continue
        g = shape(f["geometry"]).simplify(tolerance, preserve_topology=True)
        if g.is_empty:
            continue
        r = scored.loc[fid]
        feats.append(
            {
                "type": "Feature",
                "id": fid,
                "geometry": {
                    "type": g.geom_type,
                    "coordinates": _round_coords(mapping(g)["coordinates"]),
                },
                "properties": {
                    "n": r.county,
                    "s": r.state,
                    "p": round(float(r.p), 4),
                    "k": int(r[store_count]),
                    "m": int(round(r.mi_to_store)),
                    "pop": int(r.total_population),
                    "g": r.segment,
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}
