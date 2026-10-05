"""Render the self-contained HTML whitespace map."""

from __future__ import annotations

import json
from datetime import date
from importlib import resources
from pathlib import Path

from .model import Result

TABLE_COLS = ["label", "county", "state", "total_population", "median_hh_inc",
              "college_pct", "p", "mi_to_store", "stores"]


def _round(c):
    if isinstance(c[0], (float, int)):
        return [round(c[0], 3), round(c[1], 3)]
    return [_round(x) for x in c]


def _slim_geo(geo: dict, res: Result, tolerance: float) -> dict:
    from shapely.geometry import mapping, shape

    info = res.scored.set_index("fips")
    feats = []
    for f in geo["features"]:
        fid = int(f["id"])
        if fid not in info.index:
            continue
        g = shape(f["geometry"]).simplify(tolerance, preserve_topology=True)
        if g.is_empty:
            continue
        r = info.loc[fid]
        feats.append({
            "type": "Feature", "id": fid,
            "geometry": {"type": g.geom_type, "coordinates": _round(mapping(g)["coordinates"])},
            "properties": {"n": r.county, "s": r.state, "p": round(float(r.p), 4),
                           "k": int(r.stores), "m": int(round(r.mi_to_store)),
                           "pop": int(r.total_population), "g": r.segment},
        })
    return {"type": "FeatureCollection", "features": feats}


def render(res: Result, geo: dict, path: str | Path, *, store_note: str,
           radius_mi: float = 60, n_open: int = 12, n_fill: int = 8,
           tolerance: float = 0.015) -> Path:
    s = res.scored
    tables = {
        "open": s[s.segment == "open"].head(n_open)[TABLE_COLS].to_dict("records"),
        "fill": s[s.segment == "fill"].head(n_fill)[TABLE_COLS].to_dict("records"),
        "stores": s[s.segment == "store"][TABLE_COLS].to_dict("records"),
        "coef": res.coef.round(3).to_dict(),
    }
    tpl = resources.files("lelabo_whitespace").joinpath("assets/report_template.html").read_text()
    subs = {
        "__GEO__": json.dumps(_slim_geo(geo, res, tolerance), separators=(",", ":")),
        "__TABLES__": json.dumps(tables, default=float),
        "__MONTH__": date.today().strftime("%b %Y"),
        "__N_STORES__": str(int(s.stores.sum())),
        "__N_COUNTIES__": str(int((s.stores > 0).sum())),
        "__N_ALL__": f"{len(s):,}",
        "__AUC__": f"{res.auc:.2f}",
        "__AP__": f"{res.ap:.2f}",
        "__BASE__": f"{res.base_rate:.3f}",
        "__RADIUS__": f"{radius_mi:g}",
        "__STORE_NOTE__": store_note,
    }
    for k, v in subs.items():
        tpl = tpl.replace(k, v)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tpl, encoding="utf-8")
    return path
