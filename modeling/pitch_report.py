"""
pitch_report.py
---------------
Builds the client-facing PDF report from the flat files `make export_dash`
writes (counties.csv, coefficients.csv, metrics_ci.csv, loo_ranks.csv,
radius_sensitivity.csv, stores.csv, report_meta.json) plus a map image.

Every sentence that states a result is computed from those files, so the
report stays correct after the model or the store list is rerun.

Used by modeling/pitch_pdf.py (make pitch_pdf). The Dash app serves the PDF
that step copies into its data/ folder.

Public functions:
    build_html(data_dir, map_png, top_n=12, county=None) -> str
    html_to_pdf(html, pdf_path)                          -> Path
    capture_map(report_html, png_path)                   -> Path
"""

import base64
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TOP_N_MAX = 15  # rows that fit on the shortlist page

# Metro names for counties whose name alone does not say where they are.
# Counties missing here are shown with their county name only.
METRO = {
    "Santa Clara, CA": "San Jose and Silicon Valley",
    "Maricopa, AZ": "Phoenix and Scottsdale",
    "Wake, NC": "Raleigh",
    "Mecklenburg, NC": "Charlotte",
    "Orange, FL": "Orlando",
    "Philadelphia, PA": "Philadelphia",
    "San Bernardino, CA": "Inland Empire",
    "Riverside, CA": "Inland Empire",
    "Franklin, OH": "Columbus",
    "Hillsborough, FL": "Tampa",
    "Oakland, MI": "Detroit's northern suburbs",
    "Wayne, MI": "Detroit",
    "Bexar, TX": "San Antonio",
    "Tarrant, TX": "Fort Worth",
    "Sacramento, CA": "Sacramento",
    "Cuyahoga, OH": "Cleveland",
    "Marion, IN": "Indianapolis",
    "Hidalgo, TX": "McAllen",
    "Queens, NY": "New York City",
    "Bronx, NY": "New York City",
    "Hudson, NJ": "Jersey City and Hoboken",
    "Fairfax, VA": "Northern Virginia",
    "Alameda, CA": "Oakland and Berkeley",
    "Montgomery, MD": "Bethesda",
    "Broward, FL": "Fort Lauderdale",
    "Middlesex, MA": "Cambridge and Boston suburbs",
    "Kings, NY": "Brooklyn",
    "New York": "Manhattan",
    "Clark, NV": "Las Vegas",
    "Davidson, TN": "Nashville",
    "Palm Beach, FL": "West Palm Beach",
    "Montgomery, TX": "The Woodlands, Houston area",
}
CITY_ALIASES = {"Brooklyn": "New York", "Bal Harbour": "Miami", "Malibu": "Los Angeles",
                "Newport Beach": "Orange County", "The Woodlands": "Houston"}

# Census Bureau regions
REGION = {}
for _r, _states in {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}.items():
    for _s in _states.split():
        REGION[_s] = _r

FEATURE_LABELS = {
    "log_pop": "population", "log_density": "population density",
    "log_income": "median household income", "college_pct": "bachelor's degree share",
    "foreignborn_pct": "foreign-born share", "age29andunder_pct": "share age 29 and under",
    "age65andolder_pct": "share age 65 and over", "rural_pct": "rural share",
}

FONT_LINK = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=Courier+Prime:wght@400;700&family=Libre+Franklin:wght@400;500;600;700'
             '&display=swap">')


################################################################################
# Helpers
################################################################################
def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _pct(p):
    return f"{p * 100:.0f}%" if p >= 0.095 else f"{p * 100:.1f}%"


def _abbr(label):
    return label.split(", ")[-1] if ", " in label else ("DC" if "Columbia" in label else "NY")


def _metro(label):
    return METRO.get(label, "")


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _haversine_idx(lat, lon, lats, lons):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat, lon, lats, lons))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return int(np.argmin(h))


class _Data:
    """Everything the report needs, loaded once from the export folder."""

    def __init__(self, data_dir):
        d = Path(data_dir)
        self.meta = json.loads((d / "report_meta.json").read_text())
        self.df = pd.read_csv(d / "counties.csv", dtype={"fips5": str})
        self.coef = pd.read_csv(d / "coefficients.csv")
        self.ci = pd.read_csv(d / "metrics_ci.csv")
        self.loo = pd.read_csv(d / "loo_ranks.csv") if (d / "loo_ranks.csv").exists() else None
        self.sens = (pd.read_csv(d / "radius_sensitivity.csv")
                     if (d / "radius_sensitivity.csv").exists() else None)
        self.stores_raw = pd.read_csv(d / "stores.csv")

        df = self.df
        self.stores = df[df.stores > 0].sort_values("p", ascending=False)
        self.open_all = df[df.segment == "open"].sort_values("p", ascending=False)
        self.fill_all = df[df.segment == "fill"].sort_values("p", ascending=False)
        self.fill = self.fill_all.head(8)
        self.maxp = df.p.max()

        city = self.stores_raw.groupby("county_fips").city.first().replace(CITY_ALIASES)
        sc = self.stores

        def nearest(r):
            i = _haversine_idx(r.lat, r.lon, sc.lat.values, sc.lon.values)
            return city.get(int(sc.fips.values[i]), sc.label.values[i])

        self.nearest = {r.fips5: nearest(r) for r in df[df.segment != "store"].itertuples()}
        self.store_names = self.stores_raw.groupby("county_fips").name.apply(list)


################################################################################
# Sections
################################################################################
def _bar(p, maxp):
    return (f'<span class="bar" style="width:{max(2, p / maxp * 64):.0f}px"></span>'
            f'<span class="sc">{_pct(p)}</span>')


def _rows_open(D, top):
    return "".join(
        f"<tr><td class='rk'>{i}</td><td><b>{r.label}</b> <span class='sub'>{_metro(r.label)}</span></td>"
        f"<td class='n'>{r.total_population / 1e6:.1f}M</td><td class='n'>{r.college_pct:.0f}%</td>"
        f"<td>{_bar(r.p, D.maxp)}</td><td class='n'>{r.mi_to_store:.0f}</td>"
        f"<td class='sub'>{D.nearest.get(r.fips5, '')}</td></tr>"
        for i, r in enumerate(top.itertuples(), 1))


def _rows_fill(D):
    return "".join(
        f"<tr><td><b>{r.label}</b> <span class='sub'>{_metro(r.label)}</span></td>"
        f"<td class='n'>{r.total_population / 1e6:.1f}M</td><td class='n'>{r.college_pct:.0f}%</td>"
        f"<td>{_bar(r.p, D.maxp)}</td><td class='n'>{r.mi_to_store:.0f}</td></tr>"
        for r in D.fill.itertuples())


def _coef_rows(D):
    c = D.coef.sort_values("coef", ascending=False)
    mx = 1.1 * c.coef.abs().max()
    out = []
    for f, v in zip(c.feature, c.coef):
        w = abs(v) / mx * 50
        left = 50 if v >= 0 else 50 - w
        out.append(f"<div class='cl'>{FEATURE_LABELS.get(f, f).capitalize()}</div><div class='track'>"
                   f"<i class='{'pos' if v >= 0 else 'neg'}' style='left:{left:.1f}%;width:{w:.1f}%'></i></div>")
    return "".join(out)


def _tick(rank, lmax):
    return (math.log10(rank) + 0.15) / (lmax + 0.15) * 100


def _loo_rows(D):
    lmax = math.log10(max(250, D.loo.whitespace_rank.max() * 1.3))
    out = []
    for r in D.loo.sort_values("whitespace_rank").itertuples():
        name = {"Kings, NY": "Brooklyn, NY", "New York": "Manhattan, NY"}.get(r.county, r.county)
        out.append(f"<div class='ll'>{name}</div><div class='lt'><i class='{'hit' if r.whitespace_rank <= 25 else 'miss'}' "
                   f"style='width:{_tick(r.whitespace_rank, lmax):.1f}%'></i><span>#{r.whitespace_rank}</span></div>")
    return "".join(out), _tick(25, lmax)


def _metric_rows(D):
    out = []
    for r in D.ci.itertuples():
        d = 4 if r.metric.startswith("Brier") else 2
        out.append(f"<tr><td>{r.metric}</td><td class='n'>{r.estimate:.{d}f}</td>"
                   f"<td class='n'>{r.ci_low:.{d}f} to {r.ci_high:.{d}f}</td></tr>")
    return "".join(out)


def _contributions(D, row):
    """Linear-model contribution of each feature for one county, relative to the
    average county (coefficient x standardized value)."""
    out = {}
    for f, c in zip(D.coef.feature, D.coef.coef):
        if f in D.df.columns:
            col = D.df[f]
            z = (row[f] - col.mean()) / (col.std() or 1)
            out[f] = c * z
    return pd.Series(out).sort_values()


def _spotlight(D, fips5, page_no):
    hit = D.df[D.df.fips5 == fips5]
    if hit.empty:
        return ""
    r = hit.iloc[0]
    seg_txt = {
        "store": "already home to a Le Labo boutique",
        "fill": f"a fill-in market: no boutique of its own, one within {int(D.meta['radius_mi'])} miles",
        "open": f"an open market: no boutique within {int(D.meta['radius_mi'])} miles",
    }[r.segment]
    open_rank = ""
    if r.segment == "open":
        open_rank = (f" It ranks <b>#{int((D.open_all.p > r.p).sum()) + 1}</b> of "
                     f"{len(D.open_all):,} open markets.")
    near = ("" if r.stores > 0 else
            f" The nearest boutique is in {D.nearest.get(fips5, '')}, about {r.mi_to_store:.0f} miles away.")

    med_store = D.stores
    rows = [
        ("Population", f"{int(r.total_population):,}", f"{int(med_store.total_population.median()):,}",
         f"{int(D.df.total_population.median()):,}"),
        ("Bachelor's degree or higher", f"{r.college_pct:.0f}%", f"{med_store.college_pct.median():.0f}%",
         f"{D.df.college_pct.median():.0f}%"),
        ("Median household income", f"${r.median_hh_inc:,.0f}" if pd.notna(r.median_hh_inc) else "n/a",
         f"${med_store.median_hh_inc.median():,.0f}", f"${D.df.median_hh_inc.median():,.0f}"),
    ]
    for f, lab in [("foreignborn_pct", "Born abroad"), ("age29andunder_pct", "Age 29 and under"),
                   ("age65andolder_pct", "Age 65 and over")]:
        if f in D.df.columns:
            rows.append((lab, f"{r[f]:.0f}%", f"{med_store[f].median():.0f}%", f"{D.df[f].median():.0f}%"))
    table = "".join(f"<tr><td>{a}</td><td class='n'><b>{b}</b></td><td class='n'>{c}</td><td class='n'>{d}</td></tr>"
                    for a, b, c, d in rows)

    drivers = ""
    if D.meta.get("model") == "lr":
        con = _contributions(D, r).drop(labels=["rural_pct"], errors="ignore")

        def phrase(k):
            above = r[k] > D.df[k].mean()
            if k == "log_pop":
                return "large population" if above else "small population"
            return ("high " if above else "low ") + FEATURE_LABELS[k]

        ups = [phrase(k) for k, v in con[::-1].items() if v > 0.15][:3]
        downs = [phrase(k) for k, v in con.items() if v < -0.15][:3]
        bits = []
        if ups:
            bits.append(f"<b>Raises its score:</b> {_join(ups)}.")
        if downs:
            bits.append(f"<b>Holds it back:</b> {_join(downs)}.")
        if bits:
            drivers = ("<h3>What drives this county's score</h3><p>Compared with the average US county. "
                       + " ".join(bits) + "</p>")

    title = r.label + (f" <span class='muted' style='font-size:12pt'>{_metro(r.label)}</span>"
                       if _metro(r.label) else "")
    return f"""
<section class="page">
  <div class="eyebrow">County spotlight</div>
  <h2>{title}</h2>
  <div class="kpis" style="margin:10px 0 16px">
    <div><b>{_pct(r.p)}</b><span>score</span></div>
    <div><b style="font-size:15pt;line-height:1.5">{D.meta['score_bin_labels'][int(np.clip(np.digitize(r.p, D.meta['score_bins'][1:-1]), 0, 6))]}</b><span>how close a match</span></div>
    <div><b>#{int(r['rank']):,}</b><span>of {len(D.df):,} counties</span></div>
    <div><b>{int(r.stores)}</b><span>Le Labo boutiques today</span></div>
  </div>
  <p class="lead">{r.label} is {seg_txt}.{open_rank}{near}</p>
  <h3>How it compares</h3>
  <table><tr><th></th><th class="n">{r.label}</th><th class="n">Median boutique county</th><th class="n">Median US county</th></tr>{table}</table>
  {drivers}
  <p class="small" style="margin-top:12px">Score: how closely the county resembles counties that already
  have a boutique, not a sales forecast. Demographics are ACS 2012&ndash;2016 estimates.</p>
  <div class="foot"><span>Le Labo US whitespace analysis</span><span>{page_no}</span></div>
</section>"""


################################################################################
# Page assembly
################################################################################
CSS = """
@page { size: Letter; margin: 0 }
:root { --ink:#1c1d1f; --ink2:#55585d; --rule:#dcdcd6; --paper:#f5f5f2; --blue:#1f5f8b;
  --s0:#efe9dc; --s1:#e8d3a8; --s2:#ddb46b; --s3:#c98f35; --s4:#a8681a; --s5:#7a440c; --s6:#4a2705;
  --mono:'Courier Prime', 'Courier New', monospace; --sans:'Libre Franklin', Inter, Arial, sans-serif }
* { box-sizing:border-box; margin:0; padding:0 }
body { font-family:var(--sans); color:var(--ink); font-size:10.2pt; line-height:1.5;
  -webkit-print-color-adjust:exact; print-color-adjust:exact }
.page { width:8.5in; height:11in; padding:0.7in 0.75in 0.6in; position:relative; overflow:hidden;
  page-break-after:always; background:#fff }
.page:last-child { page-break-after:auto }
.foot { position:absolute; left:0.75in; right:0.75in; bottom:0.38in; display:flex;
  justify-content:space-between; font-family:var(--mono); font-size:7.5pt; color:var(--ink2);
  border-top:1px solid var(--rule); padding-top:6px; letter-spacing:.04em }
.eyebrow { font-family:var(--mono); font-size:8pt; letter-spacing:.16em; text-transform:uppercase; color:var(--ink2) }
h1 { font-family:var(--mono); font-weight:700; font-size:34pt; line-height:1.05; letter-spacing:-.01em }
h2 { font-family:var(--mono); font-weight:700; font-size:19pt; line-height:1.15; margin:6px 0 10px }
h3 { font-family:var(--mono); font-weight:700; font-size:11.5pt; margin:16px 0 6px }
p { margin:0 0 9px }
.lead { font-size:11.5pt; line-height:1.55 }
.muted { color:var(--ink2) }
.small { font-size:8.6pt; color:var(--ink2); line-height:1.45 }
b { font-weight:600 }
.cover { background:var(--paper); padding:0; display:flex; flex-direction:column }
.cover .band { height:3.15in; background:#4a2705; color:#f5efe3; padding:0 0.75in 0.42in;
  display:flex; flex-direction:column; justify-content:flex-end }
.cover .band .eyebrow { color:#e8d3a8; margin-bottom:12px }
.cover .band h1 { color:#fff; font-size:36pt }
.cover .body { padding:0.42in 0.75in 0 }
.cover img.map { width:auto; max-width:100%; max-height:3.9in; display:block; margin:6px auto 14px; border:1px solid var(--rule); background:#fff }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); border-top:1.5px solid var(--ink); border-bottom:1px solid var(--rule) }
.kpis div { padding:10px 10px 10px 0 }
.kpis b { display:block; font-family:var(--mono); font-size:22pt; font-weight:700; line-height:1.1 }
.kpis span { font-size:8.4pt; color:var(--ink2); line-height:1.3; display:block; margin-top:2px }
/* In normal flow, pushed to the bottom: it can never overlap the KPI row,
   whatever fonts the viewer's machine substitutes. */
.byline { margin:auto 0.75in 0.5in; padding-top:14px; display:flex;
  justify-content:space-between; align-items:flex-end; font-size:9.5pt }
.byline .who b { font-family:var(--mono); font-size:11pt }
.find { display:grid; grid-template-columns:28px 1fr; gap:4px 12px; margin:14px 0 4px }
.find .num { font-family:var(--mono); font-weight:700; font-size:15pt; color:var(--s4); line-height:1.1 }
.find h4 { font-weight:700; font-size:11pt; margin-bottom:3px }
.find p { margin:0 0 6px; color:#2c2d30 }
.callout { background:var(--paper); border-left:3px solid var(--s4); padding:12px 16px; margin:16px 0 0 }
.callout h4 { font-family:var(--mono); font-size:10.5pt; margin-bottom:4px }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:9.3pt }
th { font-family:var(--mono); font-weight:400; font-size:7.6pt; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink2); text-align:left; padding:5px 6px; border-bottom:1.2px solid var(--ink) }
th.n, td.n { text-align:right }
td { padding:5px 6px; border-bottom:1px solid var(--rule); vertical-align:top }
td .sub, td.sub { font-size:7.8pt; color:var(--ink2); line-height:1.25 }
td.rk { font-family:var(--mono); font-weight:700; color:var(--blue); width:22px }
.bar { display:inline-block; height:7px; background:var(--s4); margin-right:6px; vertical-align:1px }
.sc { font-family:var(--mono); font-size:8.8pt }
.legend { display:flex; flex-wrap:wrap; gap:6px 22px; align-items:center; font-size:8.5pt; color:var(--ink2); margin-top:8px }
.ramp { display:flex; align-items:center; gap:4px }
.ramp i { display:inline-block; width:22px; height:9px }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--ink); margin-right:5px }
.ring { display:inline-block; width:11px; height:11px; border-radius:50%; border:2px solid var(--blue); margin-right:5px; vertical-align:-2px }
.coef { display:grid; grid-template-columns:170px 1fr; gap:7px 12px; align-items:center; font-size:9.5pt }
.track { position:relative; height:12px }
.track::before { content:""; position:absolute; left:50%; top:-4px; bottom:-4px; border-left:1px solid var(--ink2) }
.track i { position:absolute; top:0; height:12px; border-radius:1px }
.track i.pos { background:var(--s4) } .track i.neg { background:#8a8d92 }
.axisl { display:grid; grid-template-columns:170px 1fr 1fr; gap:0 12px; font-size:7.8pt; color:var(--ink2); margin-bottom:6px; font-family:var(--mono) }
.loo { display:grid; grid-template-columns:120px 1fr; gap:1.5px 10px; align-items:center; font-size:8.6pt; position:relative }
.lt { position:relative; height:10px; display:flex; align-items:center }
.lt i { display:block; height:9px } .lt i.hit { background:var(--s5) } .lt i.miss { background:#b9b6ad }
.lt span { font-family:var(--mono); font-size:7.8pt; margin-left:5px; color:var(--ink2) }
.lgrid { position:absolute; top:0; bottom:0; left:130px; right:0; pointer-events:none }
.lgrid div { position:absolute; top:-14px; bottom:0; border-left:1px dashed var(--blue) }
.lgrid span { position:absolute; top:-15px; left:4px; font-family:var(--mono); font-size:7.5pt; color:var(--blue); white-space:nowrap }
.two { display:grid; grid-template-columns:1fr 1fr; gap:26px }
.steps { counter-reset:s }
.step { display:grid; grid-template-columns:30px 1fr; gap:0 10px; margin-bottom:11px }
.step::before { counter-increment:s; content:counter(s); font-family:var(--mono); font-weight:700; color:#fff;
  background:var(--s5); width:24px; height:24px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:10.5pt }
.step h4 { font-size:10.5pt; font-weight:700; margin-bottom:2px }
.step p { margin:0; color:#2c2d30 }
ul.clean { list-style:none }
ul.clean li { padding-left:14px; position:relative; margin-bottom:6px }
ul.clean li::before { content:""; position:absolute; left:0; top:7px; width:6px; height:6px; background:var(--s4) }
.mtab td, .mtab th { font-size:8.8pt; padding:3.5px 6px }
"""


def build_html(data_dir, map_png, top_n=12, county=None, author="Leon Shpaner, M.S.",
               org="Data Science Dynamics", site="datasciencedynamics.com"):
    """Full report HTML. top_n sets the rows in the open-market table (max 15);
    county (5-digit FIPS string) adds a County spotlight page."""
    D = _Data(data_dir)
    m = D.meta
    top_n = int(max(5, min(TOP_N_MAX, top_n or 12)))
    top = D.open_all.head(top_n)
    D.fill = D.fill_all.head(max(5, 20 - top_n) if top_n > 12 else 8)  # keep page 4 on one page
    top5 = D.open_all.head(5)
    n_all, n_sc, n_st = len(D.df), int((D.df.stores > 0).sum()), int(D.df.stores.sum())
    radius = int(m["radius_mi"])
    med_store = D.stores.p.median()
    month = pd.Timestamp(m["generated"]).strftime("%B %Y")
    map_uri = f"data:image/png;base64,{_b64(map_png)}"
    coef = dict(zip(D.coef.feature, D.coef.coef))

    # ── computed findings ────────────────────────────────────────────────
    top5_txt = _join(f"{r.label.split(',')[0]}" + (f" ({_metro(r.label)})" if _metro(r.label) else "")
                     for r in top5.itertuples())
    dist_lo, dist_hi = top5.mi_to_store.min(), top5.mi_to_store.max()

    top_regions = top.label.map(lambda l: REGION.get(_abbr(l), "Other")).value_counts()
    store_regions = D.stores_raw.state.map(lambda s: REGION.get(s, "Other")).value_counts()
    reg_a = top_regions.index[0]
    reg_b = top_regions.index[1] if len(top_regions) > 1 else None
    reg_names = reg_a + (f" and {reg_b}" if reg_b else "")
    reg_n = int(top_regions.iloc[:2].sum())
    reg_st = int(store_regions.get(reg_a, 0) + (store_regions.get(reg_b, 0) if reg_b else 0))

    fill_names = _join(dict.fromkeys((_metro(r.label) or r.label) for r in D.fill.head(6).itertuples()))
    fill_above = int((D.fill.p > med_store).sum())

    edu_finding = ""
    if coef.get("college_pct", 0) > 0 and coef.get("log_income", 0) < 0:
        edu_finding = """<div class="find"><div class="num">4</div><div>
    <h4>Education predicts a Le Labo market better than wealth</h4>
    <p>After accounting for size, the share of residents with a bachelor's degree raises a county's fit,
    while median income does not. Le Labo's existing markets are educated urban cores rather than the
    wealthiest suburbs.</p></div></div>"""
    loo_finding = ""
    if D.loo is not None:
        ones = D.loo[D.loo.whitespace_rank == 1].county.map(
            lambda c: {"Kings, NY": "Brooklyn", "New York": "Manhattan"}.get(c, c.split(",")[0])).tolist()
        loo_finding = f"""<div class="find"><div class="num">{5 if edu_finding else 4}</div><div>
    <h4>The method works on Le Labo's own history</h4>
    <p>When each existing boutique county was hidden from the model, it still ranked {m['loo_top25']} of the
    {n_sc} in its top 25 out of {m['loo_candidates']:,} counties.{(' ' + _join(ones) + (' each' if len(ones) > 1 else '') + ' came back at #1.') if ones else ''}</p></div></div>"""

    far = top.sort_values("mi_to_store", ascending=False).head(3)
    far_txt = _join(f"{_metro(r.label) or r.label}" for r in far.itertuples())
    n10 = int((D.df.p >= 0.10).sum())

    low = D.stores.sort_values("p").head(4)
    low_txt = _join(
        f"{r.label} ({_pct(r.p)}" + (f", {D.store_names.get(int(r.fips), [''])[0]}" if int(r.fips) in D.store_names else "") + ")"
        for r in low.itertuples())

    prof = ["<li><b>Size comes first.</b> Every current boutique county is a large one, and population "
            "carries most of the ranking.</li>"]
    if coef.get("college_pct", 0) > 0 and coef.get("log_income", 0) < 0:
        prof.append("<li><b>Education beats income.</b> Among counties of similar size, more college graduates "
                    "means a better fit, while higher median income slightly lowers it.</li>")
    if coef.get("foreignborn_pct", 0) > 0 and coef.get("age65andolder_pct", 0) < 0:
        prof.append("<li><b>International and working-age.</b> A larger foreign-born share helps. High shares "
                    "of residents over 65, and to a lesser degree under 30, lower the fit.</li>")
    if coef.get("rural_pct", 0) > 0:
        prof.append("<li><b>Rural share</b> is a statistical side effect: at the same population, counties with "
                    "a dense core plus outlying land score higher. It does not mean rural areas attract "
                    "boutiques.</li>")

    page = 1
    pages = []

    # ── 1. cover ─────────────────────────────────────────────────────────
    pages.append(f"""
<section class="page cover">
  <div class="band"><div class="eyebrow">US Boutique Whitespace Analysis &middot; {month}</div>
    <h1>Where Le Labo should open next</h1></div>
  <div class="body">
    <p class="lead">A county-by-county read of the United States, built from the {n_st} boutiques Le Labo
    already operates. It finds the places that look most like Le Labo's existing markets but still have no
    boutique nearby.</p>
    <img class="map" src="{map_uri}">
    <div class="kpis">
      <div><b>{n_all:,}</b><span>US counties scored</span></div>
      <div><b>{n_st}</b><span>boutiques in {n_sc} counties used to learn the profile</span></div>
      <div><b>{top_n}</b><span>open markets on the shortlist, none within {radius} miles of a boutique</span></div>
      <div><b>{m.get('loo_top25', '&mdash;')} of {n_sc}</b><span>existing markets found in a blind test</span></div>
    </div>
  </div>
  <div class="byline">
    <div class="who"><b>{author}</b><br><span class="muted">{org} &middot; {site}</span></div>
    <div class="small" style="text-align:right;max-width:3.2in">Independent analysis from public data.<br>Not commissioned by or affiliated with Le Labo.</div>
  </div>
</section>""")

    # ── 2. summary ───────────────────────────────────────────────────────
    page += 1
    pages.append(f"""
<section class="page">
  <div class="eyebrow">Summary</div>
  <h2>The short version</h2>
  <p class="lead">Le Labo's US boutiques follow a clear pattern: large, dense, highly educated and
  internationally mixed counties. Several big metros that fit that pattern closely still have no boutique
  within {radius} miles.</p>
  <div class="find"><div class="num">1</div><div>
    <h4>Five open markets stand out</h4>
    <p>{top5_txt} lead the list. Each matches the profile of Le Labo's current markets and sits between
    {dist_lo:.0f} and {dist_hi:.0f} miles from the nearest boutique.</p></div></div>
  <div class="find"><div class="num">2</div><div>
    <h4>The {reg_names} hold most of the opportunity</h4>
    <p>{reg_n} of the top {top_n} open markets are in the {reg_names}, regions that hold {reg_st} of Le Labo's
    {n_st} US boutiques today.</p></div></div>
  <div class="find"><div class="num">3</div><div>
    <h4>Existing metros still have room</h4>
    <p>{fill_names} lead the fill-in markets: strong matches that already sit near a boutique.
    {fill_above} of the top {len(D.fill)} score above the median current boutique county ({_pct(med_store)}),
    so another door in a metro Le Labo already serves would reach these residents directly.</p></div></div>
  {edu_finding}
  {loo_finding}
  <div class="callout"><h4>What this is, and what it isn't</h4>
    <p style="margin:0">The score measures how closely a county resembles places Le Labo already operates.
    It is a shortlist for site research, not a sales forecast. With Le Labo's own store performance and a
    neighborhood-level view, the same approach can rank specific trade areas and estimate revenue
    potential.</p></div>
  <div class="foot"><span>Le Labo US whitespace analysis</span><span>{page}</span></div>
</section>""")

    # ── 3. map ───────────────────────────────────────────────────────────
    page += 1
    pages.append(f"""
<section class="page">
  <div class="eyebrow">The national picture</div>
  <h2>How every US county matches Le Labo's markets</h2>
  <p>Each county is shaded by its score: how closely it resembles the counties where Le Labo already has a
  boutique. Pale means little resemblance; dark brown means a close match. Black dots are current
  boutiques. Blue rings are the strongest <b>open markets</b>, counties with no boutique within {radius}
  miles, numbered by rank.</p>
  <img src="{map_uri}" style="width:100%;display:block;margin-top:8px;border:1px solid var(--rule)">
  <div class="legend">
    <span class="ramp">Low match <i style="background:var(--s0)"></i><i style="background:var(--s1)"></i><i style="background:var(--s2)"></i><i style="background:var(--s3)"></i><i style="background:var(--s4)"></i><i style="background:var(--s5)"></i><i style="background:var(--s6)"></i> Close match</span>
    <span><span class="dot"></span>Has a boutique</span><span><span class="ring"></span>Top open market</span>
  </div>
  <h3>What the map shows</h3>
  <ul class="clean">
    <li>{n10} of {n_all:,} counties score 10% or higher. Le Labo's profile is narrow, which makes the
    shortlist meaningful rather than a list of every big city.</li>
    <li>The farthest-from-a-boutique markets on the shortlist are {far_txt}, each {far.mi_to_store.min():.0f}
    miles or more from the nearest boutique.</li>
    <li>{reg_n} of the top {top_n} open markets are in the {reg_names}.</li>
  </ul>
  <div class="foot"><span>Le Labo US whitespace analysis</span><span>{page}</span></div>
</section>""")

    # ── 4. shortlist ─────────────────────────────────────────────────────
    page += 1
    pages.append(f"""
<section class="page">
  <div class="eyebrow">The shortlist</div>
  <h2>Where to look next</h2>
  <h3 style="margin-top:4px">Open markets: no boutique within {radius} miles</h3>
  <table><tr><th>#</th><th>County</th><th class="n">Pop.</th><th class="n">Bachelor's+</th><th>Score</th><th class="n">Miles</th><th>Nearest boutique</th></tr>
  {_rows_open(D, top)}</table>
  <h3>Fill-in markets: strong fit, already near a boutique</h3>
  <table><tr><th>County</th><th class="n">Pop.</th><th class="n">Bachelor's+</th><th>Score</th><th class="n">Miles to nearest boutique</th></tr>
  {_rows_fill(D)}</table>
  <p class="small" style="margin-top:10px"><b>Bachelor's+</b> is the share of residents with a bachelor's
  degree or higher. <b>Score</b> is how closely the county matches counties that already have a boutique;
  for reference, the median current boutique county scores {_pct(med_store)}. Population and education are
  ACS 2012&ndash;2016 estimates.</p>
  <div class="foot"><span>Le Labo US whitespace analysis</span><span>{page}</span></div>
</section>""")

    # ── optional spotlight ───────────────────────────────────────────────
    if county:
        sp = _spotlight(D, county, page + 1)
        if sp:
            page += 1
            pages.append(sp)

    # ── 5. profile ───────────────────────────────────────────────────────
    page += 1
    profile_bullets = "".join(prof) if m.get("model") == "lr" else ""
    pages.append(f"""
<section class="page">
  <div class="eyebrow">The profile</div>
  <h2>What makes a county look like a Le Labo market</h2>
  <p>The model compares the {n_sc} boutique counties with every other county on eight Census traits. Bars to
  the right make a county look more like Le Labo's existing markets; bars to the left, less. Longer bars
  matter more, and each holds the other traits fixed.</p>
  <div class="axisl"><span></span><span>&larr; less like existing markets</span><span style="text-align:right">more like existing markets &rarr;</span></div>
  <div class="coef">{_coef_rows(D)}</div>
  {('<h3>Reading the profile</h3><ul class="clean">' + profile_bullets + '</ul>') if profile_bullets else ''}
  <h3>Where Le Labo already breaks its own pattern</h3>
  <p>The lowest-scoring current boutique counties are {low_txt}. Low scores mark boutiques placed by a
  different logic than the county profile, such as destination, resort or suburban-center locations.
  Neighborhood-level data and Le Labo's own sales figures would let a next version find more of these
  sites.</p>
  <div class="foot"><span>Le Labo US whitespace analysis</span><span>{page}</span></div>
</section>""")

    # ── 6. validation ────────────────────────────────────────────────────
    if D.loo is not None:
        page += 1
        loo_html, t25 = _loo_rows(D)
        sens_html = ""
        if D.sens is not None:
            cols = list(D.sens.columns[1:])
            head = "".join(f"<th>{c}{' (used here)' if c.startswith(str(radius) + ' ') else ''}</th>" for c in cols)
            body = "".join(f"<tr><td class='rk'>{r[0]}</td>" + "".join(f"<td>{v}</td>" for v in r[1:]) + "</tr>"
                           for r in D.sens.head(5).itertuples(index=False))
            sens_html = f"""<h3>The shortlist is stable</h3>
  <p>Changing how far a boutique's reach extends reshuffles the order but keeps the same markets near the top.</p>
  <table class="mtab"><tr><th>#</th>{head}</tr>{body}</table>"""
        pages.append(f"""
<section class="page">
  <div class="eyebrow">Validation</div>
  <h2>Does it work? A blind test on Le Labo's own markets</h2>
  <p>Each existing boutique county was hidden from the model, one at a time, and the model ranked every
  county without a boutique. Each bar shows where the hidden county landed (shorter is better; 1 is the top
  of the list out of {m['loo_candidates']:,}). If the model only memorized Le Labo's current map, hidden
  markets would land at random.</p>
  <div class="loo" style="margin-top:30px"><div class="lgrid"><div style="left:{t25:.1f}%"><span>top 25</span></div></div>{loo_html}</div>
  <p style="margin-top:12px"><b>{m['loo_top25']} of {n_sc}</b> existing markets landed in the top 25; the
  median landed at #{m['loo_median_rank']}.</p>
  {sens_html}
  <div class="foot"><span>Le Labo US whitespace analysis</span><span>{page}</span></div>
</section>""")

    # ── 7. next steps ────────────────────────────────────────────────────
    page += 1
    pages.append(f"""
<section class="page">
  <div class="eyebrow">Next steps</div>
  <h2>From a county shortlist to specific sites</h2>
  <p class="lead">This analysis uses only public data. Working with Le Labo's own data would turn a
  county-level shortlist into ranked trade areas with revenue estimates.</p>
  <div class="steps" style="margin-top:14px">
    <div class="step"><div><h4>Confirm the footprint</h4><p>Rerun on Le Labo's official US store list,
    including counters and recent openings. The full pipeline reruns within a day.</p></div></div>
    <div class="step"><div><h4>Go to neighborhood level</h4><p>Move from counties to Census tracts and
    shopping districts, with current ACS estimates, so the model can find specific districts inside a metro
    rather than only the metro itself.</p></div></div>
    <div class="step"><div><h4>Learn from performance, not just presence</h4><p>With boutique sales or
    traffic, the model predicts how well a new door would do, not only whether a market resembles existing
    ones.</p></div></div>
    <div class="step"><div><h4>Add the competitive set and demand signals</h4><p>Layer in nearby luxury
    fragrance and beauty retail, e-commerce orders by ZIP code, foot traffic and tourism.</p></div></div>
    <div class="step"><div><h4>Deliver a living tool</h4><p>An interactive map and dashboard the real
    estate team can filter, update each quarter and use to compare candidate sites.</p></div></div>
  </div>
  <div class="two" style="margin-top:8px">
    <div><h3>Limits of this version</h3><ul class="clean small" style="font-size:8.8pt">
      <li>Boutique list compiled from public listings; department-store counters excluded and a few
      boutiques may be missing.</li>
      <li>Demographics are 2012&ndash;2016 ACS averages and predate recent moves.</li>
      <li>Counties are large and varied inside; a strong county is a place to look, not a site.</li>
      <li>The profile repeats Le Labo's past choices and can miss markets the brand never tried.</li></ul></div>
    <div><h3>Contact</h3><p><b style="font-family:var(--mono);font-size:11pt">{author}</b><br>{org}<br>
      <span class="muted">{site}</span></p>
      <p class="small">An interactive version of the map, with county lookup and the full model results, is
      available on request.</p></div>
  </div>
  <div class="foot"><span>Le Labo US whitespace analysis</span><span>{page}</span></div>
</section>""")

    # ── 8. appendix ──────────────────────────────────────────────────────
    page += 1
    auc = dict(zip(D.ci.metric, D.ci.estimate))
    big_txt = ""
    if "ROC-AUC, counties 500k+" in auc:
        big_txt = (f"<p><b>Why large counties.</b> Every boutique county is a large urban one, so across all "
                   f"counties population alone separates them almost perfectly. The meaningful test is among the "
                   f"{m['n_large_counties']} counties of 500,000 or more, where the model reaches an AUC of "
                   f"{auc['ROC-AUC, counties 500k+']:.2f} against "
                   f"{auc.get('ROC-AUC, counties 500k+, population alone', float('nan')):.2f} for population alone.</p>")
    loo_rows = ""
    if D.loo is not None:
        loo_rows = (f"<tr><td>Blind test: median rank of hidden boutique county (of {m['loo_candidates']:,})</td>"
                    f"<td class='n'>{m['loo_median_rank']}</td><td class='n'>&mdash;</td></tr>"
                    f"<tr><td>Blind test: hidden boutique counties ranked in top 25</td>"
                    f"<td class='n'>{m['loo_top25']} of {n_sc}</td><td class='n'>&mdash;</td></tr>")
    pages.append(f"""
<section class="page">
  <div class="eyebrow">Appendix</div>
  <h2>Method and model performance</h2>
  <p><b>Outcome.</b> Whether a county hosts at least one standalone Le Labo boutique ({n_st} boutiques in
  {n_sc} counties).</p>
  <p><b>Data.</b> {n_all:,} US counties. Demographics from the American Community Survey 2012&ndash;2016
  five-year estimates (MIT Election Data and Science Lab county file), rural share from the 2010 Census,
  land area and boundaries from the Census Bureau. Eight features: log population, log density, log median
  household income, bachelor's degree share, foreign-born share, share age 29 and under, share age 65 and
  over, and rural share.</p>
  <p><b>Model.</b> {m['model_name']}, tuned on a stratified 60/20/20 split for average precision, chosen
  among logistic regression, random forest, XGBoost and CatBoost on out-of-fold average precision.</p>
  <p><b>Scores.</b> Every county is scored out-of-fold: the tuned model is refit over {m['cv_repeats']}
  repeats of stratified 5-fold cross-validation with sigmoid calibration inside each fold, and the
  predictions are averaged. No county is scored by a model that saw its own label.</p>
  {big_txt}
  <h3>Out-of-fold performance with 95% bootstrap intervals</h3>
  <table class="mtab"><tr><th>Metric</th><th class="n">Estimate</th><th class="n">95% CI</th></tr>
  {_metric_rows(D)}{loo_rows}</table>
  <p class="small" style="margin-top:8px">ROC-AUC: how well scores separate boutique from non-boutique
  counties (0.5 is chance, 1 is perfect). Average precision: how concentrated boutique counties are at the
  top of the ranking (base rate {n_sc / n_all:.3f}). Brier score: mean squared error of the probabilities
  (lower is better). Intervals from {m['n_bootstrap']:,} stratified bootstrap resamples.</p>
  <h3>Store sources</h3>
  <p class="small">{m['store_note'].replace(' (bundled fallback list)', '')} Analysis code: Python,
  scikit-learn, model_tuner and MLflow.</p>
  <div class="foot"><span>Le Labo US whitespace analysis</span><span>{page}</span></div>
</section>""")

    return (f'<!doctype html><html><head><meta charset="utf-8"><title>Le Labo US Whitespace Analysis</title>'
            f'{FONT_LINK}<style>{CSS}</style></head><body>{"".join(pages)}</body></html>')


################################################################################
# Rendering (Playwright + Chromium)
################################################################################
def _launch(p):
    import glob
    import os
    exe = os.environ.get("CHROMIUM_PATH") or next(
        iter(sorted(glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome"))), None)
    return p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()


def _apply_routes(pg, route):
    # Playwright passes (route, request) to handlers with two parameters, so the
    # handler must take exactly one: build each in its own closure.
    def handler(body, ctype):
        return lambda r: r.fulfill(body=body, content_type=ctype)
    for pattern, (body, ctype) in (route or {}).items():
        pg.route(pattern, handler(body, ctype))


def html_to_pdf(html, pdf_path, route=None):
    """Print the report HTML to a Letter PDF. `route` is an optional
    {url_glob: (body, content_type)} map for offline rendering."""
    from playwright.sync_api import sync_playwright

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = _launch(p)
        pg = b.new_page()
        _apply_routes(pg, route)
        pg.set_content(html, wait_until="load")
        pg.evaluate("document.fonts.ready.then(() => true)")
        pg.pdf(path=str(pdf_path), format="Letter", print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    return pdf_path


def capture_map(report_html, png_path, route=None):
    """Screenshot the county map from the rendered HTML report at print resolution."""
    from playwright.sync_api import sync_playwright

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = _launch(p)
        pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=3,
                        color_scheme="light")
        _apply_routes(pg, route)
        pg.goto(Path(report_html).resolve().as_uri(), wait_until="load")
        pg.wait_for_selector("#map path.c")
        pg.add_style_tag(content="#map{background:#fff}")
        pg.locator("#map").screenshot(path=str(png_path))
        b.close()
    return png_path
