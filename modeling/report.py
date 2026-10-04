"""
report.py
---------
Renders the self-contained HTML whitespace map from the county scores
(score_counties.py) and county data.

Example usage:
    python modeling/report.py --output-file ./reports/lelabo_whitespace.html
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from core.config import (
    EVAL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    TEMPLATES_DIR,
    radius_mi,
    report_authors,
    top_n_fill,
    top_n_open,
)
from core.constants import (
    coef_file,
    counties_geojson,
    labels_file,
    meta_file,
    metrics_file,
    oof_scores_file,
    report_file,
    store_count,
    stores_source,
    target_outcome,
)
from core.functions import load_geojson, segment_counties, slim_geojson

app = typer.Typer(add_completion=False)

TABLE_COLS = ["label", "county", "state", "total_population", "median_hh_inc",
              "college_pct", "p", "mi_to_store", store_count]

MODEL_NAMES = {
    "lr": ("logistic regression", "L2-penalized logistic regression"),
    "rf": ("random forest", "Random forest"),
    "xgb": ("XGBoost model", "XGBoost gradient boosting"),
    "cat": ("CatBoost model", "CatBoost gradient boosting"),
}


PLAIN = {
    "log_pop": "population",
    "log_density": "population density",
    "log_income": "median household income",
    "college_pct": "college-degree share",
    "foreignborn_pct": "foreign-born share",
    "age29andunder_pct": "share age 29 and under",
    "age65andolder_pct": "share age 65 and over",
    "rural_pct": "rural share",
}


def model_summary(coef: pd.Series, model: str, metrics: pd.Series) -> str:
    """Technical read of the fitted model for the modal (moved off the main page)."""
    c = coef.sort_values(ascending=False)
    vals = ", ".join(f"{PLAIN.get(k, k)} {v:+.2f}" for k, v in c.items())
    if model != "lr":
        return (f"Feature importances (unsigned): {vals}. Importances show which traits the "
                "model relies on, not the direction of their effect.")
    top = PLAIN.get(c.index[0], c.index[0])
    parts = [
        f"Standardized logistic-regression coefficients (log-odds per standard deviation): {vals}.",
        f"{top.capitalize()} dominates: Le Labo sits in the largest metros, and most of the "
        "ranking follows from size.",
    ]
    if c.get("log_income", 0) < 0 < c.get("college_pct", 0):
        parts.append("Income enters negatively once college share is held fixed: among "
                     "counties of equal size and education, the model favors the less affluent "
                     "one. Income and education are correlated, so read this as education "
                     "carrying the signal rather than income repelling stores.")
    if c.get("rural_pct", 0) > 0:
        parts.append("The positive rural share is a conditional effect: at a given population "
                     "and density, it most likely picks up counties with a dense core plus outlying "
                     "land (common among large western counties). It does not mean rural areas "
                     "attract boutiques.")
    parts.append("Coefficients describe association with current store locations, not causes "
                 "of store performance.")
    return " ".join(parts)


@app.command()
def main(
    outcome: str = target_outcome[0],
    output_file: Path = REPORTS_DIR / report_file,
    radius: float = radius_mi,
    template_file: Path = TEMPLATES_DIR / "report_template.html",
) -> None:
    """Build reports/lelabo_whitespace.html."""

    ############################################################################
    # Step 1. Load county scores, model metadata and county data
    ############################################################################
    results_dir = RESULTS_DIR / outcome
    scores = pd.read_parquet(results_dir / oof_scores_file)["p"]
    coef = pd.read_csv(results_dir / coef_file, index_col=0)["coef"]
    metrics = pd.read_csv(results_dir / metrics_file, index_col=0)["value"]
    labels = pd.read_parquet(PROCESSED_DATA_DIR / labels_file)[target_outcome]
    meta = pd.read_parquet(PROCESSED_DATA_DIR / meta_file)
    note = (PROCESSED_DATA_DIR / stores_source).read_text(encoding="utf-8")
    geo = load_geojson(RAW_DATA_DIR / counties_geojson)

    scored = segment_counties(scores, labels, meta, radius, target_outcome[0], store_count)


    ############################################################################
    # Step 2. Model description for the page
    ############################################################################
    model = str(metrics["model"])
    long = MODEL_NAMES.get(model, (model, model))[1]
    repeats = int(float(metrics["cv_repeats"]))
    ############################################################################
    # Step 2b. Model info modal: header, how-it-works, metrics, hyperparameters
    ############################################################################
    ci = pd.read_csv(results_dir / "oof_metrics_ci.csv")
    hyper = pd.read_csv(results_dir / "hyperparameters.csv", index_col=0)["value"]

    def n(key):
        try:
            return int(float(metrics[key]))
        except (KeyError, ValueError):
            return 0

    rows = []
    for _, r in ci.iterrows():
        digits = 4 if r.metric.startswith("Brier") else 3

        def fmt(v, d=digits):
            return f"{v:.{d}f}"

        sep = ' class="sep"' if r.metric.startswith("ROC-AUC, counties 500k+") and \
            "population" not in r.metric else ""
        rows.append(f'<tr{sep}><td>{r.metric}</td><td class="n">{fmt(r.estimate)}</td>'
                    f'<td class="n">({fmt(r.ci_low)}, {fmt(r.ci_high)})</td></tr>')
    loo_file = EVAL_DIR / outcome / f"{model}_validation_loo.csv"
    if loo_file.exists():
        loo = pd.read_csv(loo_file)
        rk = loo.whitespace_rank
        rows.append(f'<tr class="sep"><td>Leave-one-out median rank (of {int(loo.of_candidates.iloc[0]):,})</td>'
                    f'<td class="n">{int(rk.median())}</td><td class="n">&mdash;</td></tr>')
        rows.append(f'<tr><td>Store counties ranked in top 25 when held out</td>'
                    f'<td class="n">{int((rk <= 25).sum())} of {len(rk)}</td><td class="n">&mdash;</td></tr>')
    metric_note = (
        f"Out-of-fold scores for all {len(scored):,} counties ({int(labels[target_outcome[0]].sum())} "
        f"store counties). Confidence intervals from {n('n_bootstrap'):,} stratified bootstrap "
        "resamples, percentile method. All metrics are threshold-independent: the map ranks "
        "counties and applies no cut-off. Counties 500k+ (n = "
        f"{int((meta.total_population >= 500_000).sum())}) is the subgroup where the population "
        "baseline no longer separates store counties on its own."
    )
    hyper_rows = "".join(
        f'<tr><td>{k}</td><td class="n">{v}</td></tr>' for k, v in hyper.items()
    ) + ('<tr><td>out-of-fold calibration</td><td class="n">sigmoid, 3-fold inner CV</td></tr>')

    split = ""
    if n("n test"):
        split = (f"60/20/20 split (Train/Validation/Test) for tuning: "
                 f"{n('n train'):,} / {n('n valid'):,} / {n('n test'):,} counties")
    how = {
        "lr": "L2-penalized logistic regression with balanced class weights, on eight "
              "standardized county features (log population, log density, log median income, "
              "college share, foreign-born share, age 29 and under, age 65 and over, rural share). "
              "Each coefficient reads directly as the direction and strength of a feature's pull.",
        "rf": "A random forest with balanced class weights on eight county features.",
        "xgb": "XGBoost gradient-boosted trees on eight county features, refit at the tuned "
               "number of trees.",
        "cat": "CatBoost gradient-boosted trees on eight county features, refit at the tuned "
               "number of trees.",
    }.get(model, model)

    coef_note = (
        "Bars to the right mark traits that make a county look more like the places "
        "Le Labo already operates; bars to the left, less like them. Longer bars matter more. "
        "Each bar holds the other traits fixed."
        if model == "lr"
        else "The traits the model leans on most. Longer bars matter more."
    )
    model_learned = model_summary(coef, model, metrics)
    loo_top25 = (
        str(int((pd.read_csv(loo_file).whitespace_rank <= 25).sum())) if loo_file.exists() else "n/a"
    )

    ############################################################################
    # Step 3. Tables embedded in the page
    ############################################################################
    tables = {
        "open": scored[scored.segment == "open"].head(top_n_open)[TABLE_COLS].to_dict("records"),
        "fill": scored[scored.segment == "fill"].head(top_n_fill)[TABLE_COLS].to_dict("records"),
        "stores": scored[scored.segment == "store"][TABLE_COLS].to_dict("records"),
        "coef": coef.round(3).to_dict(),
    }

    ############################################################################
    # Step 4. Fill the template
    ############################################################################
    subs = {
        "__GEO__": json.dumps(slim_geojson(geo, scored, store_count), separators=(",", ":")),
        "__TABLES__": json.dumps(tables, default=float),
        "__MONTH__": date.today().strftime("%b %Y"),
        "__N_STORES__": str(int(scored[store_count].sum())),
        "__N_COUNTIES__": str(int((scored[store_count] > 0).sum())),
        "__N_ALL__": f"{len(scored):,}",
        "__RADIUS__": f"{radius:g}",
        "__STORE_NOTE__": note,
        "__COEF_NOTE__": coef_note,
        "__MODEL_LEARNED__": model_learned,
        "__LOO_TOP25__": loo_top25,
        "__YEAR__": str(date.today().year),
        "__AUTHORS__": report_authors,
        "__M_MODEL__": long,
        "__M_DATA__": (f"{len(scored):,} US counties, ACS 2012&ndash;2016 demographics; "
                       f"{int(scored[store_count].sum())} boutiques in "
                       f"{int((scored[store_count] > 0).sum())} counties"),
        "__M_SPLIT__": split or "Tuned on model_tuner's stratified 60/20/20 split",
        "__M_HOW__": how,
        "__REPEATS__": str(repeats),
        "__METRIC_ROWS__": "\n        ".join(rows),
        "__METRIC_NOTE__": metric_note,
        "__HYPER_ROWS__": hyper_rows,
        "__RUN_NAME__": str(metrics.get("run_name", f"{model}_orig_training")),
    }
    html = template_file.read_text(encoding="utf-8")
    for k, v in subs.items():
        html = html.replace(k, v)

    import re
    left = sorted(set(re.findall(r"__[A-Z0-9_]+__", html)))
    if left:
        raise typer.BadParameter(
            f"Unfilled placeholders {left} in {template_file}. The template and "
            "modeling/report.py are out of sync; update both together."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    print(f"Map: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")
    logger.success("Report complete.")


if __name__ == "__main__":
    app()
