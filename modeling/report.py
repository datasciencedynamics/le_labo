"""
report.py
---------
Renders the self-contained HTML whitespace map from the training and
evaluation outputs.

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
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    TEMPLATES_DIR,
    radius_mi,
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


@app.command()
def main(
    output_file: Path = typer.Option(REPORTS_DIR / report_file, "--output-file"),
    radius: float = typer.Option(radius_mi, "--radius"),
    template_file: Path = typer.Option(
        TEMPLATES_DIR / "report_template.html", "--template-file"
    ),
) -> None:
    """Build reports/lelabo_whitespace.html."""

    ############################################################################
    # Step 1. Load model outputs and county data
    ############################################################################
    scores = pd.read_parquet(RESULTS_DIR / oof_scores_file)["p"]
    labels = pd.read_parquet(PROCESSED_DATA_DIR / labels_file)[[target_outcome, store_count]]
    meta = pd.read_parquet(PROCESSED_DATA_DIR / meta_file)
    coef = pd.read_csv(RESULTS_DIR / coef_file, index_col=0)["coef"]
    metrics = pd.read_csv(RESULTS_DIR / metrics_file, index_col=0)["value"]
    note = (PROCESSED_DATA_DIR / stores_source).read_text(encoding="utf-8")
    geo = load_geojson(RAW_DATA_DIR / counties_geojson)

    scored = segment_counties(scores, labels, meta, radius)

    ############################################################################
    # Step 2. Tables embedded in the page
    ############################################################################
    tables = {
        "open": scored[scored.segment == "open"].head(top_n_open)[TABLE_COLS].to_dict("records"),
        "fill": scored[scored.segment == "fill"].head(top_n_fill)[TABLE_COLS].to_dict("records"),
        "stores": scored[scored.segment == "store"][TABLE_COLS].to_dict("records"),
        "coef": coef.round(3).to_dict(),
    }

    ############################################################################
    # Step 3. Fill the template
    ############################################################################
    subs = {
        "__GEO__": json.dumps(slim_geojson(geo, scored), separators=(",", ":")),
        "__TABLES__": json.dumps(tables, default=float),
        "__MONTH__": date.today().strftime("%b %Y"),
        "__N_STORES__": str(int(scored[store_count].sum())),
        "__N_COUNTIES__": str(int((scored[store_count] > 0).sum())),
        "__N_ALL__": f"{len(scored):,}",
        "__AUC__": f"{float(metrics['cv_auc']):.2f}",
        "__AP__": f"{float(metrics['cv_average_precision']):.2f}",
        "__BASE__": f"{float(metrics['base_rate']):.3f}",
        "__RADIUS__": f"{radius:g}",
        "__STORE_NOTE__": note,
    }
    html = template_file.read_text(encoding="utf-8")
    for k, v in subs.items():
        html = html.replace(k, v)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    print(f"Map: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")
    logger.success("Report complete.")


if __name__ == "__main__":
    app()
