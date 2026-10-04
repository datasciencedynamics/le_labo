from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from core.config import EVAL_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, radius_mi
from core.constants import labels_file, meta_file, oof_scores_file, store_count, target_outcome
from core.functions import segment_counties

app = typer.Typer(add_completion=False)

KEEP = [
    "label",
    "total_population",
    "median_hh_inc",
    "college_pct",
    store_count,
    "p",
    "rank",
    "mi_to_store",
    "segment",
]

################################################################################
# ---- STEP 1: Define command-line arguments with default values ----
################################################################################


@app.command()
def main(
    scores_path: Path = typer.Option(RESULTS_DIR / oof_scores_file, "--scores-path"),
    labels_path: Path = typer.Option(PROCESSED_DATA_DIR / labels_file, "--labels-path"),
    meta_path: Path = typer.Option(PROCESSED_DATA_DIR / meta_file, "--meta-path"),
    eval_path: Path = typer.Option(EVAL_DIR, "--eval-path"),
    radius: float = typer.Option(
        radius_mi, "--radius", help="Miles from a store county that count as served."
    ),
    top_n: int = typer.Option(10, "--top-n", help="Rows printed per segment."),
):

    ############################################################################
    # STEP 2: Load Scores, Labels and County Metadata
    ############################################################################
    scores = pd.read_parquet(scores_path)["p"]
    labels = pd.read_parquet(labels_path)[[target_outcome, store_count]]
    meta = pd.read_parquet(meta_path)

    ############################################################################
    # STEP 3: Distance to Nearest Store County, Segments, Ranks
    ############################################################################
    scored = segment_counties(scores, labels, meta, radius)

    ############################################################################
    # STEP 4: Save County Scores and Whitespace Tables
    ############################################################################
    eval_path.mkdir(parents=True, exist_ok=True)
    scored[KEEP].to_csv(eval_path / "county_scores.csv")
    scored.loc[scored.segment == "open", KEEP].to_csv(eval_path / "whitespace_open.csv")
    scored.loc[scored.segment == "fill", KEEP].to_csv(eval_path / "whitespace_fill.csv")

    ############################################################################
    # STEP 5: Print Summary
    ############################################################################
    seg = scored.segment.value_counts()
    print(f"Radius: {radius:g} mi")
    print(f"Counties: {seg.get('store', 0)} store, {seg.get('fill', 0)} fill-in, "
          f"{seg.get('open', 0)} open")

    cols = ["label", "p", "mi_to_store", "total_population"]
    for name in ("open", "fill"):
        print(f"\n{'=' * 60}\nTop {name} markets\n{'=' * 60}")
        print(scored.loc[scored.segment == name, cols].head(top_n)
              .round(3).to_string(index=False))

    print(f"\n{'=' * 60}\nStore counties by model score\n{'=' * 60}")
    print(scored.loc[scored.segment == "store", ["label", store_count, "p", "rank"]]
          .round(3).to_string(index=False))

    print(f"\nWrote county_scores.csv, whitespace_open.csv, whitespace_fill.csv to {eval_path}")
    logger.success("Modeling evaluation complete.")


if __name__ == "__main__":
    app()
