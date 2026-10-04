"""
compare_models.py
-----------------
Side-by-side metrics for every model logged to MLflow for an outcome: the
train/valid/test metrics from evaluation.py and, where score_counties.py has
run, the out-of-fold metrics over all counties.

Select the map model on validation metrics (or out-of-fold), never on test.

Example usage:
    python modeling/compare_models.py --outcome has_store
"""

import os

import mlflow
import pandas as pd
import typer
from loguru import logger

from core.config import EVAL_DIR
from core.constants import mlflow_models_data, target_outcome

app = typer.Typer(add_completion=False)

COLUMNS = [
    "valid Average Precision",
    "valid AUC ROC",
    "valid Brier Score",
    "test Average Precision",
    "test AUC ROC",
    "oof Average Precision",
    "oof AUC ROC",
    "oof AUC ROC large counties",
    "oof AUC ROC large counties population only",
    "loo median rank",
    "loo top 25",
]


@app.command()
def main(outcome: str = target_outcome[0]):

    ############################################################################
    # Step 1. Query every run in the outcome's experiment
    ############################################################################
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_models_data)}")
    experiment = mlflow.get_experiment_by_name(f"{outcome}_model")
    if experiment is None:
        typer.secho(f"No MLflow experiment '{outcome}_model'. Train first.", fg="red")
        raise typer.Exit(code=1)

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        typer.secho("No runs found.", fg="red")
        raise typer.Exit(code=1)

    ############################################################################
    # Step 2. One row per run, metrics side by side
    ############################################################################
    table = pd.DataFrame({"run": runs["tags.mlflow.runName"]})
    for col in COLUMNS:
        key = f"metrics.{col}"
        table[col] = runs[key] if key in runs else float("nan")
    # Validation/test splits hold only a handful of store counties, so sort on
    # out-of-fold average precision (every county) when score_counties has run
    sort_col = (
        "oof Average Precision"
        if table["oof Average Precision"].notna().any()
        else "valid Average Precision"
    )
    table = table.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    print(f"\n{'=' * 100}\nModel comparison: {outcome} (sorted by {sort_col})\n{'=' * 100}")
    print(table.round(3).to_string(index=False))

    ############################################################################
    # Step 3. Save
    ############################################################################
    out = EVAL_DIR / outcome / "model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nWrote {out}")
    logger.success("Model comparison complete.")


if __name__ == "__main__":
    app()
