import pickle
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

################################################################################
# Step 1. Import Configurations and Constants
################################################################################

from core.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    cv_folds,
    cv_repeats,
    features,
    lr_C,
    lr_name,
    make_model,
    rstate,
)
from core.constants import (
    coef_file,
    features_file,
    labels_file,
    metrics_file,
    model_file,
    oof_scores_file,
    target_outcome,
    var_index,
)
from core.functions import fit_metrics, oof_scores

app = typer.Typer(add_completion=False)

################################################################################
# Step 2. Define CLI Arguments with Default Values
################################################################################


@app.command()
def main(
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / features_file, "--features-path"),
    labels_path: Path = typer.Option(PROCESSED_DATA_DIR / labels_file, "--labels-path"),
    results_path: Path = typer.Option(RESULTS_DIR, "--results-path"),
    repeats: int = typer.Option(cv_repeats, "--repeats", min=1, help="CV repeats"),
):

    ############################################################################
    # Step 3. Load Feature and Label Datasets
    ############################################################################
    X = pd.read_parquet(features_path)[features]
    y = pd.read_parquet(labels_path)[target_outcome].loc[X.index]

    print(f"X: {X.shape} | store counties: {int(y.sum())} of {len(y):,}")

    ############################################################################
    # Step 4. Out-of-Fold Scores
    ############################################################################
    # Every county is scored by a model trained without it, averaged over
    # `repeats` stratified k-fold partitions. These scores drive the map.
    logger.info(
        f"Scoring {lr_name} (C={lr_C}) with {repeats} x {cv_folds}-fold CV ..."
    )
    p = oof_scores(X, y, repeats=repeats, seed=rstate)
    metrics = fit_metrics(y, p)
    metrics["cv_repeats"] = repeats

    print(f"\n{'=' * 60}\nCross-Validated Results\n{'=' * 60}")
    print(f"AUC ROC            {metrics['cv_auc']:.3f}")
    print(f"Average Precision  {metrics['cv_average_precision']:.3f}")
    print(f"Base rate          {metrics['base_rate']:.4f}")

    ############################################################################
    # Step 5. Fit Final Model on All Counties (for coefficients)
    ############################################################################
    model = make_model().fit(X.values, y.values)
    coef = pd.Series(model[-1].coef_[0], index=features, name="coef").sort_values(
        ascending=False
    )
    print(f"\n{'=' * 60}\nStandardized Coefficients\n{'=' * 60}")
    print(coef.round(3).to_string())

    ############################################################################
    # Step 6. Save Results
    ############################################################################
    results_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"p": p}, index=pd.Index(X.index, name=var_index)).to_parquet(
        results_path / oof_scores_file
    )
    coef.to_csv(results_path / coef_file)
    pd.Series(metrics).to_csv(results_path / metrics_file, header=["value"])
    with open(results_path / model_file, "wb") as fh:
        pickle.dump(model, fh)

    print(f"\nSaved scores, coefficients, metrics and model to {results_path}")
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
