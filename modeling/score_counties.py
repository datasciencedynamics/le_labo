"""
score_counties.py
-----------------
Scores every US county for whitespace with the tuned model from MLflow.

The trained model_tuner model (train.py) is fit on a 60/20/20 split, so its own
predictions on training counties are in-sample. For the map, every county needs
a score from a model that never saw its label. This script pulls the tuned
pipeline from MLflow, refits it out-of-fold over repeated stratified k-fold
(sigmoid-calibrated inside each fold), and averages the scores.

Outputs:
    models/results/<outcome>/oof_scores_<model>.parquet  per-model scores (always)
    models/results/<outcome>/oof_metrics_ci_<model>.csv  per-model CIs (always)
    models/results/<outcome>/oof_scores.parquet   one score per county
    models/results/<outcome>/coefficients.csv     coefficients or importances
    models/results/<outcome>/oof_metrics.csv      out-of-fold AUC, AP, Brier
    models/eval/<outcome>/county_scores.csv       scores, segments, ranks
    models/eval/<outcome>/whitespace_open.csv     open markets
    models/eval/<outcome>/whitespace_fill.csv     fill-in markets

Example usage:
    python modeling/score_counties.py --model-type lr --pipeline-type orig
"""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

################################################################################
# Step 1. Import Configurations and Constants
################################################################################

from core.config import (
    EVAL_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    cv_folds,
    cv_repeats,
    large_county_pop,
    min_store_counties,
    n_bootstrap,
    model_definitions,
    oof_calibration_cv,
    radius_mi,
    rstate,
)
from core.constants import (
    coef_file,
    metrics_file,
    oof_scores_file,
    store_count,
    target_outcome,
    var_index,
)
from core.functions import (
    fresh_pipeline,
    bootstrap_ci,
    tuned_hyperparameters,
    log_mlflow_metrics,
    mlflow_load_model,
    model_importances,
    oof_calibration_plot,
    oof_fit_metrics,
    oof_pr_plot,
    oof_roc_plot,
    oof_scores,
    segment_counties,
    subgroup_metrics,
)

app = typer.Typer(add_completion=False)

KEEP = ["label", "total_population", "median_hh_inc", "college_pct", store_count,
        "p", "rank", "mi_to_store", "segment"]


@app.command()
def main(
    model_type: str = "lr",
    pipeline_type: str = "orig",
    outcome: str = target_outcome[0],
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y.parquet",
    meta_path: Path = PROCESSED_DATA_DIR / "county_meta.parquet",
    repeats: int = cv_repeats,
    radius: float = radius_mi,
    top_n: int = 10,
    map_outputs: bool = typer.Option(
        True, help="Write the map/report files (scores, coefficients, whitespace "
        "tables). Off when scoring a comparison model, so it does not overwrite "
        "the map model's outputs."),
):

    ############################################################################
    # Step 2. Load the Tuned Model from MLflow
    ############################################################################
    estimator_name = model_definitions[model_type]["estimator_name"]
    run_name = f"{estimator_name}_{pipeline_type}_training"

    model = mlflow_load_model(
        experiment_name=f"{outcome}_model",
        run_name=run_name,
        model_name=f"{estimator_name}_{outcome}",
    )
    pipeline = fresh_pipeline(model)
    print(f"Tuned pipeline from MLflow run '{run_name}':\n{pipeline}\n")

    # Split sizes from model_tuner's 60/20/20 split, for the report
    split_sizes = {}
    for part in ("train", "valid", "test"):
        idx = getattr(model, f"X_{part}_index", None)
        split_sizes[f"n {part}"] = len(idx) if idx is not None else 0

    ############################################################################
    # Step 3. Load Features, Labels and County Metadata
    ############################################################################
    X = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path).loc[X.index]
    y = labels[target_outcome[0]].squeeze()
    meta = pd.read_parquet(meta_path).loc[X.index]

    ############################################################################
    # Step 4. Out-of-Fold Scores for Every County
    ############################################################################
    logger.info(f"Scoring counties with {estimator_name}: {repeats} x {cv_folds}-fold ...")
    p = oof_scores(pipeline, X, y, repeats=repeats, seed=rstate, folds=cv_folds,
                   calibration_cv=oof_calibration_cv, min_positives=min_store_counties)
    scores = pd.Series(p, index=pd.Index(X.index, name=var_index), name="p")
    metrics = oof_fit_metrics(y, p)

    # Separating big cities from rural counties is trivial, so also report
    # discrimination among large counties, where siting decisions are made
    big = (meta.total_population >= large_county_pop).values
    big_label = f"pop >= {large_county_pop // 1000}k"  # plot titles
    # MLflow metric names allow letters, digits, spaces and _-./: only
    metrics.update(subgroup_metrics(y, p, big, "large counties"))
    metrics.update(subgroup_metrics(y, meta.total_population.values, big,
                                    "large counties population only"))

    print(f"\n{'=' * 60}\nOut-of-Fold Results ({estimator_name})\n{'=' * 60}")
    for k, v in metrics.items():
        print(f"{k:24s} {v:.4f}" if isinstance(v, float) else f"{k:24s} {v}")

    ############################################################################
    # Step 4b. Bootstrap 95% CIs for the report's metrics table
    ############################################################################
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    pop = meta.total_population.values
    ci_rows = [
        ("ROC-AUC, all counties", y, p, roc_auc_score),
        ("Average Precision, all counties", y, p, average_precision_score),
        ("Brier Score, all counties", y, p, brier_score_loss),
        ("ROC-AUC, counties 500k+", y[big], p[big], roc_auc_score),
        ("ROC-AUC, counties 500k+, population alone", y[big], pop[big], roc_auc_score),
        ("Average Precision, counties 500k+", y[big], p[big], average_precision_score),
    ]
    ci_table = pd.DataFrame(
        [(name, *bootstrap_ci(yy, pp, fn, n_boot=n_bootstrap, seed=rstate))
         for name, yy, pp, fn in ci_rows],
        columns=["metric", "estimate", "ci_low", "ci_high"],
    )
    print(f"\n{'=' * 60}\nBootstrap 95% CIs ({n_bootstrap} resamples)\n{'=' * 60}")
    print(ci_table.round(3).to_string(index=False))

    hyper = tuned_hyperparameters(
        pipeline,
        model_definitions[model_type]["tuned_parameters"],
        extra={"lr": ["penalty", "class_weight", "solver", "max_iter"],
               "rf": ["class_weight", "min_samples_leaf"],
               "xgb": ["n_estimators"],
               "cat": ["n_estimators"]}.get(model_type, []),
    )

    ############################################################################
    # Step 5. Coefficients (linear) or Importances (trees)
    ############################################################################
    importances = model_importances(pipeline, X, y)
    print(f"\n{'=' * 60}\nCoefficients / Importances\n{'=' * 60}")
    print(importances.round(3).to_string())

    ############################################################################
    # Step 6. Segments: store / fill-in / open
    ############################################################################
    scored = segment_counties(scores, labels, meta, radius, target_outcome[0], store_count)

    seg = scored.segment.value_counts()
    print(f"\nRadius: {radius:g} mi | {seg.get('store', 0)} store, "
          f"{seg.get('fill', 0)} fill-in, {seg.get('open', 0)} open")
    cols = ["label", "p", "mi_to_store", "total_population"]
    for name in ("open", "fill"):
        print(f"\n{'=' * 60}\nTop {name} markets\n{'=' * 60}")
        print(scored.loc[scored.segment == name, cols].head(top_n).round(3).to_string(index=False))

    ############################################################################
    # Step 7. Save Results
    ############################################################################
    results_dir = RESULTS_DIR / outcome
    eval_dir = EVAL_DIR / outcome
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Per-model out-of-fold scores and CIs, kept side by side for comparison
    scores.to_frame().to_parquet(results_dir / f"oof_scores_{estimator_name}.parquet")
    ci_table.to_csv(results_dir / f"oof_metrics_ci_{estimator_name}.csv", index=False)

    if map_outputs:
        scores.to_frame().to_parquet(results_dir / oof_scores_file)
        importances.to_csv(results_dir / coef_file)
        pd.Series({**metrics, **split_sizes, "model": estimator_name,
                   "pipeline": pipeline_type, "run_name": run_name,
                   "cv_repeats": repeats, "n_bootstrap": n_bootstrap}
                  ).to_csv(results_dir / metrics_file, header=["value"])
        ci_table.to_csv(results_dir / "oof_metrics_ci.csv", index=False)
        hyper.to_csv(results_dir / "hyperparameters.csv", header=["value"])

        scored[KEEP].to_csv(eval_dir / "county_scores.csv")
        scored.loc[scored.segment == "open", KEEP].to_csv(eval_dir / "whitespace_open.csv")
        scored.loc[scored.segment == "fill", KEEP].to_csv(eval_dir / "whitespace_fill.csv")

    ############################################################################
    # Step 8. Out-of-Fold Plots (every county, bootstrap 95% bands)
    ############################################################################
    plots = {
        f"oof_roc_{estimator_name}.png": oof_roc_plot(
            y, p, baseline=pop, n_boot=n_bootstrap,
            title=f"{estimator_name}: ROC, out-of-fold, all counties"),
        f"oof_roc_large_{estimator_name}.png": oof_roc_plot(
            y[big], p[big], baseline=pop[big], n_boot=n_bootstrap,
            title=f"{estimator_name}: ROC, out-of-fold, counties {big_label}"),
        f"oof_pr_{estimator_name}.png": oof_pr_plot(
            y, p, n_boot=n_bootstrap,
            title=f"{estimator_name}: precision-recall, out-of-fold"),
        f"oof_calibration_large_{estimator_name}.png": oof_calibration_plot(
            y[big], p[big], n_bins=5,
            title=f"{estimator_name}: calibration, out-of-fold, counties {big_label}"),
    }
    fig_dir = eval_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.items():
        fig.savefig(fig_dir / name, dpi=150)

    ############################################################################
    # Step 9. Log Out-of-Fold Metrics and Plots to the Same MLflow Run
    ############################################################################
    log_mlflow_metrics(
        experiment_name=f"{outcome}_model",
        run_name=run_name,
        metrics=pd.Series({k: v for k, v in metrics.items() if k.startswith("oof")}),
        images=plots,
    )
    print(f"Saved {len(plots)} figures to {fig_dir}")

    print(f"\nWrote scores to {results_dir} and whitespace tables to {eval_dir}")
    logger.success("County scoring complete.")


if __name__ == "__main__":
    app()
