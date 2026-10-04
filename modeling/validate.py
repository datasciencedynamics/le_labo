#!/usr/bin/env python
"""Leave-one-out validation and radius sensitivity.

Leave-one-out: each store county is relabeled as having no store, the tuned
pipeline from MLflow is refit out-of-fold, and the hidden county is ranked
against every county without a store. A low rank means the model would have
flagged that market before Le Labo opened there, which separates real signal
from memorizing the labels.

Hyperparameters are held at their tuned values. They were selected on the
original split, which included every store county, so each held-out county
inherits a choice that saw its label. That is a mild optimism; record it as a
limitation.

Radius sensitivity: the top open markets when "served by a store" means each
radius in --radii, side by side.

Run from the project root, or via `make validate_model`.
"""

from pathlib import Path
from typing import List

import pandas as pd
import typer
from loguru import logger

from core.config import (
    EVAL_DIR,
    PROCESSED_DATA_DIR,
    cv_folds,
    loo_repeats,
    min_store_counties,
    model_definitions,
    oof_calibration_cv,
    rstate,
    sensitivity_radii,
)
from core.constants import store_count, target_outcome
from core.functions import (
    fresh_pipeline,
    log_mlflow_metrics,
    loo_rank_plot,
    mlflow_load_model,
    oof_scores,
    segment_counties,
)

app = typer.Typer(add_completion=False, help=__doc__)


def leave_one_out(pipeline, X, labels, meta, repeats) -> pd.DataFrame:
    y = labels[target_outcome[0]]
    rows = []
    for fips in y.index[y == 1]:
        y_held = y.copy()
        y_held.loc[fips] = 0
        p = pd.Series(
            oof_scores(pipeline, X, y_held, repeats=repeats, seed=rstate, folds=cv_folds,
                       calibration_cv=oof_calibration_cv,
                       min_positives=min_store_counties - 1),
            index=X.index,
        )
        cand = p[y_held == 0].sort_values(ascending=False)
        rows.append(
            {
                "fips": int(fips),
                "county": meta.loc[fips, "label"],
                store_count: int(labels.loc[fips, store_count]),
                "p_held_out": round(float(p.loc[fips]), 4),
                "whitespace_rank": int(cand.index.get_loc(fips)) + 1,
                "of_candidates": len(cand),
            }
        )
    return pd.DataFrame(rows).sort_values("whitespace_rank").reset_index(drop=True)


def radius_sensitivity(scores, labels, meta, radii, top) -> pd.DataFrame:
    cols = {}
    for r in radii:
        s = segment_counties(scores, labels, meta, r, target_outcome[0], store_count)
        cols[f"{r:g} mi"] = s.loc[s.segment == "open", "label"].head(top).tolist()
    out = pd.DataFrame(cols)
    out.index = range(1, len(out) + 1)
    out.index.name = "rank"
    return out


@app.command()
def main(
    model_type: str = "lr",
    pipeline_type: str = "orig",
    outcome: str = target_outcome[0],
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y.parquet",
    meta_path: Path = PROCESSED_DATA_DIR / "county_meta.parquet",
    repeats: int = typer.Option(loo_repeats, min=1, help="CV repeats per refit"),
    radii: List[float] = typer.Option(sensitivity_radii),
    top_n: int = 10,
) -> None:
    """Run leave-one-out validation and radius sensitivity."""

    ############################################################################
    # Step 1. Load the tuned model and data
    ############################################################################
    estimator_name = model_definitions[model_type]["estimator_name"]
    model = mlflow_load_model(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        model_name=f"{estimator_name}_{outcome}",
    )
    pipeline = fresh_pipeline(model)

    X = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path).loc[X.index]
    meta = pd.read_parquet(meta_path).loc[X.index]

    ############################################################################
    # Step 2. Leave-one-out over store counties
    ############################################################################
    n = int(labels[target_outcome[0]].sum())
    logger.info(f"Leave-one-out with {estimator_name} over {n} store counties "
                f"({repeats} CV repeats each) ...")
    loo = leave_one_out(pipeline, X, labels, meta, repeats)

    print(f"\n{'=' * 72}\nLeave-one-out: rank of each store county when the model never saw it\n{'=' * 72}")
    print(loo[["county", store_count, "p_held_out", "whitespace_rank"]].to_string(index=False))
    r = loo.whitespace_rank
    print(f"\nMedian rank {r.median():.0f} of {loo.of_candidates.iloc[0]:,} candidate counties | "
          f"top 25: {(r <= 25).sum()}/{len(loo)} | top 100: {(r <= 100).sum()}/{len(loo)}")

    ############################################################################
    # Step 3. Radius sensitivity (one full scoring, re-segmented per radius)
    ############################################################################
    y = labels[target_outcome[0]]
    scores = pd.Series(
        oof_scores(pipeline, X, y, repeats=repeats, seed=rstate, folds=cv_folds,
                   calibration_cv=oof_calibration_cv, min_positives=min_store_counties),
        index=X.index,
    )
    sens = radius_sensitivity(scores, labels, meta, radii, top_n)
    print(f"\n{'=' * 72}\nTop open markets by radius\n{'=' * 72}")
    print(sens.to_string())

    ############################################################################
    # Step 4. Save
    ############################################################################
    eval_dir = EVAL_DIR / outcome
    eval_dir.mkdir(parents=True, exist_ok=True)
    loo.to_csv(eval_dir / f"{estimator_name}_validation_loo.csv", index=False)
    sens.to_csv(eval_dir / f"{estimator_name}_sensitivity_radius.csv")

    fig = loo_rank_plot(loo, title=f"{estimator_name}: leave-one-out whitespace rank")
    (eval_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_dir / "figures" / f"loo_rank_{estimator_name}.png", dpi=150)
    log_mlflow_metrics(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        metrics=pd.Series({
            "loo median rank": float(r.median()),
            "loo top 25": float((r <= 25).sum()),
            "loo top 100": float((r <= 100).sum()),
        }),
        images={f"loo_rank_{estimator_name}.png": fig},
    )
    print(f"\nWrote {estimator_name}_validation_loo.csv and "
          f"{estimator_name}_sensitivity_radius.csv to {eval_dir}")
    logger.success("Validation complete.")


if __name__ == "__main__":
    app()
