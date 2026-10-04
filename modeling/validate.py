#!/usr/bin/env python
"""Leave-one-out validation and radius sensitivity.

Leave-one-out: each store county is relabeled as having no store, the model is
refit, and the hidden county is ranked against every county without a store.
A low rank means the model would have flagged that market before Le Labo
opened there, which separates real signal from memorizing the labels.

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
    features,
    loo_repeats,
    rstate,
    sensitivity_radii,
)
from core.constants import features_file, labels_file, meta_file, store_count, target_outcome
from core.functions import oof_scores, segment_counties

app = typer.Typer(add_completion=False, help=__doc__)


def leave_one_out(X: pd.DataFrame, labels: pd.DataFrame, meta: pd.DataFrame,
                  repeats: int) -> pd.DataFrame:
    y = labels[target_outcome]
    rows = []
    for fips in y.index[y == 1]:
        y_held = y.copy()
        y_held.loc[fips] = 0
        p = pd.Series(oof_scores(X, y_held, repeats=repeats, seed=rstate), index=X.index)
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


def radius_sensitivity(scores: pd.Series, labels: pd.DataFrame, meta: pd.DataFrame,
                       radii: List[float], top: int) -> pd.DataFrame:
    cols = {}
    for r in radii:
        s = segment_counties(scores, labels, meta, r)
        cols[f"{r:g} mi"] = s.loc[s.segment == "open", "label"].head(top).tolist()
    out = pd.DataFrame(cols)
    out.index = range(1, len(out) + 1)
    out.index.name = "rank"
    return out


@app.command()
def main(
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / features_file, "--features-path"),
    labels_path: Path = typer.Option(PROCESSED_DATA_DIR / labels_file, "--labels-path"),
    meta_path: Path = typer.Option(PROCESSED_DATA_DIR / meta_file, "--meta-path"),
    eval_path: Path = typer.Option(EVAL_DIR, "--eval-path"),
    repeats: int = typer.Option(loo_repeats, "--repeats", min=1,
                                help="CV repeats per refit"),
    radii: List[float] = typer.Option(sensitivity_radii, "--radii"),
    top_n: int = typer.Option(10, "--top-n"),
) -> None:
    """Run leave-one-out validation and radius sensitivity."""

    ############################################################################
    # Step 1. Load data
    ############################################################################
    X = pd.read_parquet(features_path)[features]
    labels = pd.read_parquet(labels_path).loc[X.index, [target_outcome, store_count]]
    meta = pd.read_parquet(meta_path).loc[X.index]

    ############################################################################
    # Step 2. Leave-one-out over store counties
    ############################################################################
    n = int(labels[target_outcome].sum())
    logger.info(f"Leave-one-out over {n} store counties ({repeats} CV repeats each) ...")
    loo = leave_one_out(X, labels, meta, repeats)

    print(f"\n{'=' * 72}\nLeave-one-out: rank of each store county when the model never saw it\n{'=' * 72}")
    print(loo[["county", store_count, "p_held_out", "whitespace_rank"]].to_string(index=False))
    r = loo.whitespace_rank
    print(f"\nMedian rank {r.median():.0f} of {loo.of_candidates.iloc[0]:,} candidate counties | "
          f"top 25: {(r <= 25).sum()}/{len(loo)} | top 100: {(r <= 100).sum()}/{len(loo)}")

    ############################################################################
    # Step 3. Radius sensitivity (one full scoring, re-segmented per radius)
    ############################################################################
    scores = pd.Series(
        oof_scores(X, labels[target_outcome], repeats=repeats, seed=rstate), index=X.index
    )
    sens = radius_sensitivity(scores, labels, meta, radii, top_n)
    print(f"\n{'=' * 72}\nTop open markets by radius\n{'=' * 72}")
    print(sens.to_string())

    ############################################################################
    # Step 4. Save
    ############################################################################
    eval_path.mkdir(parents=True, exist_ok=True)
    loo.to_csv(eval_path / "validation_loo.csv", index=False)
    sens.to_csv(eval_path / "sensitivity_radius.csv")
    print(f"\nWrote validation_loo.csv and sensitivity_radius.csv to {eval_path}")
    logger.success("Validation complete.")


if __name__ == "__main__":
    app()
