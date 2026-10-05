"""
export_dash.py
--------------
Exports everything a separate Dash app needs to rebuild the whitespace report,
as flat files in one folder. Copy that folder into the Dash project's data/
directory; the app then needs no access to this project, MLflow or the model.

Run after score_counties.py and validate.py (or via `make export_dash`).

Outputs (in --output-dir):
    counties.csv             one row per county: names, centroid, demographics,
                             model features, labels, score, rank, segment
    oof_predictions.csv      out-of-fold score of every county from every model
                             scored with score_counties (p_lr, p_rf, ...)
    counties.geojson         simplified county shapes, id = 5-digit FIPS string
    stores.csv               individual boutiques with their county FIPS
    coefficients.csv         feature, label, coef (or importance), sorted
    metrics_ci.csv           out-of-fold metrics with bootstrap 95% CIs
    hyperparameters.csv      tuned hyperparameters
    loo_ranks.csv            leave-one-out rank of each store county
    radius_sensitivity.csv   top open markets at each radius
    model_comparison.csv     all models side by side
    report_meta.json         counts, run name, radius, plain-language text

Example usage:
    python modeling/export_dash.py --output-dir ./reports/dash_data
"""

import json
import shutil
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
    feature_labels,
    model_definitions,
    large_county_pop,
    radius_mi,
    report_authors,
)
from core.constants import (
    coef_file,
    counties_geojson,
    labels_file,
    meta_file,
    metrics_file,
    oof_scores_file,
    store_count,
    stores_source,
    target_outcome,
)
from core.functions import load_geojson, segment_counties, slim_geojson
from modeling.report import MODEL_NAMES, model_summary

app = typer.Typer(add_completion=False)


@app.command()
def main(
    outcome: str = target_outcome[0],
    output_dir: Path = REPORTS_DIR / "dash_data",
    radius: float = radius_mi,
) -> None:

    ############################################################################
    # Step 1. Load scores, model outputs and county data
    ############################################################################
    results_dir = RESULTS_DIR / outcome
    eval_dir = EVAL_DIR / outcome

    scores = pd.read_parquet(results_dir / oof_scores_file)["p"]
    coef = pd.read_csv(results_dir / coef_file, index_col=0)["coef"]
    metrics = pd.read_csv(results_dir / metrics_file, index_col=0)["value"]
    labels = pd.read_parquet(PROCESSED_DATA_DIR / labels_file)[target_outcome]
    meta = pd.read_parquet(PROCESSED_DATA_DIR / meta_file)
    X = pd.read_parquet(PROCESSED_DATA_DIR / "X.parquet")
    model = str(metrics["model"])

    scored = segment_counties(scores, labels, meta, radius, target_outcome[0], store_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    ############################################################################
    # Step 2. counties.csv: one row per county
    ############################################################################
    counties = scored.join(X.drop(columns=scored.columns.intersection(X.columns)), how="left")
    counties.insert(0, "fips5", counties.index.map(lambda f: f"{int(f):05d}"))
    counties["large_county"] = counties.total_population >= large_county_pop
    counties.index.name = "fips"
    counties.to_csv(output_dir / "counties.csv")

    ############################################################################
    # Step 2b. oof_predictions.csv: one column per scored model, for the
    # model-comparison ROC / PR / calibration curves
    ############################################################################
    oof = counties[["fips5", target_outcome[0], "total_population", "large_county"]].copy()
    oof_models = []
    for name in model_definitions:
        f = results_dir / f"oof_scores_{name}.parquet"
        if not f.exists() and name == model:
            f = results_dir / oof_scores_file  # map model scored before per-model files
        if f.exists():
            oof[f"p_{name}"] = pd.read_parquet(f)["p"].reindex(oof.index)
            oof_models.append(name)
    missing = [m for m in model_definitions if m not in oof_models]
    if missing:
        logger.warning(f"No out-of-fold scores for {missing}; run `make score_all_models`.")
    oof.to_csv(output_dir / "oof_predictions.csv")

    ############################################################################
    # Step 3. counties.geojson: id is the 5-digit FIPS string (plotly matches
    # it to locations=counties.fips5 with featureidkey="id")
    ############################################################################
    geo = slim_geojson(load_geojson(RAW_DATA_DIR / counties_geojson), scored, store_count)
    for f in geo["features"]:
        f["id"] = f"{int(f['id']):05d}"
    (output_dir / "counties.geojson").write_text(
        json.dumps(geo, separators=(",", ":")), encoding="utf-8"
    )

    ############################################################################
    # Step 4. Stores, coefficients and evaluation tables
    ############################################################################
    stores = pd.read_parquet(PROCESSED_DATA_DIR / "stores.parquet")
    stores.to_csv(output_dir / "stores.csv", index=False)

    coef_out = coef.sort_values(ascending=False).rename("coef").to_frame()
    coef_out.insert(0, "label", [feature_labels.get(k, k) for k in coef_out.index])
    coef_out.index.name = "feature"
    coef_out.to_csv(output_dir / "coefficients.csv")

    copies = {
        results_dir / "oof_metrics_ci.csv": "metrics_ci.csv",
        results_dir / "hyperparameters.csv": "hyperparameters.csv",
        eval_dir / f"{model}_validation_loo.csv": "loo_ranks.csv",
        eval_dir / f"{model}_sensitivity_radius.csv": "radius_sensitivity.csv",
        eval_dir / "model_comparison.csv": "model_comparison.csv",
    }
    for src, name in copies.items():
        if src.exists():
            shutil.copyfile(src, output_dir / name)
        else:
            logger.warning(f"{src.name} not found; skipped {name}. "
                           "Run validate_model / compare_models first.")

    ############################################################################
    # Step 5. report_meta.json: counts and text the app displays
    ############################################################################
    def num(key, cast=int):
        try:
            return cast(float(metrics[key]))
        except (KeyError, ValueError):
            return None

    loo_file = eval_dir / f"{model}_validation_loo.csv"
    loo = pd.read_csv(loo_file) if loo_file.exists() else None
    report_meta = {
        "generated": date.today().isoformat(),
        "authors": report_authors,
        "outcome": outcome,
        "model": model,
        "oof_models": oof_models,
        "model_name": MODEL_NAMES.get(model, (model, model))[1],
        "run_name": str(metrics.get("run_name", f"{model}_orig_training")),
        "radius_mi": radius,
        "large_county_pop": large_county_pop,
        "cv_repeats": num("cv_repeats"),
        "n_bootstrap": num("n_bootstrap"),
        "split": {k: num(f"n {k}") for k in ("train", "valid", "test")},
        "n_counties": int(len(scored)),
        "n_store_counties": int((scored[store_count] > 0).sum()),
        "n_stores": int(scored[store_count].sum()),
        "n_large_counties": int(counties.large_county.sum()),
        "loo_median_rank": int(loo.whitespace_rank.median()) if loo is not None else None,
        "loo_top25": int((loo.whitespace_rank <= 25).sum()) if loo is not None else None,
        "loo_candidates": int(loo.of_candidates.iloc[0]) if loo is not None else None,
        "store_note": (PROCESSED_DATA_DIR / stores_source).read_text(encoding="utf-8").strip(),
        "model_learned": model_summary(coef, model, metrics),
        "feature_labels": feature_labels,
        "score_bins": [0, 0.005, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0],
        "score_bin_labels": ["little resemblance", "faint resemblance", "some resemblance",
                             "moderate match", "good match", "strong match",
                             "very close match"],
    }
    (output_dir / "report_meta.json").write_text(json.dumps(report_meta, indent=2),
                                                 encoding="utf-8")

    for p in sorted(output_dir.iterdir()):
        print(f"{p.name:26s} {p.stat().st_size / 1e3:9.1f} KB")
    logger.success(f"Dash data exported to {output_dir}")


if __name__ == "__main__":
    app()
