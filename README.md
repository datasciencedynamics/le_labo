# lelabo-whitespace

Scrapes Le Labo's store locations, scores every US county on how closely it matches the demographic profile of counties that already host a boutique, validates the model by holding out each existing store county, and renders a self-contained HTML map of the whitespace.

## Requirements

- Python 3.10 to 3.12 (3.12 recommended). model_tuner pins `pandas<2.2.3`, which has no wheels for Python 3.13
- MLflow is pinned below 3 to match the kidney project's `mlruns/` layout. The code also runs on MLflow 3: it sets `MLFLOW_ALLOW_FILE_STORE=true`, logs models with cloudpickle, and `core/model_registry.py` reads both layouts
- `make` and `bash` (preinstalled on macOS and Linux; on Windows use WSL, or follow the [manual steps](#without-make))
- Internet access (county data downloads from GitHub; scraping needs lelabofragrances.com)

## Project layout

```
lelabo-whitespace/
├── Makefile
├── README.md
├── requirements.txt
├── core/
│   ├── config.py             # paths, model_definitions (lr/rf/xgb/cat), pipelines, whitespace settings
│   ├── constants.py          # MLflow paths and run names, target_outcome, URLs, file names
│   ├── functions.py          # MLflow helpers, PlotMetrics, metrics, whitespace helpers
│   ├── model_registry.py     # browse every model logged under mlruns/
│   └── templates/
│       └── report_template.html
├── preprocessing/
│   ├── data_gen.py           # download county boundaries + demographics -> data/raw
│   ├── scrape_stores.py      # scrape the store locator -> data/raw/stores_scraped.csv
│   ├── preprocessing.py      # county table + store-to-county assignment
│   └── feat_gen.py           # features X, labels y, county metadata
├── modeling/
│   ├── train.py              # model_tuner tuning, calibration, MLflow logging
│   ├── evaluation.py         # train/valid/test metrics and plots to MLflow
│   ├── compare_models.py     # every logged model side by side
│   ├── score_counties.py     # out-of-fold scores for every county from the tuned MAP_MODEL
│   ├── validate.py           # leave-one-out validation + radius sensitivity
│   └── report.py             # HTML map -> reports/
├── data/
│   ├── external/             # stores_fallback.csv (38 boutiques compiled from Yelp, Oct 2026)
│   ├── raw/                  # downloads, scraped stores, scrape/ debug output
│   ├── interim/              # counties.parquet
│   └── processed/            # stores.parquet, X.parquet, y.parquet, county_meta.parquet
├── models/
│   ├── results/has_store/    # training logs, oof_scores.parquet, coefficients.csv, oof_metrics.csv
│   └── eval/has_store/       # eval logs, model_comparison.csv, whitespace_*.csv, *_validation_loo.csv
├── mlruns/                   # MLflow: preprocessing artifacts + has_store_model runs, models, plots
├── reports/                  # lelabo_whitespace.html
└── notebooks/
    └── colab_quickstart.ipynb
```

Only code, config and `.gitkeep` placeholders are tracked. Everything under `data/`, `models/`, `reports/` and `mlruns/`, plus `*.csv`, `*.json`, `*.geojson`, `*.parquet`, `*.pkl`, `*.xlsx` and `*.zip` anywhere, is gitignored. A fresh `git clone` therefore has no fallback store list; copy `data/external/stores_fallback.csv` from the project zip, or scrape.

## Start to finish

### 1. Unzip and enter the folder

```bash
unzip lelabo-whitespace.zip
cd lelabo-whitespace
make help                      # every target with a description
```

### 2. Create and activate an environment

**venv**

```bash
make create_venv
source lelabo_venv/bin/activate
```

**conda**

```bash
conda create -n conda_lelabo python=3.12
conda activate conda_lelabo
```

`make create_conda_env` and `make activate_venv` print these commands. Every later target calls `python` from the active environment.

### 3. Install dependencies

```bash
make requirements
make install_browsers          # headless Chromium for the scraper
```

On Linux, `install_browsers` may ask for sudo to install system libraries. To skip them, run `make install_browsers BROWSER_FLAGS=`.

### 4. Scrape the stores

```bash
make scrape_stores
```

Writes `data/raw/stores_scraped.csv`, with the raw locator responses, endpoint URLs and rendered page in `data/raw/scrape/`. A failed scrape (no network, no Chromium, site blocked) or an empty one is not fatal: it prints a warning and later steps use the existing scraped file or `data/external/stores_fallback.csv`.

You can skip this step: `preproc_pipeline` scrapes automatically when `data/raw/stores_scraped.csv` doesn't exist yet. Run `make scrape_stores` when you want to refresh an existing scrape.

### 5. Check the scraped stores

Open `data/raw/stores_scraped.csv`. The locator may list department-store counters alongside boutiques. Find the column that tells them apart and pass it as `QUERY` in the next step, for example `QUERY="type == 'Boutique'"`. `QUERY` accepts any pandas query expression.

### 6. Preprocess

```bash
make preproc_pipeline QUERY="type == 'Boutique'"     # omit QUERY if no filtering is needed
```

Runs four steps:

| Target | Script | Output |
|---|---|---|
| `data_gen` | `preprocessing/data_gen.py` | `data/raw/counties.geojson`, `data/raw/county_demographics.csv` (skipped once downloaded) |
| scrape, if needed | `preprocessing/scrape_stores.py` | `data/raw/stores_scraped.csv` (only when missing; skip with `SCRAPE=0`) |
| `data_prep_preprocessing` | `preprocessing/preprocessing.py` | `data/interim/counties.parquet`, `data/processed/stores.parquet` |
| `feat_gen` | `preprocessing/feat_gen.py` | `data/processed/X.parquet`, `y.parquet`, `county_meta.parquet` |

`preprocessing.py` uses `data/raw/stores_scraped.csv` when it exists and the fallback list otherwise. Override with `STORES_FILE=path.csv`.

### 7. Train and evaluate the models

```bash
make train_eval_pipeline
```

Same structure as the kidney project: each model is tuned with model_tuner on a stratified 60/20/20 split (randomized grid, scored on average precision), calibrated, and logged to MLflow under experiment `has_store_model`, run `<model>_<pipeline>_training`.

| Target | Script | Output |
|---|---|---|
| `train_logistic_regression`, `train_random_forest`, `train_xgboost`, `train_catboost` (or `train_all_models`) | `modeling/train.py` | model + best hyperparameters in MLflow; log `models/results/has_store/<model>_orig.txt` |
| `eval_logistic_regression` ... `eval_catboost` (or `eval_all_models`) | `modeling/evaluation.py` | train/valid/test metrics and ROC, PR, calibration, confusion and threshold plots in MLflow; log `models/eval/has_store/<model>_eval_orig.txt` |
| `compare_models` | `modeling/compare_models.py` | `models/eval/has_store/model_comparison.csv` |

`PRETRAINED=1` reloads the trained models from MLflow and recalibrates instead of re-tuning, as in the kidney Makefile. `make mlflow_ui` opens the runs at http://localhost:5501.

With only 22 store counties, the validation and test splits hold about 4 or 5 store counties each, so their metrics swing a lot from split to split. The out-of-fold metrics from step 8 use every county and are the ones to compare.

### 8. Score counties, validate and render the map

```bash
make whitespace_pipeline                 # MAP_MODEL=lr by default
```

| Target | Script | Output |
|---|---|---|
| `score_counties` | `modeling/score_counties.py` | `models/results/has_store/oof_scores.parquet`, `coefficients.csv`, `oof_metrics.csv`; `models/eval/has_store/county_scores.csv`, `whitespace_open.csv`, `whitespace_fill.csv`; `figures/oof_roc_*.png`, `oof_roc_large_*.png`, `oof_pr_*.png`, `oof_calibration_large_*.png`; out-of-fold metrics and figures added to the model's MLflow run |
| `validate_model` | `modeling/validate.py` | `models/eval/has_store/<model>_validation_loo.csv`, `<model>_sensitivity_radius.csv`, `figures/loo_rank_<model>.png` |
| `render_report` | `modeling/report.py` | `reports/lelabo_whitespace.html` |
| `export_dash` | `modeling/export_dash.py` | `reports/dash_data/` (CSVs, GeoJSON, JSON for the Dash app; set `DASH_DIR=` to write elsewhere) |

**Why a separate scoring step.** The trained model saw 60% of counties during fitting, so its predictions on those counties are in-sample. `score_counties.py` pulls the tuned pipeline (best hyperparameters) for `MAP_MODEL` from MLflow and refits it out-of-fold over 20 repeats of stratified 5-fold CV, with sigmoid calibration inside each fold, so every county's score comes from a model that never saw its label. Boosters are refit at their tuned number of trees.

**Which plots to show.** The ROC and PR plots from `evaluation.py` rest on 4 or 5 store counties per split, so they are staircases that say little; treat them as diagnostics. Use the out-of-fold figures in `models/eval/has_store/figures/` instead: they cover every county and carry stratified bootstrap 95% bands. The headline is `oof_roc_large_<model>.png`, discrimination among counties of 500k or more against population alone (logistic regression: AUC 0.84, 95% band 0.73 to 0.94, versus 0.75). Across all counties, population alone already reaches 0.99, so the all-county AUC says little about the model.

**Validation.** Leave-one-out hides each store county in turn, refits the tuned pipeline, and ranks the hidden county against every county without a store. A low rank means the model would have flagged that market before Le Labo opened there. On the fallback list with logistic regression, the median rank is 8 of 3,090, and 17 of 22 store counties rank in the top 25. The radius check lists the top open markets at 30, 60 and 100 miles.

To map a different model: `make whitespace_pipeline MAP_MODEL=cat`.

### 9. Review and share

```bash
make results                   # lists outputs and the map path
make package                   # zips models/ and reports/ to lelabo_whitespace_output.zip
make mlflow_ui                 # browse runs, metrics and plots at http://localhost:5501
```

Open `reports/lelabo_whitespace.html` in any browser. Fonts and the D3 library load from Google Fonts and cdnjs; everything else is embedded.

### One command

```bash
make preproc_train_eval QUERY="type == 'Boutique'"   # steps 4 to 8 (scrapes only if no scraped file yet)
make preproc_train_eval SCRAPE=0                     # never scrape: existing scraped file or fallback
make scrape_stores preproc_train_eval                # force a fresh scrape first
```

To force the bundled list even when a scraped file exists, add `STORES_FILE=data/external/stores_fallback.csv`.

## Make variables

Set on the command line, e.g. `make train_eval_pipeline RADIUS=40`.

| Variable | Default | Meaning |
|---|---|---|
| `QUERY` | empty | pandas query to filter stores in `data_prep_preprocessing` |
| `STORES_FILE` | empty | store CSV; empty means scraped if present, else fallback |
| `SCRAPE` | `1` | `1` scrapes in `preproc_pipeline` when no scraped file exists; `0` never scrapes |
| `OUTCOMES` | `has_store` | outcome(s) looped over by train/eval targets |
| `PIPELINES` | `orig` | pipelines from `core/config.py` (`orig`, `smote`, `under`), e.g. `PIPELINES="orig smote"` |
| `SCORING` | `average_precision` | model_tuner tuning metric |
| `PRETRAINED` | `0` | `1` reloads models from MLflow and recalibrates instead of re-tuning |
| `MAP_MODEL` | `lr` | model used for county scores, validation and the map (`lr`, `rf`, `xgb`, `cat`) |
| `MAP_PIPELINE` | `orig` | pipeline of the map model |
| `RADIUS` | `60` | miles from a store county that count as already served |
| `RADII` | `30 60 100` | radii compared in `validate_model` |
| `CV_REPEATS` | `20` | repeats of stratified 5-fold CV in `score_counties` |
| `LOO_REPEATS` | `5` | CV repeats per refit in `validate_model` |
| `PYTHON_INTERPRETER` | `python` | use `python3` if your system has no `python` |
| `BROWSER_FLAGS` | `--with-deps` | Playwright install flags |

Hyperparameter grids, samplers, random state (222), fold counts and report table sizes live in `core/config.py`, laid out as in the kidney project.

## Cleaning up

| Target | Removes |
|---|---|
| `clean_models` | everything in `models/results`, `models/eval`, `reports` except `.gitkeep` |
| `clean_mlruns` | `mlruns/` (all MLflow runs, models, artifacts) |
| `clean_data` | everything in `data/raw`, `data/interim`, `data/processed` except `.gitkeep` (keeps `data/external`) |
| `clean` | compiled Python files |
| `clean_venv` | `lelabo_venv/` |

## Running in Google Colab

Colab has its own Python and root access, so skip the environment step:

```python
!unzip -oq lelabo-whitespace.zip
%cd lelabo-whitespace
!make requirements install_browsers
!make preproc_train_eval               # scrapes first if no scraped file exists
```

```python
from IPython.display import HTML
HTML(open('reports/lelabo_whitespace.html').read())
```

To call the scraper from a notebook cell, use `await`, since Jupyter already runs an event loop:

```python
from preprocessing.scrape_stores import scrape
stores = await scrape()
```

`notebooks/colab_quickstart.ipynb` has the full sequence.

## Without make

Run each script from the project root with the root on `PYTHONPATH`, in this order:

```bash
export PYTHONPATH=$(pwd)
export MLFLOW_ALLOW_FILE_STORE=true     # needed for MLflow >= 3.x with ./mlruns
python preprocessing/data_gen.py
python preprocessing/scrape_stores.py
python preprocessing/preprocessing.py --query "type == 'Boutique'"
python preprocessing/feat_gen.py
python modeling/train.py --model-type lr
python modeling/evaluation.py --model-type lr
python modeling/compare_models.py
python modeling/score_counties.py --model-type lr
python modeling/validate.py --model-type lr
python modeling/report.py
```

Every script has `--help` and defaults to the paths in `core/config.py`. `python -m preprocessing.data_gen` (and so on) also works from the root without setting `PYTHONPATH`.

## Method

- **Stores to counties.** Scraped stores are placed in counties by point-in-polygon on Census county boundaries. Non-US stores drop out.
- **Features.** Log population, log density, log median household income, college share, foreign-born share, age 29 and under, age 65 and over, rural share (ACS 2012–2016 five-year estimates, via the MIT Election Data and Science Lab county file).
- **Models.** Logistic regression, random forest, XGBoost and CatBoost, tuned with model_tuner (stratified 60/20/20, randomized grid on average precision, calibrated) and tracked in MLflow.
- **Map scores.** Out-of-fold predictions from the tuned `MAP_MODEL` pipeline, averaged over 20 repeats of stratified 5-fold CV with per-fold sigmoid calibration, so no county is scored by a model that saw its own label.
- **Model choice.** Logistic regression is the default map model. With 22 positives, the tree ensembles score lower on out-of-fold average precision and leave-one-out rank; check `compare_models` and `validate_model` for your own data.
- **Segments.** Counties with a store; *fill-in* (no store, but within the radius of a store county's centroid); *open* (no store within the radius).

## Limits

- Population dominates the fit, so the high AUC mostly reflects that large urban counties have stores. The useful signal is the ranking among counties of similar size.
- Counties are coarse. A real site study would use tract-level data and foot traffic.
- Demographics are from 2012–2016 and miss pandemic-era migration.
- Review lelabofragrances.com's terms of use before scraping.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'core'` | Run through make, or `export PYTHONPATH=$(pwd)` from the project root |
| `python: command not found` | Activate your environment, or add `PYTHON_INTERPRETER=python3` |
| `No store records found` | Check `data/raw/scrape/endpoints.txt`, or open the locator in Chrome DevTools > Network > Fetch/XHR to find the store API |
| `Playwright Sync API inside the asyncio loop` | You're in a notebook; use `await scrape()` |
| `install_browsers` asks for sudo | Run `make install_browsers BROWSER_FLAGS=` and install missing libraries yourself if Chromium fails to launch |
| `Need at least 5 store counties` | `QUERY` filtered out too many stores; loosen it |
| `MlflowException: The filesystem tracking backend ... is in ...` | MLflow >= 3.x; run through make, or `export MLFLOW_ALLOW_FILE_STORE=true` |
| `No runs found with run_name 'lr_orig_training'` | Train that model first (`make train_logistic_regression`) |
| `Store file not found` | Copy `data/external/stores_fallback.csv` from the project zip, or pass `STORES_FILE=` |
