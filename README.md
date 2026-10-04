# lelabo-whitespace

Scrapes Le Labo's store locations, scores every US county on how closely it matches the demographic profile of counties that already host a boutique, validates the model by holding out each existing store county, and renders a self-contained HTML map of the whitespace.

## Requirements

- Python 3.9 or newer (3.12 recommended)
- `make` and `bash` (preinstalled on macOS and Linux; on Windows use WSL, or follow the [manual steps](#without-make))
- Internet access (county data downloads from GitHub; scraping needs lelabofragrances.com)

## Project layout

```
lelabo-whitespace/
├── Makefile
├── README.md
├── requirements.txt
├── core/
│   ├── config.py             # paths, random state, CV settings, radius, features, model
│   ├── constants.py          # URLs, file names, column names
│   ├── functions.py          # shared helpers (downloads, geometry, scoring, segmentation)
│   └── templates/
│       └── report_template.html
├── preprocessing/
│   ├── data_gen.py           # download county boundaries + demographics -> data/raw
│   ├── scrape_stores.py      # scrape the store locator -> data/raw/stores_scraped.csv
│   ├── preprocessing.py      # county table + store-to-county assignment
│   └── feat_gen.py           # features X, labels y, county metadata
├── modeling/
│   ├── train.py              # out-of-fold scores, coefficients, final model
│   ├── evaluation.py         # segments + whitespace tables
│   ├── validate.py           # leave-one-out validation + radius sensitivity
│   └── report.py             # HTML map -> reports/
├── data/
│   ├── external/             # stores_fallback.csv (38 boutiques compiled from Yelp, Oct 2026)
│   ├── raw/                  # downloads, scraped stores, scrape/ debug output
│   ├── interim/              # counties.parquet
│   └── processed/            # stores.parquet, X.parquet, y.parquet, county_meta.parquet
├── models/
│   ├── results/              # oof_scores.parquet, coefficients.csv, metrics.csv, model .pkl, logs
│   └── eval/                 # county_scores.csv, whitespace_*.csv, validation_*.csv, logs
├── reports/                  # lelabo_whitespace.html
└── notebooks/
    └── colab_quickstart.ipynb
```

Only code, config and `.gitkeep` placeholders are tracked. Everything under `data/`, `models/` and `reports/`, plus `*.csv`, `*.json`, `*.geojson`, `*.parquet`, `*.pkl`, `*.xlsx` and `*.zip` anywhere, is gitignored. A fresh `git clone` therefore has no fallback store list; copy `data/external/stores_fallback.csv` from the project zip, or scrape.

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

Writes `data/raw/stores_scraped.csv`, with the raw locator responses, endpoint URLs and rendered page in `data/raw/scrape/`. If it finds nothing, it says so and later steps use `data/external/stores_fallback.csv`.

### 5. Check the scraped stores

Open `data/raw/stores_scraped.csv`. The locator may list department-store counters alongside boutiques. Find the column that tells them apart and pass it as `QUERY` in the next step, for example `QUERY="type == 'Boutique'"`. `QUERY` accepts any pandas query expression.

### 6. Preprocess

```bash
make preproc_pipeline QUERY="type == 'Boutique'"     # omit QUERY if no filtering is needed
```

Runs three steps:

| Target | Script | Output |
|---|---|---|
| `data_gen` | `preprocessing/data_gen.py` | `data/raw/counties.geojson`, `data/raw/county_demographics.csv` (skipped once downloaded) |
| `data_prep_preprocessing` | `preprocessing/preprocessing.py` | `data/interim/counties.parquet`, `data/processed/stores.parquet` |
| `feat_gen` | `preprocessing/feat_gen.py` | `data/processed/X.parquet`, `y.parquet`, `county_meta.parquet` |

`preprocessing.py` uses `data/raw/stores_scraped.csv` when it exists and the fallback list otherwise. Override with `STORES_FILE=path.csv`.

### 7. Train, evaluate, validate and render the map

```bash
make train_eval_pipeline
```

| Target | Script | Output |
|---|---|---|
| `train_model` | `modeling/train.py` | `models/results/oof_scores.parquet`, `coefficients.csv`, `metrics.csv`, `whitespace_lr.pkl` |
| `eval_model` | `modeling/evaluation.py` | `models/eval/county_scores.csv`, `whitespace_open.csv`, `whitespace_fill.csv` |
| `validate_model` | `modeling/validate.py` | `models/eval/validation_loo.csv`, `sensitivity_radius.csv` |
| `render_report` | `modeling/report.py` | `reports/lelabo_whitespace.html` |

Each target also saves its console output with `tee` to a `.txt` next to its results (`lr_train.txt`, `lr_eval.txt`, `validation.txt`, `report.txt`).

**Validation.** Leave-one-out hides each store county from the model in turn, refits, and ranks the hidden county against every county without a store. A low rank means the model would have flagged that market before Le Labo opened there. On the fallback list, the median rank is 9 of 3,090, and 18 of 22 store counties rank in the top 25. The radius check lists the top open markets at 30, 60 and 100 miles.

### 8. Review and share

```bash
make results                   # lists outputs and the map path
make package                   # zips models/ and reports/ to lelabo_whitespace_output.zip
```

Open `reports/lelabo_whitespace.html` in any browser. Fonts and the D3 library load from Google Fonts and cdnjs; everything else is embedded.

### One command

```bash
make scrape_preproc_train_eval QUERY="type == 'Boutique'"   # scrape + steps 6 and 7
make preproc_train_eval                                     # steps 6 and 7 with existing stores
```

## Make variables

Set on the command line, e.g. `make train_eval_pipeline RADIUS=40`.

| Variable | Default | Meaning |
|---|---|---|
| `QUERY` | empty | pandas query to filter stores in `data_prep_preprocessing` |
| `STORES_FILE` | empty | store CSV; empty means scraped if present, else fallback |
| `RADIUS` | `60` | miles from a store county that count as already served |
| `RADII` | `30 60 100` | radii compared in `validate_model` |
| `CV_REPEATS` | `20` | repeats of stratified 5-fold CV in `train_model` |
| `LOO_REPEATS` | `5` | CV repeats per refit in `validate_model` |
| `PYTHON_INTERPRETER` | `python` | use `python3` if your system has no `python` |
| `BROWSER_FLAGS` | `--with-deps` | Playwright install flags |

Fixed settings (random state 222, 5 folds, L2 penalty `C=0.5`, feature list, report table sizes) live in `core/config.py`.

## Cleaning up

| Target | Removes |
|---|---|
| `clean_models` | everything in `models/results`, `models/eval`, `reports` except `.gitkeep` |
| `clean_data` | everything in `data/raw`, `data/interim`, `data/processed` except `.gitkeep` (keeps `data/external`) |
| `clean` | compiled Python files |
| `clean_venv` | `lelabo_venv/` |

## Running in Google Colab

Colab has its own Python and root access, so skip the environment step:

```python
!unzip -oq lelabo-whitespace.zip
%cd lelabo-whitespace
!make requirements install_browsers
!make scrape_preproc_train_eval        # or: !make preproc_train_eval
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
python preprocessing/data_gen.py
python preprocessing/scrape_stores.py
python preprocessing/preprocessing.py --query "type == 'Boutique'"
python preprocessing/feat_gen.py
python modeling/train.py
python modeling/evaluation.py
python modeling/validate.py
python modeling/report.py
```

Every script has `--help` and defaults to the paths in `core/config.py`. `python -m preprocessing.data_gen` (and so on) also works from the root without setting `PYTHONPATH`.

## Method

- **Stores to counties.** Scraped stores are placed in counties by point-in-polygon on Census county boundaries. Non-US stores drop out.
- **Features.** Log population, log density, log median household income, college share, foreign-born share, age 29 and under, age 65 and over, rural share (ACS 2012–2016 five-year estimates, via the MIT Election Data and Science Lab county file).
- **Model.** L2-penalized logistic regression on standardized features. Scores are out-of-fold predictions averaged over 20 repeats of stratified 5-fold CV, so no county is scored by a model that saw its own label.
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
| `Store file not found` | Copy `data/external/stores_fallback.csv` from the project zip, or pass `STORES_FILE=` |
