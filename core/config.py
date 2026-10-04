from pathlib import Path

from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
SCRAPE_DIR = RAW_DATA_DIR / "scrape"  # locator debug output

MODELS_DIR = PROJ_ROOT / "models"
RESULTS_DIR = MODELS_DIR / "results"
EVAL_DIR = MODELS_DIR / "eval"

REPORTS_DIR = PROJ_ROOT / "reports"
TEMPLATES_DIR = PROJ_ROOT / "core" / "templates"

################################################################################
############################ Global Constants ##################################
################################################################################

rstate = 222  # random state for reproducibility

# Miles from a store county's centroid that count as already served. Counties
# without a store inside this radius are "open"; inside it, "fill-in".
radius_mi = 60
sensitivity_radii = [30, 60, 100]

# Out-of-fold scoring: repeats of stratified k-fold, averaged
cv_folds = 5
cv_repeats = 20
loo_repeats = 5  # fewer repeats per refit inside leave-one-out

top_n_open = 12  # open markets shown in the report
top_n_fill = 8  # fill-in markets shown in the report

min_store_counties = 5  # below this the model is not fit

################################################################################
############################ Feature Definitions ###############################
################################################################################

features = [
    "log_pop",
    "log_density",
    "log_income",
    "college_pct",
    "foreignborn_pct",
    "age29andunder_pct",
    "age65andolder_pct",
    "rural_pct",
]

feature_labels = {
    "log_pop": "Population (log)",
    "log_density": "Density (log)",
    "log_income": "Median income (log)",
    "college_pct": "College share",
    "foreignborn_pct": "Foreign-born share",
    "age29andunder_pct": "Age 29 and under",
    "age65andolder_pct": "Age 65 and over",
    "rural_pct": "Rural share",
}

################################################################################
########################## Logistic Regression #################################
################################################################################

lr_name = "lr"
lr_C = 0.5  # L2 penalty strength (smaller = stronger shrinkage)


def make_model() -> Pipeline:
    """Standardize, then L2-penalized logistic regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (lr_name, LogisticRegression(C=lr_C, max_iter=5000)),
        ]
    )
