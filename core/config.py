import numpy as np

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from core.constants import (
    exp_artifact_name,
    preproc_run_name,
    mlflow_artifacts_data,
    features,
)
from core.functions import mlflow_loadArtifact

# Load environment variables from .env file if it exists
load_dotenv()

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
FIGURES_DIR = REPORTS_DIR / "figures"
TEMPLATES_DIR = PROJ_ROOT / "core" / "templates"

features_path = PROCESSED_DATA_DIR / "X.parquet"

################################################################################
############################ Global Constants ##################################
################################################################################

rstate = 222  # random state for reproducibility
threshold_target_metric = "youden"  # "precision", "recall", or "youden"
target_precision = 0.5  # unused when threshold_target_metric == "youden"

sampler_definitions = {
    "None": None,
    "SMOTE": SMOTE(random_state=rstate),
    "RandomUnderSampler": RandomUnderSampler(random_state=rstate),
}

################################################################################
############################ Whitespace Settings ###############################
################################################################################

# Miles from a store county's centroid that count as already served. Counties
# without a store inside this radius are "open"; inside it, "fill-in".
radius_mi = 60
sensitivity_radii = [30, 60, 100]

# Out-of-fold scoring of every county with the tuned champion pipeline:
# repeats of stratified k-fold, averaged, sigmoid-calibrated inside each fold
cv_folds = 5
cv_repeats = 20
loo_repeats = 5  # fewer repeats per refit inside leave-one-out
oof_calibration_cv = 3  # inner folds for the per-fold sigmoid calibration

top_n_open = 12  # open markets shown in the report
top_n_fill = 8  # fill-in markets shown in the report

min_store_counties = 5  # below this the model is not fit

# Large-county subgroup for evaluation: separating New York from rural counties
# is trivial, so discrimination is also reported among counties at or above
# this population, where the real siting decisions are made
large_county_pop = 500_000
n_bootstrap = 1000  # bootstrap resamples for 95% bands and CIs

# Footer line on the HTML report
report_authors = "Leon Shpaner, M.S."

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
############################ Column Definitions ################################
################################################################################

# All engineered county features are numeric
categorical_cols = []

# Load feature column names from Mlflow. Before feat_gen.py has run (e.g. while
# downloading or preprocessing) the artifact does not exist yet, so fall back to
# the static feature list instead of failing on import.
X_columns_list = None
if Path(mlflow_artifacts_data).exists():
    try:
        X_columns_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,  # Use the same run_name as training
            obj_name="X_columns_list",
            verbose=False,
        )
    except Exception as e:
        logger.warning(f"X_columns_list not in MLflow yet ({type(e).__name__}).")

if X_columns_list is None:
    X_columns_list = list(features)

# Subset the numerical columns only; categorical columns defined above
numerical_cols = [col for col in X_columns_list if col not in categorical_cols]


################################################################################
############################### Transformers ###################################
################################################################################

# Impute first, then scale. Scaling before imputation lets NaNs pass through
# untouched and makes the fill value dependent on transformer ordering.
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
    ],
)

################################################################################
################################ Pipelines #####################################
################################################################################

pipeline_scale_imp = [
    ("Preprocessor", preprocessor),
]

pipelines = {
    "orig": {
        "pipeline": pipeline_scale_imp,
        "sampler": None,
        "feature_selection": False,
    },
    "smote": {
        "pipeline": pipeline_scale_imp,
        "sampler": SMOTE(random_state=rstate),
        "feature_selection": False,
    },
    "under": {
        "pipeline": pipeline_scale_imp,
        "sampler": RandomUnderSampler(random_state=rstate),
        "feature_selection": False,
    },
}


################################################################################
########################## Logistic Regression #################################
################################################################################

# Define the hyperparameters for Logistic Regression
lr_name = "lr"

lr_Cs = np.logspace(-5, 0, 5)

tuned_parameters_lr = [
    {
        "lr__C": lr_Cs,
    }
]

lr = LogisticRegression(
    class_weight="balanced",
    random_state=rstate,
    solver="lbfgs",
    max_iter=1000,
)

lr_definition = {
    "clc": lr,
    "estimator_name": lr_name,
    "tuned_parameters": tuned_parameters_lr,
    "randomized_grid": True,
    "n_iter": 5,
    "early": False,
}
################################################################################
########################## Random Forest Classifier ############################
################################################################################

# Define the hyperparameters for Random Forest
rf_name = "rf"

rf_n_estimators = [100, 200, 300, 400, 500]
rf_max_depths = [None, 5, 10, 15, 20]
rf_criterions = ["gini", "entropy"]
rf_parameters = [
    {
        "rf__n_estimators": rf_n_estimators,
        "rf__max_depth": rf_max_depths,
        "rf__criterion": rf_criterions,
    }
]

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=rstate,
    n_jobs=-2,
)

rf_definition = {
    "clc": rf,
    "estimator_name": rf_name,
    "tuned_parameters": rf_parameters,
    "randomized_grid": True,
    "n_iter": 30,
    "early": False,
}

################################################################################
############################## XGBoost Classifier ##############################
################################################################################

# Estimator name prefix for use in GridSearchCV or similar tools
xgb_name = "xgb"

xgb = XGBClassifier(
    objective="binary:logistic",
    random_state=rstate,
    tree_method="hist",
    device="cpu",  # "cuda" to train on GPU; 3k rows train fast on CPU
    n_jobs=16,
)

# Define the hyperparameters for XGBoost
xgb_learning_rates = [0.01]  # Learning rate or eta
xgb_n_estimators = [10000]  # Number of trees
xgb_max_depths = [2, 3, 4, 5]  # Maximum depth of the trees
xgb_subsamples = [0.8, 1.0]  # Subsample ratio of the training instances
xgb_colsample_bytree = [0.8, 1.0]
xgb_alpha = [0, 0.1, 1, 10]  # L1 regularization (alpha)
xgb_lambda = [0, 0.1, 10, 100]  # L2 regularization (lambda)
xgb_eval_metric = ["logloss"]  # check out "aucpr"
xgb_early_stopping_rounds = [10]
xgb_verbose = [0]
# Subsample ratio of columns when constructing each tree

# Combining the hyperparameters in a dictionary
xgb_parameters = [
    {
        "xgb__learning_rate": xgb_learning_rates,
        "xgb__n_estimators": xgb_n_estimators,
        "xgb__max_depth": xgb_max_depths,
        "xgb__subsample": xgb_subsamples,
        "xgb__alpha": xgb_alpha,  # L1 regularization (alpha)
        "xgb__lambda": xgb_lambda,  # L2 regularization (lambda)
        "xgb__colsample_bytree": xgb_colsample_bytree,
        "xgb__eval_metric": xgb_eval_metric,
        "xgb__early_stopping_rounds": xgb_early_stopping_rounds,
        "xgb__verbose": xgb_verbose,
    }
]

xgb_definition = {
    "clc": xgb,
    "estimator_name": xgb_name,
    "tuned_parameters": xgb_parameters,
    "randomized_grid": True,
    "n_iter": 30,
    "early": True,
}

################################################################################
############################ CatBoost Classifier ###############################
################################################################################

cat_name = "cat"

cat = CatBoostClassifier(
    task_type="CPU",
    random_state=rstate,
    eval_metric="Logloss",
)

# Define the hyperparameters for CatBoost
cat_depths = [4, 6, 8, 10]  # Depth of the trees
cat_learning_rates = [0.01]  # Learning rate
cat_l2_leaf_regs = [3, 10, 100]  # L2 regularization
cat_bagging_temperatures = [0, 0.5, 1]  # Bagging temperature
cat_n_estimators = [10000]  # Number of trees
cat_early_stopping_rounds = [10]  # Early stopping rounds
cat_random_strengths = [1, 10]  # Random strength for feature randomness
cat_verbose = [0]  # Verbosity level

# Combining the hyperparameters in a dictionary
cat_parameters = [
    {
        "cat__depth": cat_depths,
        "cat__learning_rate": cat_learning_rates,
        "cat__l2_leaf_reg": cat_l2_leaf_regs,
        "cat__bagging_temperature": cat_bagging_temperatures,
        "cat__n_estimators": cat_n_estimators,
        "cat__early_stopping_rounds": cat_early_stopping_rounds,
        "cat__random_strength": cat_random_strengths,
        "cat__verbose": cat_verbose,
    }
]

cat_definition = {
    "clc": cat,
    "estimator_name": cat_name,
    "tuned_parameters": cat_parameters,
    "randomized_grid": True,
    "n_iter": 30,
    "early": True,
}


model_definitions = {
    lr_name: lr_definition,
    rf_name: rf_definition,
    xgb_name: xgb_definition,
    cat_name: cat_definition,
}
