"""Cross-validated whitespace model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["log_pop", "log_density", "log_income", "college_pct", "foreignborn_pct",
            "age29andunder_pct", "age65andolder_pct", "rural_pct"]


@dataclass
class Result:
    scored: pd.DataFrame      # every county with p, stores, distance, segment
    coef: pd.Series           # standardized coefficients from the full-data fit
    auc: float
    ap: float
    base_rate: float


def _miles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la1, lo1 = np.radians(a[:, None, 0]), np.radians(a[:, None, 1])
    la2, lo2 = np.radians(b[None, :, 0]), np.radians(b[None, :, 1])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(h))


def fit(counties: pd.DataFrame, store_counts: pd.Series, *, C: float = 0.5,
        repeats: int = 20, radius_mi: float = 60, seed: int = 0) -> Result:
    """store_counts: Series indexed by county FIPS with number of stores."""
    d = counties.copy()
    d["stores"] = d.fips.map(store_counts).fillna(0).astype(int)
    d["y"] = (d.stores > 0).astype(int)
    if d.y.sum() < 5:
        raise ValueError("Need at least 5 store counties to fit the model.")

    X, y = d[FEATURES].values, d.y.values
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    folds = min(5, int(y.sum()))
    preds = [
        cross_val_predict(model, X, y, method="predict_proba",
                          cv=StratifiedKFold(folds, shuffle=True, random_state=seed + r))[:, 1]
        for r in range(repeats)
    ]
    d["p"] = np.mean(preds, axis=0)

    model.fit(X, y)
    coef = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values(ascending=False)

    store_pts = d.loc[d.y == 1, ["lat", "lon"]].values
    d["mi_to_store"] = _miles(d[["lat", "lon"]].values, store_pts).min(axis=1)
    d["segment"] = np.select([d.y == 1, d.mi_to_store < radius_mi], ["store", "fill"], "open")
    d["rank"] = d.p.rank(ascending=False, method="first").astype(int)

    return Result(d.sort_values("p", ascending=False).reset_index(drop=True), coef,
                  roc_auc_score(y, d.p), average_precision_score(y, d.p), y.mean())
