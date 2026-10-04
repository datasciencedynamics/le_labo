################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from core.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from core.constants import (
    exp_artifact_name,
    features,
    preproc_run_name,
    counties_interim,
    features_file,
    labels_file,
    meta_cols,
    meta_file,
    store_count,
    stores_processed,
    target_outcome,
    var_index,
)
from core.functions import mlflow_dumpArtifact

################################################################################
################ Step 2: Define Typer Application ##############################
################################################################################

app = typer.Typer(add_completion=False)

################################################################################
################ Step 3: Define Main Function ##################################
################################################################################


@app.command()
def main(
    counties_file: Path = typer.Option(
        INTERIM_DATA_DIR / counties_interim, "--counties-file"
    ),
    stores_file: Path = typer.Option(
        PROCESSED_DATA_DIR / stores_processed, "--stores-file"
    ),
    data_path: Path = typer.Option(PROCESSED_DATA_DIR, "--data-path"),
):
    """
    Generates the feature space X, labels y, and county metadata.

    Outputs (all indexed by county FIPS):
        X.parquet            engineered features
        y.parquet            has_store (0/1) and store count
        county_meta.parquet  label, state, centroid and reporting columns
    """

    ############################################################################
    ################ Step 4: Load Input Data ###################################
    ############################################################################
    counties = pd.read_parquet(counties_file)
    stores = pd.read_parquet(stores_file)
    print(f"Counties: {len(counties):,} | Stores: {len(stores)}")

    ############################################################################
    ################ Step 5: Engineer Features #################################
    ############################################################################
    # Population and density are heavily right-skewed, so both enter on the log
    # scale; income is logged so a coefficient reads as a proportional change.
    df = counties.copy()
    df["log_pop"] = np.log(df.total_population)
    df["log_density"] = np.log(df.total_population / df.area_sqmi)
    df["log_income"] = np.log(df.median_hh_inc)
    df["college_pct"] = 100 - df.lesscollege_pct

    X = df[features].copy()
    print(f"\n{'=' * 80}\nX\n{'=' * 80}\n{X.head()}")
    print(f"\nShape of X: {X.shape}")

    ############### Store Final List of Features for Production ##
    X_columns_list = X.columns.to_list()
    mlflow_dumpArtifact(
        experiment_name=exp_artifact_name,
        run_name=preproc_run_name,  # Consistent run_name for all artifacts
        obj_name="X_columns_list",
        obj=X_columns_list,
    )

    ############################################################################
    ################ Step 6: Generate Target Variable ##########################
    ############################################################################
    counts = stores.county_fips.value_counts()
    y = pd.DataFrame(index=df.index)
    y[target_outcome[0]] = (df.index.map(counts).fillna(0) > 0).astype(int)
    y[store_count] = df.index.map(counts).fillna(0).astype(int)

    print(f"\nBreakdown of y:\n{y[target_outcome[0]].value_counts()}\n")
    print(f"Stores: {int(y[store_count].sum())} in {int(y[target_outcome[0]].sum())} counties")

    ############################################################################
    ################ Step 7: County Metadata for Reporting #####################
    ############################################################################
    meta = df[meta_cols].copy()

    ############################################################################
    ################ Step 8: Save ##############################################
    ############################################################################
    data_path.mkdir(parents=True, exist_ok=True)
    for frame, name in ((X, features_file), (y, labels_file), (meta, meta_file)):
        frame.index.name = var_index
        frame.to_parquet(data_path / name)
        print(f"Saved {data_path / name}")


################################################################################

if __name__ == "__main__":
    app()
