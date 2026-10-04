################################################################################
######################### Import Requisite Libraries ###########################
################################################################################

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

# import pickling scripts
from model_tuner.pickleObjects import dumpObjects

from core.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from core.constants import (
    exp_artifact_name,
    preproc_run_name,
    counties_geojson,
    counties_interim,
    demog_cols,
    demographics_csv,
    state_abbr,
    stores_fallback,
    stores_processed,
    stores_scraped,
    stores_source,
    var_index,
)
from core.functions import (
    mlflow_dumpArtifact,
    safe_to_numeric,
    assign_counties,
    county_geometry_table,
    extract_coordinates,
    load_geojson,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    geojson_file: Path = typer.Option(RAW_DATA_DIR / counties_geojson, "--geojson-file"),
    demographics_file: Path = typer.Option(
        RAW_DATA_DIR / demographics_csv, "--demographics-file"
    ),
    stores_file: Optional[Path] = typer.Option(
        None,
        "--stores-file",
        help="Store CSV. Default: data/raw/stores_scraped.csv if present, "
        "else data/external/stores_fallback.csv.",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        help="pandas query to keep boutiques only, e.g. \"type == 'Boutique'\".",
    ),
    counties_output_file: Path = typer.Option(
        INTERIM_DATA_DIR / counties_interim, "--counties-output-file"
    ),
    stores_output_file: Path = typer.Option(
        PROCESSED_DATA_DIR / stores_processed, "--stores-output-file"
    ),
):
    """
    Builds the county table and assigns every US store to its county.

    Outputs:
        data/interim/counties.parquet    one row per county, raw demographics,
                                         land area and centroid
        data/processed/stores.parquet    stores with county_fips
        data/processed/stores_source.txt where the store list came from
    """

    ############################################################################
    # Step 1. Read the raw county inputs
    ############################################################################
    geo = load_geojson(geojson_file)
    demog = pd.read_csv(demographics_file)
    print(f"Demographics: {demog.shape[0]:,} rows x {demog.shape[1]} columns")

    ############################################################################
    # Step 2. Keep modeling columns, drop counties with missing values
    ############################################################################
    demog = demog[demog_cols]
    n_before = len(demog)
    demog = demog.dropna()
    print(f"Dropped {n_before - len(demog)} counties with missing demographics.")

    ############################################################################
    # Step 3. Join land area and centroids from the boundaries
    ############################################################################
    counties = demog.merge(county_geometry_table(geo), on=var_index)
    n_zero = int((counties.area_sqmi <= 0).sum())
    counties = counties[counties.area_sqmi > 0].copy()
    print(f"Dropped {n_zero} counties with zero land area.")

    ############################################################################
    # Step 4. Display label, e.g. "Cook, IL"
    ############################################################################
    abbr = counties.state.map(state_abbr).fillna(counties.state)
    counties["label"] = np.where(
        counties.county == counties.state, counties.county, counties.county + ", " + abbr
    )
    counties = counties.set_index(var_index).sort_index()

    ############################################################################
    # Step 5. String Columns Handling
    ############################################################################
    # String columns (state, county, label) are kept for reporting but never
    # reach the model. The list is stored in MLflow for reference only.
    ############################################################################
    counties = counties.apply(lambda x: safe_to_numeric(x))
    string_cols_list = counties.select_dtypes("object").columns.to_list()
    print(f"\nThere are {len(string_cols_list)} string columns: {string_cols_list}")

    processed_dir = stores_output_file.parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    dumpObjects(string_cols_list, str(processed_dir / "string_cols_list.pkl"))
    mlflow_dumpArtifact(
        experiment_name=exp_artifact_name,
        run_name=preproc_run_name,
        obj_name="string_cols_list",
        obj=string_cols_list,
    )

    ############################################################################
    # Step 6. Zero Variance Columns
    ############################################################################
    numeric_cols = counties.select_dtypes(include=["number"]).columns
    var_indf = counties[numeric_cols].var()
    zero_varlist_list = list(var_indf[var_indf == 0].index)

    print("*" * 80)
    print(f"Zero-variance columns: {zero_varlist_list}")
    print("*" * 80)

    dumpObjects(zero_varlist_list, str(processed_dir / "zero_varlist_list.pkl"))
    mlflow_dumpArtifact(
        experiment_name=exp_artifact_name,
        run_name=preproc_run_name,
        obj_name="zero_varlist_list",
        obj=zero_varlist_list,
    )
    counties = counties.drop(columns=zero_varlist_list)

    counties_output_file.parent.mkdir(parents=True, exist_ok=True)
    counties.to_parquet(counties_output_file)
    print(f"Counties: {len(counties):,} -> {counties_output_file}")

    ############################################################################
    # Step 7. Choose the store list
    ############################################################################
    if stores_file is None:
        scraped = RAW_DATA_DIR / stores_scraped
        stores_file = scraped if scraped.exists() else EXTERNAL_DATA_DIR / stores_fallback
    if not stores_file.exists():
        typer.secho(f"Store file not found: {stores_file}", fg="red", err=True)
        raise typer.Exit(code=1)

    is_fallback = stores_file.name == stores_fallback
    stores = pd.read_csv(stores_file)
    print(f"\nStores file: {stores_file} ({len(stores)} rows)")

    ############################################################################
    # Step 8. Filter (e.g. drop department-store counters)
    ############################################################################
    if query:
        stores = stores.query(query)
        print(f"After query [{query}]: {len(stores)} rows")

    ############################################################################
    # Step 9. Assign stores to counties
    ############################################################################
    if "county_fips" not in stores.columns:
        stores = assign_counties(extract_coordinates(stores), geo)

    unmatched = ~stores.county_fips.isin(counties.index)
    if unmatched.any():
        print(f"{int(unmatched.sum())} stores sit in counties without demographics; dropped.")
        stores = stores[~unmatched]

    print(
        f"US stores kept: {len(stores)} in {stores.county_fips.nunique()} counties"
    )

    ############################################################################
    # Step 10. Save stores and a note on their source (shown in the report)
    ############################################################################
    stores_output_file.parent.mkdir(parents=True, exist_ok=True)
    stores.reset_index(drop=True).to_parquet(stores_output_file)

    if is_fallback:
        note = (
            f"{len(stores)} standalone US boutiques compiled from Yelp listings in "
            "October 2026 (bundled fallback list). Department-store counters "
            "excluded; a few boutiques may be missing."
        )
    else:
        note = (
            f"{len(stores)} US locations from the official Le Labo store locator"
            f"{' (filtered: ' + query + ')' if query else ''}, assigned to counties "
            "by point-in-polygon."
        )
    (stores_output_file.parent / stores_source).write_text(note, encoding="utf-8")
    print(f"Stores -> {stores_output_file}")


if __name__ == "__main__":
    app()
