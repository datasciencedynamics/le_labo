#!/usr/bin/env python3
"""
data_gen.py
-----------
Downloads the raw county inputs: Census county boundaries (GeoJSON) and ACS
2012-2016 county demographics (CSV). Files already present are reused unless
--force is passed.

Example usage:
    python preprocessing/data_gen.py \
        --geojson-file ./data/raw/counties.geojson \
        --demographics-file ./data/raw/county_demographics.csv
"""

from pathlib import Path

import typer

from core.config import RAW_DATA_DIR
from core.constants import counties_geojson, demog_url, demographics_csv, geo_url
from core.functions import download_file

app = typer.Typer(add_completion=False)


@app.command()
def main(
    geojson_file: Path = typer.Option(
        RAW_DATA_DIR / counties_geojson,
        "--geojson-file",
        help="Where to save the county boundaries.",
    ),
    demographics_file: Path = typer.Option(
        RAW_DATA_DIR / demographics_csv,
        "--demographics-file",
        help="Where to save the county demographics.",
    ),
    force: bool = typer.Option(
        False, "--force/--no-force", help="Re-download even if files exist."
    ),
) -> None:
    """Fetch the raw county datasets into data/raw."""

    ############################################################################
    # Step 1. Download county boundaries
    ############################################################################
    download_file(geo_url, geojson_file, force=force)

    ############################################################################
    # Step 2. Download county demographics
    ############################################################################
    download_file(demog_url, demographics_file, force=force)

    typer.echo("Data generation complete.")


if __name__ == "__main__":
    app()
