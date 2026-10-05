"""
pitch_pdf.py
------------
Builds the client-facing PDF report from the Dash export folder.

Steps:
    1. Screenshot the county map from the rendered HTML report (render_report)
       into <data-dir>/map.png for the cover and map pages.
    2. Build the report from <data-dir> (export_dash) with pitch_report.py.
    3. Write the PDF to --output-file and copy it into <data-dir>, where the
       Dash app's Download PDF report button serves it.

Needs Playwright's Chromium (make install_browsers).

Example usage:
    python modeling/pitch_pdf.py --data-dir ./reports/dash_data
    python modeling/pitch_pdf.py --county 48453      # add a county spotlight page
"""

import shutil
from pathlib import Path

import typer
from loguru import logger

from core.config import REPORTS_DIR
from core.constants import report_file
from modeling.pitch_report import build_html, capture_map, html_to_pdf

app = typer.Typer(add_completion=False)

PDF_NAME = "Le_Labo_US_Whitespace_Analysis.pdf"


@app.command()
def main(
    data_dir: Path = REPORTS_DIR / "dash_data",
    report_html: Path = REPORTS_DIR / report_file,
    output_file: Path = REPORTS_DIR / PDF_NAME,
    top_n: int = typer.Option(12, min=5, max=15, help="Rows in the open-market table"),
    county: str = typer.Option("", help="5-digit FIPS for a County spotlight page"),
) -> None:

    ############################################################################
    # Step 1. Map image from the HTML report
    ############################################################################
    if not (data_dir / "counties.csv").exists():
        typer.secho(f"No export in {data_dir}. Run `make export_dash` first.", fg="red")
        raise typer.Exit(code=1)
    if not report_html.exists():
        typer.secho(f"{report_html} not found. Run `make render_report` first.", fg="red")
        raise typer.Exit(code=1)

    map_png = capture_map(report_html, data_dir / "map.png")
    logger.info(f"Map image: {map_png}")

    ############################################################################
    # Step 2. Report HTML -> PDF
    ############################################################################
    html = build_html(data_dir, map_png, top_n=top_n, county=county or None)
    (output_file.with_suffix(".html")).write_text(html, encoding="utf-8")
    html_to_pdf(html, output_file)

    ############################################################################
    # Step 3. Copy next to the Dash data
    ############################################################################
    shutil.copyfile(output_file, data_dir / PDF_NAME)
    print(f"PDF: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB), copied to {data_dir}")
    logger.success("Pitch PDF complete.")


if __name__ == "__main__":
    app()
