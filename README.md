# lelabo-whitespace

Scrapes Le Labo's US store locations, scores every US county on how closely it matches the demographic profile of counties that already host a boutique, and renders a self-contained HTML map of the whitespace.

## Install

```bash
pip install ./lelabo-whitespace            # model + report
pip install "./lelabo-whitespace[scrape]"  # adds Playwright for scraping
playwright install --with-deps chromium    # one-time browser download
```

## Run

**Command line**

```bash
lelabo-whitespace --scrape                  # scrape the official locator, then model
lelabo-whitespace                           # use the bundled store list (no scraping)
lelabo-whitespace --stores my_stores.csv    # bring your own list
lelabo-whitespace --scrape --query "type == 'Boutique'"   # filter scraped rows
```

**Jupyter / Colab** (notebooks already run an event loop, so use `await`)

```python
import lelabo_whitespace as lw
res = await lw.run_with_scrape()    # scrape + model + map
# or
res = lw.run()                      # bundled store list
```

`notebooks/colab_quickstart.ipynb` has the full sequence, including installs.

## Outputs (in `output/`)

| File | Contents |
|---|---|
| `lelabo_whitespace.html` | Interactive county map, ranked tables, coefficients, method notes |
| `county_scores.csv` | Every county with model score, rank, store count, miles to nearest store county, segment |
| `whitespace_open.csv` | High-scoring counties with no store within the radius (default 60 mi) |
| `whitespace_fill.csv` | High-scoring counties near an existing store |
| `coefficients.csv` | Standardized logistic regression coefficients |
| `stores_used.csv` | The store list the model was fit on |
| `scrape/` | Raw locator responses, captured endpoint URLs, rendered page |

## Method

- **Stores to counties.** Scraped stores are placed in counties by point-in-polygon on Census county boundaries. Non-US locations drop out automatically.
- **Features.** Log population, log density, log median household income, college share, foreign-born share, age 29 and under, age 65 and over, rural share (ACS 2012–2016 five-year, via the MIT Election Data and Science Lab county file).
- **Model.** L2-penalized logistic regression on standardized features. Scores are out-of-fold predictions averaged over 20 repeats of stratified 5-fold CV, so no county is scored by a model that saw its own label.
- **Segments.** Counties with a store, *fill-in* (no store but within the radius of a store county centroid), and *open* (no store within the radius).

## Scraping notes

The locator renders with JavaScript, so the scraper opens it in headless Chromium and captures every JSON response, keeping records with latitude/longitude. If it finds nothing, check `output/scrape/endpoints.txt` or open the locator in Chrome DevTools (Network > Fetch/XHR) to find the API call. The scraped data may include department-store counters; inspect `output/scrape/stores_raw.csv` and pass `--query` to keep boutiques only. Review the site's terms before scraping.

## Limits

Population dominates the fit, so a high AUC mostly reflects that large urban counties have stores; the useful signal is the ranking among similar-sized counties. Counties are coarse, and a real site study would use tract-level data and foot traffic. Demographics are dated (2012–2016).
