"""Command line: lelabo-whitespace [--scrape | --stores FILE] [--out DIR]"""

import argparse

from . import pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lelabo-whitespace",
                                 description="Score US counties for Le Labo whitespace and render an HTML map.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--scrape", action="store_true", help="scrape the official store locator first")
    src.add_argument("--stores", help="CSV of stores (lat/lon or county_fips columns)")
    ap.add_argument("--query", help="pandas query to filter stores, e.g. \"type == 'Boutique'\"")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    ap.add_argument("--radius", type=float, default=60, help="miles that count as served by a store")
    ap.add_argument("--repeats", type=int, default=20, help="CV repeats (default 20)")
    a = ap.parse_args(argv)

    kw = dict(out_dir=a.out, query=a.query, radius_mi=a.radius, repeats=a.repeats)
    if a.scrape:
        pipeline.run_with_scrape_sync(**kw)
    else:
        pipeline.run(a.stores, **kw)


if __name__ == "__main__":
    main()
