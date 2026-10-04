################################################################################
############################# Source URLs ######################################
################################################################################

# Census county boundaries (GeoJSON, FIPS-keyed)
geo_url = (
    "https://raw.githubusercontent.com/plotly/datasets/master/"
    "geojson-counties-fips.json"
)

# ACS 2012-2016 county demographics (MIT Election Data and Science Lab)
demog_url = (
    "https://raw.githubusercontent.com/MEDSL/2018-elections-unoffical/"
    "master/election-context-2018.csv"
)

# Official Le Labo store locator (renders with JavaScript)
locator_url = "https://www.lelabofragrances.com/store-search.html"

user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

################################################################################
############################# File Names #######################################
################################################################################

counties_geojson = "counties.geojson"  # data/raw
demographics_csv = "county_demographics.csv"  # data/raw
stores_scraped = "stores_scraped.csv"  # data/raw
stores_fallback = "stores_fallback.csv"  # data/external

counties_interim = "counties.parquet"  # data/interim
stores_processed = "stores.parquet"  # data/processed
stores_source = "stores_source.txt"  # data/processed
features_file = "X.parquet"  # data/processed
labels_file = "y.parquet"  # data/processed
meta_file = "county_meta.parquet"  # data/processed

model_file = "whitespace_lr.pkl"  # models/results
oof_scores_file = "oof_scores.parquet"  # models/results
coef_file = "coefficients.csv"  # models/results
metrics_file = "metrics.csv"  # models/results

report_file = "lelabo_whitespace.html"  # reports

################################################################################
########################## Variable/DataFrame Constants ########################
################################################################################

var_index = "fips"  # county FIPS code, index of every county-level frame
target_outcome = "has_store"  # 1 if the county hosts at least one boutique
store_count = "stores"  # number of boutiques in the county

demog_cols = [
    "state",
    "county",
    "fips",
    "total_population",
    "foreignborn_pct",
    "age29andunder_pct",
    "age65andolder_pct",
    "median_hh_inc",
    "lesscollege_pct",
    "rural_pct",
]

# Columns kept for reporting alongside the model scores
meta_cols = [
    "label",
    "county",
    "state",
    "lat",
    "lon",
    "total_population",
    "median_hh_inc",
    "college_pct",
]

# Key names a scraped store record may use for coordinates
lat_keys = {"lat", "latitude", "geo_lat", "lat_coord"}
lon_keys = {"lng", "lon", "long", "longitude", "geo_lng", "lng_coord"}

state_abbr = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
