# Makefile
# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
PROJECT_NAME = lelabo_whitespace
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
VENV_DIR = lelabo_venv
CONDA_ENV_NAME = conda_lelabo
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIRECTORY := $(abspath $(MAKEFILE_DIR))

# Scripts import `core`, so the project root must be importable
export PYTHONPATH := $(PROJECT_DIRECTORY):$(PYTHONPATH)

# Fail a recipe when any command in a pipe fails (keeps `| tee` from hiding errors)
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

RAW_GEOJSON     ?= $(PROJECT_DIRECTORY)/data/raw/counties.geojson
RAW_DEMOG       ?= $(PROJECT_DIRECTORY)/data/raw/county_demographics.csv
DATA_GEN_SCRIPT  = $(PROJECT_DIRECTORY)/preprocessing/data_gen.py

# Store CSV for preprocessing. Empty = data/raw/stores_scraped.csv if present,
# else data/external/stores_fallback.csv
STORES_FILE ?=
# pandas query to keep boutiques only, e.g. QUERY="type == 'Boutique'"
QUERY ?=
# Playwright browser install flags (use BROWSER_FLAGS= to skip system deps)
BROWSER_FLAGS ?= --with-deps

############################## Training Globals ################################

# Repeats of stratified 5-fold CV averaged into the out-of-fold scores
CV_REPEATS ?= 20
# CV repeats per refit inside leave-one-out validation
LOO_REPEATS ?= 5
# Miles from a store county that count as already served (fill-in vs open)
RADIUS ?= 60
# Radii compared in the sensitivity check
RADII ?= 30 60 100


# ------------------------------------------------------------------------------
# COMMANDS
# ------------------------------------------------------------------------------

## Print the command to create the conda environment
create_conda_env:
	@echo "Run 'conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)' to create conda environment"

## Create a virtual environment
create_venv:
	# Create the virtual environment using the specified Python version
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Virtual environment created with $(PYTHON_INTERPRETER)$(PYTHON_VERSION)"

## Print the commands to activate the environment
activate_venv:
	@echo "Run 'conda deactivate' to deactivate the $(CONDA_ENV_NAME) conda environment"
	@echo "Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment"

## Remove the virtual environment
clean_venv:
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Download headless Chromium for the scraper
.PHONY: install_browsers
install_browsers:
	$(PYTHON_INTERPRETER) -m playwright install $(BROWSER_FLAGS) chromium

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

## Create folders, then print environment setup commands
setup_dir_venv: create_folders create_conda_env create_venv activate_venv

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

################################################################################
################################ Folder Creation  ##############################
################################################################################

## Create data, models and reports folders
.PHONY: create_folders
create_folders:
# Create data subdirectories
	mkdir -p data/external data/interim data/processed data/raw/scrape
	mkdir -p models/results models/eval reports
	mkdir -p core preprocessing modeling notebooks

	touch data/external/.gitkeep data/interim/.gitkeep data/processed/.gitkeep data/raw/.gitkeep
	touch models/results/.gitkeep models/eval/.gitkeep reports/.gitkeep

	touch core/__init__.py preprocessing/__init__.py modeling/__init__.py

################################################################################
####################### Preprocessing (+) Dataprep Pipeline ####################
################################################################################

## Download county boundaries and demographics to data/raw
.PHONY: data_gen
data_gen: $(RAW_DEMOG)

$(RAW_DEMOG): $(DATA_GEN_SCRIPT)
	$(PYTHON_INTERPRETER) $(DATA_GEN_SCRIPT) \
		--geojson-file $(RAW_GEOJSON) \
		--demographics-file $(RAW_DEMOG)
	@touch $@

## Scrape store locations from the official Le Labo locator
.PHONY: scrape_stores
scrape_stores: create_folders
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/scrape_stores.py \
		--output-data-file ./data/raw/stores_scraped.csv \
		--debug-dir ./data/raw/scrape \
	2>&1 | tee data/raw/scrape/scrape_stores.txt

## Build the county table and assign stores to counties (use QUERY=...)
.PHONY: data_prep_preprocessing
data_prep_preprocessing: create_folders
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
		--geojson-file $(RAW_GEOJSON) \
		--demographics-file $(RAW_DEMOG) \
		$(if $(STORES_FILE),--stores-file $(STORES_FILE)) \
		$(if $(QUERY),--query "$(QUERY)") \
		--counties-output-file ./data/interim/counties.parquet \
		--stores-output-file ./data/processed/stores.parquet \
	2>&1 | tee data/processed/preprocessing.txt

## Generate features X, labels y and county metadata
.PHONY: feat_gen
feat_gen: create_folders
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
		--counties-file ./data/interim/counties.parquet \
		--stores-file ./data/processed/stores.parquet \
		--data-path ./data/processed \
	2>&1 | tee data/processed/feat_gen.txt

## Delete generated data (keeps data/external and .gitkeep files)
.PHONY: clean_data
clean_data:
	find data/raw data/interim data/processed -type f ! -name .gitkeep -delete
	rm -rf data/raw/scrape

## Download, preprocess and generate features
preproc_pipeline: data_gen scrape_stores data_prep_preprocessing feat_gen

################################################################################
################################# Training #####################################
################################################################################

## Out-of-fold scores, coefficients and final model
train_model:
	@mkdir -p models/results
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/train.py \
		--features-path ./data/processed/X.parquet \
		--labels-path ./data/processed/y.parquet \
		--results-path ./models/results \
		--repeats $(CV_REPEATS) \
	2>&1 | tee models/results/lr_train.txt

################################################################################
############################## Model Evaluation ################################
################################################################################

## Segment counties and write the whitespace tables
eval_model:
	@mkdir -p models/eval
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
		--scores-path ./models/results/oof_scores.parquet \
		--labels-path ./data/processed/y.parquet \
		--meta-path ./data/processed/county_meta.parquet \
		--eval-path ./models/eval \
		--radius $(RADIUS) \
	2>&1 | tee models/eval/lr_eval.txt

## Leave-one-out validation and radius sensitivity
validate_model:
	@mkdir -p models/eval
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/validate.py \
		--features-path ./data/processed/X.parquet \
		--labels-path ./data/processed/y.parquet \
		--meta-path ./data/processed/county_meta.parquet \
		--eval-path ./models/eval \
		--repeats $(LOO_REPEATS) \
		$(foreach r,$(RADII),--radii $(r)) \
	2>&1 | tee models/eval/validation.txt

## Render the HTML whitespace map to reports/
render_report:
	@mkdir -p reports
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/report.py \
		--output-file ./reports/lelabo_whitespace.html \
		--radius $(RADIUS) \
	2>&1 | tee reports/report.txt

## Delete model outputs and reports (keeps .gitkeep files)
.PHONY: clean_models
clean_models:
	find models/results models/eval reports -type f ! -name .gitkeep -delete

## Train, evaluate, validate and render the report
train_eval_pipeline: train_model eval_model validate_model render_report

################################################################################
########## Preprocessing, Feature Generation, Training and Evaluation ##########
################################################################################

# These pipelines run the full preprocessing, training, evaluation, validation
# and report in one command

## Full pipeline from existing stores (scraped if present, else fallback)
preproc_train_eval: preproc_pipeline train_eval_pipeline

## Full pipeline including a fresh scrape of the store locator
scrape_preproc_train_eval: scrape_stores preproc_train_eval

################################################################################
#################################### Outputs ###################################
################################################################################

## List model outputs and the report path
.PHONY: results
results:
	@ls -1 models/results models/eval reports | grep -v '.gitkeep'
	@echo ""
	@echo "Map: $(PROJECT_DIRECTORY)/reports/lelabo_whitespace.html"

## Zip models/ and reports/ for sharing
.PHONY: package
package:
	zip -qr $(PROJECT_NAME)_output.zip models reports -x '*.gitkeep'
	@echo "Wrote $(PROJECT_NAME)_output.zip"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]*?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:28}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
