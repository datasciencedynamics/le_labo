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

# MLflow >= 3.x refuses the local file store (./mlruns) without this
export MLFLOW_ALLOW_FILE_STORE := true

# Fail a recipe when any command in a pipe fails (keeps `| tee` from hiding errors)
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

RAW_GEOJSON     ?= $(PROJECT_DIRECTORY)/data/raw/counties.geojson
RAW_DEMOG       ?= $(PROJECT_DIRECTORY)/data/raw/county_demographics.csv
SCRAPED_STORES  ?= $(PROJECT_DIRECTORY)/data/raw/stores_scraped.csv
SCRAPE_SCRIPT    = $(PROJECT_DIRECTORY)/preprocessing/scrape_stores.py
DATA_GEN_SCRIPT  = $(PROJECT_DIRECTORY)/preprocessing/data_gen.py

# Store CSV for preprocessing. Empty = data/raw/stores_scraped.csv if present,
# else data/external/stores_fallback.csv
STORES_FILE ?=
# 1 = preproc_pipeline scrapes the locator when data/raw/stores_scraped.csv is
# missing; 0 = never scrape (use the existing scraped file or the fallback)
SCRAPE ?= 1
# pandas query to keep boutiques only, e.g. QUERY="type == 'Boutique'"
QUERY ?=
# Playwright browser install flags (use BROWSER_FLAGS= to skip system deps)
BROWSER_FLAGS ?= --with-deps

############################## Training Globals ################################

# Define variables for looping
OUTCOMES = has_store
PIPELINES = orig
SCORING = average_precision
# 0 to train the models, 1 to recalibrate pretrained models from MLflow
PRETRAINED ?= 0

############################# Whitespace Globals ###############################

# Tuned model from MLflow used to score counties and draw the map
MAP_MODEL ?= lr
MAP_PIPELINE ?= orig
# Repeats of stratified 5-fold CV averaged into the out-of-fold county scores
CV_REPEATS ?= 20
# CV repeats per refit inside leave-one-out validation
LOO_REPEATS ?= 5
# Miles from a store county that count as already served (fill-in vs open)
RADIUS ?= 60
# Radii compared in the sensitivity check
RADII ?= 30 60 100
# Export folder for the Dash app (make export_dash)
DASH_DIR ?= reports/dash_data


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

## Launch the MLflow UI for the model runs
.PHONY: mlflow_ui
mlflow_ui:
	mlflow ui --backend-store-uri mlruns/models --host 0.0.0.0 --port 5501

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

# Create models subdirectories for each outcome
	@for outcome in $(OUTCOMES); do \
		mkdir -p models/results/$$outcome; \
		mkdir -p models/eval/$$outcome; \
	done

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

## Scrape the store locator now (refreshes data/raw/stores_scraped.csv)
.PHONY: scrape_stores
scrape_stores: create_folders
	$(PYTHON_INTERPRETER) $(SCRAPE_SCRIPT) \
		--output-data-file $(SCRAPED_STORES) \
		--debug-dir ./data/raw/scrape \
	2>&1 | tee data/raw/scrape/scrape_stores.txt

# Used by preproc_pipeline: scrape only when no scraped file exists yet
$(SCRAPED_STORES): | create_folders
	$(PYTHON_INTERPRETER) $(SCRAPE_SCRIPT) \
		--output-data-file $(SCRAPED_STORES) \
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

## Download, scrape (if needed; SCRAPE=0 to skip), preprocess and generate features
preproc_pipeline: data_gen $(if $(filter 1,$(SCRAPE)),$(SCRAPED_STORES)) data_prep_preprocessing feat_gen

################################################################################
################################# Training #####################################
################################################################################

## Train logistic regression (model_tuner, logged to MLflow)
train_logistic_regression:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type lr \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/lr_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

## Train random forest
train_random_forest:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type rf \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/rf_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

## Train XGBoost
train_xgboost:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type xgb \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/xgb_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

## Train CatBoost
train_catboost:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type cat \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/cat_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

## Train all four models
train_all_models: train_logistic_regression train_random_forest train_xgboost train_catboost

################################################################################
############################## Model Evaluation ################################
################################################################################

## Evaluate logistic regression: metrics and plots to MLflow
eval_logistic_regression:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/eval/$$outcome; \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type lr \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/lr_eval_$$pipeline.txt; \
		done; \
	done

## Evaluate random forest
eval_random_forest:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/eval/$$outcome; \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type rf \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/rf_eval_$$pipeline.txt; \
		done; \
	done

## Evaluate XGBoost
eval_xgboost:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/eval/$$outcome; \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type xgb \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/xgb_eval_$$pipeline.txt; \
		done; \
	done

## Evaluate CatBoost
eval_catboost:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/eval/$$outcome; \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type cat \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/cat_eval_$$pipeline.txt; \
		done; \
	done

## Evaluate all four models
eval_all_models: eval_logistic_regression eval_random_forest eval_xgboost eval_catboost

## Compare every logged model side by side
compare_models:
	@for outcome in $(OUTCOMES); do \
		mkdir -p models/eval/$$outcome; \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/compare_models.py \
		--outcome $$outcome 2>&1 | tee models/eval/$$outcome/model_comparison.txt; \
	done

## Train, evaluate and compare all models
train_eval_pipeline: train_all_models eval_all_models compare_models

################################################################################
######################### Whitespace Scoring and Report ########################
################################################################################

## Score every county out-of-fold with the tuned MAP_MODEL from MLflow
score_counties:
	@for outcome in $(OUTCOMES); do \
		mkdir -p models/results/$$outcome models/eval/$$outcome; \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/score_counties.py \
		--model-type $(MAP_MODEL) \
		--pipeline-type $(MAP_PIPELINE) \
		--outcome $$outcome \
		--repeats $(CV_REPEATS) \
		--radius $(RADIUS) \
		2>&1 | tee models/results/$$outcome/$(MAP_MODEL)_score_counties.txt; \
	done

## Leave-one-out validation and radius sensitivity for MAP_MODEL
validate_model:
	@for outcome in $(OUTCOMES); do \
		mkdir -p models/eval/$$outcome; \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/validate.py \
		--model-type $(MAP_MODEL) \
		--pipeline-type $(MAP_PIPELINE) \
		--outcome $$outcome \
		--repeats $(LOO_REPEATS) \
		$(foreach r,$(RADII),--radii $(r)) \
		2>&1 | tee models/eval/$$outcome/$(MAP_MODEL)_validation.txt; \
	done

## Render the HTML whitespace map to reports/
render_report:
	@mkdir -p reports
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/report.py \
		--outcome $(firstword $(OUTCOMES)) \
		--output-file ./reports/lelabo_whitespace.html \
		--radius $(RADIUS) \
	2>&1 | tee reports/report.txt

## Export county scores, shapes and metrics for the Dash app (DASH_DIR=...)
.PHONY: export_dash
export_dash:
	@mkdir -p $(DASH_DIR)
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/export_dash.py \
		--outcome $(firstword $(OUTCOMES)) \
		--output-dir $(DASH_DIR) \
		--radius $(RADIUS) \
	2>&1 | tee reports/export_dash.txt

## Score counties, validate and render the report
whitespace_pipeline: score_counties validate_model render_report

## Delete model outputs and reports (keeps .gitkeep files and mlruns/)
.PHONY: clean_models
clean_models:
	find models/results models/eval reports -type f ! -name .gitkeep -delete

## Delete MLflow tracking data (all logged runs, models and artifacts)
.PHONY: clean_mlruns
clean_mlruns:
	rm -rf mlruns

################################################################################
########## Preprocessing, Feature Generation, Training and Evaluation ##########
################################################################################

# This pipeline runs consecutively the full preprocessing, training,
# evaluation, county scoring, validation and report in one command

## Full pipeline: data, scrape (if needed), features, train, evaluate, map
preproc_train_eval: preproc_pipeline train_all_models eval_all_models whitespace_pipeline compare_models


################################################################################
#################################### Outputs ###################################
################################################################################

## List model outputs and the report path
.PHONY: results
results:
	@ls -R models reports | grep -v '.gitkeep'
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
