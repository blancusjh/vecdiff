PYTHON ?= ../../.venv/bin/python
BUILD_DIR := ../../.cache/manuscript_fresnel
DOCUMENT := direction_resolved_fresnel
DATA := ../../validation/reference_data

.PHONY: all figure
all: $(DOCUMENT).pdf

figure:
	MPLCONFIGDIR=../../.cache/matplotlib $(PYTHON) make_figure.py

fresnel_comparison.png: make_figure.py $(DATA)/focal_pipeline.npz $(DATA)/focal_spectral_N280_a18_c1e-07_t3.npz
	MPLCONFIGDIR=../../.cache/matplotlib $(PYTHON) make_figure.py

$(DOCUMENT).pdf: $(DOCUMENT).tex fresnel_comparison.png
	mkdir -p $(BUILD_DIR)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $(DOCUMENT).tex
	cp $(BUILD_DIR)/$(DOCUMENT).pdf $@
