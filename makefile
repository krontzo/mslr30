.POSSIX:
.SUFFIXES:
all: run

.PHONY: all run

run: view_config.py
	python3 $< 
