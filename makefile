.POSIX:
.SUFFIXES:
all: run

dataset_files = MSLR-TrainingValidation.zip MSLR-Testing.zip

.PHONY: all run download

run: cinves/view_config.py
	python3 $< 

download: $(dataset_files)

%.zip:
	curl -L "https://github.com/krontzo/mslr30/releases/download/v0.0.0/$@" -O

