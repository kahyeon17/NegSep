#!/bin/sh
# WaterBirds -> PlacesBG
# Set NEGSEP_DATA to the dataset root (see README).
python src/eval.py --in_dataset_name waterbird --ood_dataset_name_list placesbg --output_folder output/waterbird "$@"
