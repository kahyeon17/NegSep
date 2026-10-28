#!/bin/sh
# ImageNet-10 -> ImageNet-20, ImageNet-99
# Set NEGSEP_DATA to the dataset root (see README).
python src/eval.py --in_dataset_name imagenet10 --ood_dataset_name_list imagenet20 imagenet99 --output_folder output/img10 "$@"
