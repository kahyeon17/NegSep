#!/bin/sh
# ImageNet-99 -> ImageNet-10
# Set NEGSEP_DATA to the dataset root (see README).
python src/eval.py --in_dataset_name imagenet99 --ood_dataset_name_list imagenet10 --output_folder output/img99 "$@"
