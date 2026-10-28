#!/bin/sh
# ImageNet-20 -> ImageNet-10
# Set NEGSEP_DATA to the dataset root (see README).
python src/eval.py --in_dataset_name imagenet20 --ood_dataset_name_list imagenet10 --output_folder output/img20 "$@"
