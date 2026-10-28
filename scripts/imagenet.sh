#!/bin/sh
# ImageNet-1K -> iNaturalist, SUN, Places, Textures
# Set NEGSEP_DATA to the dataset root (see README).
python src/eval.py --in_dataset_name imagenet --ood_dataset_name_list inaturalist sun places dtd --output_folder output/imagenet "$@"
