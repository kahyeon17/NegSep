# NegSep: Background / Non-Background Negative Label Separation for Zero-Shot OOD Detection

NegSep is a training-free, zero-shot OOD detector built on CLIP and negative labels. It splits the negative label vocabulary into **background** labels (places, landforms, scenes, weather, …) and **non-background** labels, and scores each image against the two pools separately:

$$
S_b(x) = \frac{1}{G}\sum_{g=1}^{G} \frac{\sum_{i \in \mathrm{ID}} e^{s_i(x)}}{\sum_{i \in \mathrm{ID}} e^{s_i(x)} + \sum_{n \in \mathcal{N}_{b,g}} e^{s_n(x)}}, \qquad b \in \{\mathrm{bg}, \mathrm{nbg}\}
$$

where $s(x) = \cos(f(x), t) / \tau$ with $\tau = 0.01$, and $\mathcal{N}_{b,g}$ is the $g$-th of $G = 10$ contiguous groups of branch $b$. The final score fuses the two branches with a power mean ($p = 2$):

$$
S_{\mathrm{NegSep}}(x) = \left(\frac{S_{\mathrm{nbg}}(x)^p + S_{\mathrm{bg}}(x)^p}{2}\right)^{1/p}.
$$

Higher scores mean more ID-like. Separating the background labels keeps background-matched OOD images, such as scene-only images that CLIP associates with an ID class through a background shortcut, from being normalised away by the much larger non-background pool.

## Repository structure

```
NegSep/
├─ negtxt/
│  ├─ refine_neg_bg.txt    # 1,076 background negative labels
│  └─ refine_neg_nbg.txt   # 5,294 non-background negative labels
├─ scripts/                # one script per benchmark setting
└─ src/
   ├─ class_names.py       # ID class names and prompt templates
   ├─ clip_ood.py          # NegSep score
   ├─ eval.py              # evaluation entry point
   └─ ood_evaluate.py      # AUROC / FPR95
```

The two label files are the negative label set refined by NegRefine for ImageNet-1K (4,998 nouns + 1,372 adjectives), partitioned into background and non-background labels. The same vocabulary is used for every ID dataset; no labels are re-mined per dataset.

## Setup

```bash
pip install -r requirements.txt
```

We used Python 3.12, PyTorch 2.9.1 (CUDA 12.8) and OpenAI CLIP at commit `dcba3cb`, with the ViT-B/16 checkpoint.

## Data

Point `--data_root` (or the `NEGSEP_DATA` environment variable) to a folder with the following layout:

```
<data_root>/
├─ waterbird_complete95_forest2water2/   # WaterBirds (ImageFolder, 11,788 images)
├─ placesbg/{land,water}/                # PlacesBG from Spurious-OOD (5,000 + 5,000 images)
├─ ImageNet10/<wnid>/                    # ImageNet-10  (10 classes  x 50 ILSVRC-2012 val images)
├─ ImageNet20/<wnid>/                    # ImageNet-20  (20 classes  x 50)
├─ ImageNet99/<wnid>/                    # ImageNet-99  (99 classes  x 50)
├─ ImageNet/val/<wnid>/                  # ImageNet-1K validation set (50,000 images)
├─ iNaturalist/images/  SUN/images/  Places/images/   # flat image folders
└─ dtd/images/<category>/                # Textures
```

- **WaterBirds:** `waterbird_complete95_forest2water2` from [Sagawa et al.](https://github.com/kohpangwei/group_DRO).
- **PlacesBG:** the background-only OOD set released with [Spurious-OOD (Ming et al.)](https://github.com/deeplearning-wisc/Spurious_OOD), with its `land/` and `water/` folders.
- **ImageNet-10 / ImageNet-20 / ImageNet-100:** class lists from [MCM](https://github.com/deeplearning-wisc/MCM). Each class uses all 50 ILSVRC-2012 validation images.
- **ImageNet-99:** ImageNet-100 without `race car` (n04037443), whose images duplicate `sports car`, following NegRefine.
- **iNaturalist / SUN / Places / Textures:** the standard OOD subsets from [MOS](https://github.com/deeplearning-wisc/large_scale_ood).

Folder names can be changed in `FOLDER_DATASETS` / `FLAT_DATASETS` in `src/eval.py`.

## Usage

```bash
python src/eval.py --data_root /path/to/data \
    --in_dataset_name waterbird --ood_dataset_name_list placesbg \
    --output_folder output/waterbird
```

Or run one of the prepared settings:

```bash
export NEGSEP_DATA=/path/to/data
sh scripts/waterbird.sh   # WaterBirds -> PlacesBG
sh scripts/img10.sh       # ImageNet-10 -> ImageNet-20, ImageNet-99
sh scripts/img20.sh       # ImageNet-20 -> ImageNet-10
sh scripts/img99.sh       # ImageNet-99 -> ImageNet-10
sh scripts/imagenet.sh    # ImageNet-1K -> iNaturalist, SUN, Places, Textures
```

The script prints AUROC and FPR95 for the NegSep score and for each branch alone (`s_bg`, `s_nbg`). With `--output_folder`, it also writes `result.txt` and per-image scores (`scores_<dataset>.npz`, with image paths).

Options:

| Option | Default | Meaning |
|---|---|---|
| `--neg_bg_file`, `--neg_nbg_file` | `negtxt/refine_neg_{bg,nbg}.txt` | negative label files |
| `--ngroup` | 10 | number of contiguous groups per branch |
| `--p` | 2 | power-mean exponent |
| `--arch` | `ViT-B/16` | CLIP backbone |

## Results

CLIP ViT-B/16, obtained with this code (AUROC ↑ / FPR95 ↓, %).

**ImageNet-1K** (50,000 validation images as ID)

| OOD | AUROC | FPR95 |
|---|---:|---:|
| iNaturalist | 99.66 | 1.21 |
| SUN | 94.99 | 21.75 |
| Places | 90.60 | 38.93 |
| Textures | 93.92 | 23.97 |
| Average | 94.79 | 21.47 |

**Spurious and ImageNet-subset OOD**

| ID → OOD | AUROC | FPR95 |
|---|---:|---:|
| WaterBirds → PlacesBG | 93.15 | 11.19 |
| ImageNet-10 → ImageNet-20 | 99.14 | 4.10 |
| ImageNet-10 → ImageNet-99 | 99.49 | 1.72 |
| ImageNet-20 → ImageNet-10 | 98.83 | 3.40 |
| ImageNet-99 → ImageNet-10 | 91.02 | 44.80 |

Image decoding differs slightly between Pillow / torchvision versions, so values can move by a few hundredths in other environments.

## Acknowledgements

This code builds on [NegRefine](https://github.com/ah-ansari/NegRefine) (MIT License), from which `class_names.py`, `ood_evaluate.py` and the negative-label scoring routine are taken, and on [NegLabel](https://github.com/tmlr-group/NegLabel) and [CSP](https://github.com/MengyuanChen21/NeurIPS2024-CSP).
