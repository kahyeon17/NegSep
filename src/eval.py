import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import clip_ood
import ood_evaluate

# Dataset folders relative to --data_root (see README). ImageFolder layout unless listed in FLAT_DATASETS.
FOLDER_DATASETS = {
    'imagenet':        'ImageNet/val',
    'imagenet10':      'ImageNet10',
    'imagenet20':      'ImageNet20',
    'imagenet99':      'ImageNet99',
    'waterbird':       'waterbird_complete95_forest2water2',
    'placesbg':        'placesbg',
    'dtd':             'dtd/images',
}

FLAT_DATASETS = {
    'inaturalist': 'iNaturalist/images',
    'sun':         'SUN/images',
    'places':      'Places/images',
}


class FlatImageDataset(Dataset):
    EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    def __init__(self, root, transform):
        self.paths = sorted(p for p in Path(root).rglob('*') if p.suffix in self.EXTS)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB')), 0


def load_dataset(data_root, dataset_name, transform):
    if dataset_name in FOLDER_DATASETS:
        ds = tv.datasets.ImageFolder(os.path.join(data_root, FOLDER_DATASETS[dataset_name]), transform=transform)
        paths = [p for p, _ in ds.samples]
    elif dataset_name in FLAT_DATASETS:
        ds = FlatImageDataset(os.path.join(data_root, FLAT_DATASETS[dataset_name]), transform)
        paths = [str(p) for p in ds.paths]
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Add it to FOLDER_DATASETS or FLAT_DATASETS in src/eval.py.")
    return ds, paths


def get_scores(dataset, model, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = {'negsep': [], 's_nbg': [], 's_bg': []}
    for images, _ in tqdm(loader):
        for k, v in model.detection_score(images).items():
            out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def _format_result(scores_in, scores_ood):
    rows = []
    for name in ('negsep', 's_bg', 's_nbg'):
        auroc, _, _, fpr = ood_evaluate.get_measures(scores_in[name], scores_ood[name])
        rows.append(f"{name:<8}  AUROC: {100 * auroc:.2f}    FPR95: {100 * fpr:.2f}")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.environ.get("NEGSEP_DATA", "data"))
    parser.add_argument("--in_dataset_name", type=str, default="imagenet")
    parser.add_argument("--ood_dataset_name_list", type=str, nargs='+', default=["inaturalist", "sun", "places", "dtd"])
    parser.add_argument("--arch", type=str, default="ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--neg_bg_file", type=str, default="negtxt/refine_neg_bg.txt")
    parser.add_argument("--neg_nbg_file", type=str, default="negtxt/refine_neg_nbg.txt")
    parser.add_argument("--ngroup", type=int, default=10)
    parser.add_argument("--p", type=float, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_folder", type=str, default=None,
                        help="If set, writes result.txt and per-image scores (scores_<dataset>.npz) here.")
    args = parser.parse_args()
    print(args)

    model = clip_ood.NegSep(train_dataset=args.in_dataset_name, arch=args.arch, device=args.device,
                            neg_bg_file=args.neg_bg_file, neg_nbg_file=args.neg_nbg_file,
                            ngroup=args.ngroup, p=args.p)

    t0 = time.time()
    in_ds, in_paths = load_dataset(args.data_root, args.in_dataset_name, model.clip_preprocess)
    print(f"ID ({args.in_dataset_name}): {len(in_ds)}")
    scores_in = get_scores(in_ds, model, args.batch_size, args.num_workers)
    results = {args.in_dataset_name: (scores_in, in_paths)}
    for name in args.ood_dataset_name_list:
        ds, paths = load_dataset(args.data_root, name, model.clip_preprocess)
        print(f"OOD ({name}): {len(ds)}")
        results[name] = (get_scores(ds, model, args.batch_size, args.num_workers), paths)
    print(f"time: {time.time() - t0:.1f}s")

    lines = [f"in: {args.in_dataset_name}", ""]
    for name in args.ood_dataset_name_list:
        lines += [f"------------------ {name} -------------------", _format_result(scores_in, results[name][0]), ""]
    print("\n".join(lines))

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
        with open(os.path.join(args.output_folder, 'result.txt'), 'w') as f:
            f.write("\n".join(lines) + "\n")
        for name, (scores, paths) in results.items():
            np.savez(os.path.join(args.output_folder, f'scores_{name}.npz'), paths=np.array(paths), **scores)
        print(f"saved to {args.output_folder}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
