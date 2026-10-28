import clip
import numpy as np
import torch

from class_names import CLASS_NAME, preset_noun_prompt_templates, preset_noun_prompt_templates_for_sketch

_EPS = 1e-8


def _load_txt(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def power_mean(s_nbg, s_bg, p=2):
    """Fuses the two branch scores: ((S_nbg^p + S_bg^p) / 2)^(1/p), computed in float64."""
    a = np.maximum(np.asarray(s_nbg, dtype=np.float64), _EPS)
    b = np.maximum(np.asarray(s_bg, dtype=np.float64), _EPS)
    if p == 1:
        return (a + b) / 2.0
    return ((a ** p + b ** p) / 2.0) ** (1.0 / p)


class NegSep():
    """Zero-shot OOD detection with background / non-background negative labels.

    Each branch is a NegLabel-style grouped score computed against its own negative pool:
        S_b(x) = mean_g [ sum_ID exp(s) / (sum_ID exp(s) + sum_{n in group g of branch b} exp(s)) ]
    with cosine similarities scaled by 100 (tau = 0.01) and G contiguous groups per branch
    (the last n mod G labels of each file are dropped). The final score is the power mean of S_nbg and S_bg.
    Higher score = more ID-like.
    """

    def __init__(self, train_dataset='imagenet', arch='ViT-B/16', device='cuda:0',
                 neg_bg_file='negtxt/refine_neg_bg.txt', neg_nbg_file='negtxt/refine_neg_nbg.txt',
                 ngroup=10, p=2):
        self.device = device
        self.ngroup = ngroup
        self.p = p

        self.clip_model, self.clip_preprocess = clip.load(arch, self.device, jit=False)
        self.clip_model.eval()

        class_name = CLASS_NAME[train_dataset]
        if train_dataset == 'imagenet_sketch':
            self.noun_prompt_templates = preset_noun_prompt_templates_for_sketch
        else:
            self.noun_prompt_templates = preset_noun_prompt_templates

        neg_bg_labels = _load_txt(neg_bg_file)
        neg_nbg_labels = _load_txt(neg_nbg_file)
        print(f"ID classes: {len(class_name)}, neg_bg: {len(neg_bg_labels)}, neg_nbg: {len(neg_nbg_labels)}")

        self.pos_features = self._embed_text_labels(class_name, self.noun_prompt_templates)
        self.neg_bg_features = self._embed_text_labels(neg_bg_labels, self.noun_prompt_templates)
        self.neg_nbg_features = self._embed_text_labels(neg_nbg_labels, self.noun_prompt_templates)

    def _embed_text_labels(self, labels, prompt_templates, batch_size=1000):
        texts = [t.format(w) for w in labels for t in prompt_templates]
        tokenized = torch.cat([clip.tokenize(c) for c in texts]).to(self.device)

        with torch.no_grad():
            features = []
            for i in range(0, len(tokenized), batch_size):
                features.append(self.clip_model.encode_text(tokenized[i:i + batch_size]))
            features = torch.cat(features, dim=0)
            features = features.view(-1, len(prompt_templates), features.shape[-1]).mean(dim=1)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.to(torch.float32)

        return features

    def _grouping(self, pos, neg):
        group = self.ngroup
        drop = neg.shape[1] % group
        if drop > 0:
            neg = neg[:, :-drop]
        negs = neg.reshape(pos.shape[0], group, -1).contiguous()
        scores = []
        for i in range(group):
            full_sim = torch.cat([pos, negs[:, i, :]], dim=-1).softmax(dim=-1)
            scores.append(full_sim[:, :pos.shape[1]].sum(dim=-1).unsqueeze(-1))
        return torch.cat(scores, dim=-1).mean(dim=-1)

    def encode_images(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images.to(self.device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.to(torch.float32)

    def branch_scores(self, image_features):
        """Returns (S_nbg, S_bg) as float64 numpy arrays for a batch of L2-normalised image features."""
        with torch.no_grad():
            pos_sim = 100.0 * image_features @ self.pos_features.T
            s_bg = self._grouping(pos_sim, 100.0 * image_features @ self.neg_bg_features.T)
            s_nbg = self._grouping(pos_sim, 100.0 * image_features @ self.neg_nbg_features.T)
        return s_nbg.double().cpu().numpy(), s_bg.double().cpu().numpy()

    def detection_score(self, images):
        """Returns dict with the NegSep score and the two branch scores for a batch of preprocessed images."""
        s_nbg, s_bg = self.branch_scores(self.encode_images(images))
        return dict(negsep=power_mean(s_nbg, s_bg, self.p), s_nbg=s_nbg, s_bg=s_bg)
