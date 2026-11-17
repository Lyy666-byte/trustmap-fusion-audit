# -*- coding: utf-8 -*-
"""Figure of Section 5.3: one MSRS image fused by MMDRFuse, native and transformed, with its per-image metric values
and the prediction of MMDRFuse's own retrained segmentation model (seed 0, weights kept by seg_retrain.py --conds).
Image rule, fixed before drawing: of the 100 MSRS metric images, the night image with the most labelled person pixels.
Run with the .venv interpreter (torch, transformers). Writes ../ir_vis_fusion/results/audit_example.png."""
import csv, json, os
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import SegformerConfig, SegformerForSemanticSegmentation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'ir_vis_fusion')
TEST = os.path.join(ROOT, 'tmp_msrs', 'test')
STIM = os.path.join(HERE, 'stim', 'MSRS')
M = 'MMDRFuse'
CONDS = [('native', 'Native'), ('noise1', 'Noise, σ = 0.01'), ('sharp2', 'Unsharp, k = 2'), ('blend50', 'Blend, a = 0.5')]

S = json.load(open(os.path.join(HERE, 'results', 'image_sets.json')))
night = [n for n in S['metric_msrs'] if n[-5] == 'N']
lab = {n: np.asarray(Image.open(os.path.join(TEST, 'Segmentation_labels', n))) for n in night}
name = max(night, key=lambda n: int((lab[n] == 2).sum()))
L = lab[name]
print('image', name, 'person pixels', int((L == 2).sum()))

vals = {r['condition']: r for r in csv.DictReader(open(os.path.join(HERE, 'results', 'cond_metrics.csv')))
        if r['dataset'] == 'MSRS' and r['method'] == M and r['image'] == name}
assert set(c for c, _ in CONDS) <= set(vals), sorted(vals)

vi = np.asarray(Image.open(os.path.join(TEST, 'vi', name)).convert('RGB'), np.float32) / 255
yv = .299 * vi[..., 0] + .587 * vi[..., 1] + .114 * vi[..., 2]
rgb = lambda y: np.clip(vi - yv[..., None] + y[..., None], 0, 1)
ir = np.asarray(Image.open(os.path.join(TEST, 'ir', name)).convert('L'), np.float32) / 255

d = os.path.join(HERE, 'models', 'segformer-b5-finetuned-cityscapes-1024-1024')
cfg = SegformerConfig.from_pretrained(d)
cfg.num_labels = 9; cfg.id2label = {i: str(i) for i in range(9)}; cfg.label2id = {str(i): i for i in range(9)}
net = SegformerForSemanticSegmentation(cfg)
net.load_state_dict(torch.load(os.path.join(HERE, 'results', 'seg_retrain_cond', 'weights', M + '_s0.pt'), map_location='cpu', weights_only=True))
net.eval()
mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def seg(x):
    t = torch.from_numpy(x).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        lo = net(pixel_values=(t - mean) / std).logits
    return torch.nn.functional.interpolate(lo, size=x.shape[:2], mode='bilinear', align_corners=False).argmax(1)[0].numpy()


COL = {1: (0.12, 0.47, 0.71), 2: (0.84, 0.15, 0.16), 3: (1.0, 0.75, 0.0)}    # MSRS: 1 car, 2 person, 3 bike


def overlay(base, m):
    o = np.repeat(base[..., None], 3, 2) * 0.55
    for c, col in COL.items():
        o[m == c] = 0.35 * o[m == c] + 0.65 * np.array(col)
    return o


def iou(m, c):
    return np.sum((m == c) & (L == c)) / max(np.sum((m == c) | (L == c)), 1)


fig, ax = plt.subplots(2, 5, figsize=(15, 6.3))
ax[0, 0].imshow(ir, cmap='gray'); ax[0, 0].set_title('Infrared', fontsize=13)
ax[1, 0].imshow(overlay(yv, L)); ax[1, 0].set_title('Label', fontsize=13)
for j, (c, lab_c) in enumerate(CONDS, 1):
    y = np.asarray(Image.open(os.path.join(STIM, M, c, name)), np.float32) / 255
    x = rgb(y); m = seg(x); v = vals[c]
    ax[0, j].imshow(x)
    ax[0, j].set_title(lab_c, fontsize=13)
    ax[0, j].text(0.02, 0.98, 'AG %.4f  SF %.4f\nEN %.3f  Qabf %.3f\nMI %.3f  SSIM %.3f' % (float(v['AG']), float(v['SF']), float(v['EN']), float(v['Qabf']), float(v['MI']), float(v['SSIM'])),
                  transform=ax[0, j].transAxes, va='top', ha='left', fontsize=10, family='monospace', color='w',
                  bbox=dict(facecolor='black', alpha=0.55, pad=2, lw=0))
    ax[1, j].imshow(overlay(yv, m))
    ax[1, j].set_title('Person IoU %.3f' % iou(m, 2), fontsize=13)
    print(c, {k: v[k] for k in ('AG', 'SF', 'EN', 'Qabf', 'MI', 'SSIM')}, 'person IoU %.4f car IoU %.4f' % (iou(m, 2), iou(m, 1)),
          'RMS change %.2f grey levels' % (np.sqrt(np.mean((y - np.asarray(Image.open(os.path.join(STIM, M, 'native', name)), np.float32) / 255) ** 2)) * 255))
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.8)
out = os.path.join(ROOT, 'results', 'audit_example_v2.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out, 'image', name)
