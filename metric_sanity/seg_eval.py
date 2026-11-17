# -*- coding: utf-8 -*-
"""Segmentation of every MSRS condition by a pretrained Cityscapes model. Per-image intersection, prediction and
label pixel counts are saved per condition, so the analysis can bootstrap over images. Resumable.
results/seg/<model>/<method>_<condition>.npz with names, inter[N,3], pred[N,3], gt[N,3]; classes person, car, bike.
Usage: python seg_eval.py segformer|mask2former"""
import glob, json, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'ir_vis_fusion')
STIM = os.path.join(HERE, 'stim', 'MSRS')
METHODS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse']
CONDS = ['native', 'blend25', 'blend50', 'blend75', 'noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
if '--native' in sys.argv:    # the proposed network of the companion paper, native only (addendum 2026-11-16)
    CONDS = ['native']
MSRS_ID = {'person': 2, 'car': 1, 'bike': 3}
CITY = {'person': ['person', 'rider'], 'car': ['car', 'truck', 'bus'], 'bike': ['bicycle', 'motorcycle']}
SCALE = 2           # 640x480 -> 1280x960, closer to the object scale the models were trained on
BATCH = 1
which = sys.argv[1]
torch.backends.cudnn.benchmark = True
dev = 'cuda'
torch.cuda.set_per_process_memory_fraction(0.65)   # fail instead of spilling into host memory
if which == 'segformer':
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    d = os.path.join(HERE, 'models', 'segformer-b5-finetuned-cityscapes-1024-1024')
    cfg = SegformerConfig.from_pretrained(d)
    net = SegformerForSemanticSegmentation(cfg)
    print(net.load_state_dict(torch.load(os.path.join(d, 'pytorch_model.bin'), map_location='cpu', weights_only=True)))
else:
    from transformers import Mask2FormerForUniversalSegmentation
    d = os.path.join(HERE, 'models', 'mask2former-swin-small-cityscapes-semantic')
    net = Mask2FormerForUniversalSegmentation.from_pretrained(d, use_safetensors=True)
net = net.eval().to(dev)
pc = json.load(open(os.path.join(d, 'preprocessor_config.json')))
mean = torch.tensor(pc['image_mean'], device=dev).view(1, 3, 1, 1); std = torch.tensor(pc['image_std'], device=dev).view(1, 3, 1, 1)
print('rescale', pc.get('do_rescale', True), 'mean', pc['image_mean'], 'std', pc['image_std'])
id2 = {int(k): v.lower() for k, v in net.config.id2label.items()}
print(id2)
GROUP = [[k for k, v in id2.items() if v in CITY[c]] for c in ['person', 'car', 'bike']]
assert [len(g) for g in GROUP] == [2, 3, 2], GROUP
OUT = os.path.join(HERE, 'results', 'seg', which); os.makedirs(OUT, exist_ok=True)

names = json.load(open(os.path.join(HERE, 'results', 'image_sets.json')))['msrs_all']
labels = {n: np.asarray(Image.open(os.path.join(ROOT, 'tmp_msrs', 'test', 'Segmentation_labels', n))) for n in names}


def chroma(n):
    """RGB = Y + (channel - Y) in BT.601. Read per batch to keep host memory low."""
    vi = np.asarray(Image.open(os.path.join(ROOT, 'tmp_msrs', 'test', 'vi', n)).convert('RGB'), np.float32) / 255
    y = .299 * vi[..., 0] + .587 * vi[..., 1] + .114 * vi[..., 2]
    return np.stack([vi[..., 0] - y, vi[..., 1] - y, vi[..., 2] - y], 0)


@torch.inference_mode()
def infer(paths):
    """Y plus the visible chroma, returns class maps at label size."""
    x = []
    for p, n in paths:
        y = np.asarray(Image.open(p), np.float32) / 255
        x.append(np.clip(chroma(n) + y[None], 0, 1))
    x = torch.from_numpy(np.stack(x)).to(dev)
    h, w = x.shape[2:]
    x = torch.nn.functional.interpolate(x, scale_factor=SCALE, mode='bilinear', align_corners=False)
    x = (x - mean) / std
    with torch.autocast('cuda', dtype=torch.float16):
        out = net(pixel_values=x)
    if which == 'segformer':
        lo = torch.nn.functional.interpolate(out.logits.float(), size=(h, w), mode='bilinear', align_corners=False)
        return lo.argmax(1).cpu().numpy()
    cls = out.class_queries_logits.float().softmax(-1)[..., :-1]
    mas = torch.nn.functional.interpolate(out.masks_queries_logits.float(), size=(h, w), mode='bilinear', align_corners=False).sigmoid()
    return torch.einsum('bqc,bqhw->bchw', cls, mas).argmax(1).cpu().numpy()


def run(key, folder):
    f = os.path.join(OUT, key + '.npz')
    if os.path.exists(f):
        return
    t = time.time()
    N = len(names); inter = np.zeros((N, 3), np.int64); pred = inter.copy(); gt = inter.copy()
    for s in range(0, N, BATCH):
        b = names[s:s + BATCH]
        maps = infer([(os.path.join(folder, n), n) for n in b])
        for j, n in enumerate(b):
            for c, (cn, grp) in enumerate(zip(['person', 'car', 'bike'], GROUP)):
                P = np.isin(maps[j], grp); G = labels[n] == MSRS_ID[cn]
                inter[s + j, c] = np.sum(P & G); pred[s + j, c] = P.sum(); gt[s + j, c] = G.sum()
    np.savez_compressed(f, names=np.array(names), inter=inter, pred=pred, gt=gt)
    iou = inter.sum(0) / np.maximum(pred.sum(0) + gt.sum(0) - inter.sum(0), 1)
    print('%-22s person %.3f car %.3f bike %.3f mIoU3 %.3f  %.0fs' % (key, *iou, iou.mean(), time.time() - t), flush=True)


for c in ['ir', 'vi', 'avg']:
    run('base_' + c, os.path.join(STIM, 'base', c))
for m in METHODS:
    for c in CONDS:
        run(m + '_' + c, os.path.join(STIM, m, c))
print('DONE', which, flush=True)
