# -*- coding: utf-8 -*-
"""Retrain a segmentation model on each input of ANALYSIS_PLAN.md (addendum on retraining), MSRS train -> MSRS test.
Usage: python seg_retrain.py <input> <seed>   input = a method, or avg, vi, ir
       python seg_retrain.py <input> <seed> --bench   speed on the training data only, nothing saved, test set untouched
Per-image pixel counts over the 9 MSRS classes: results/seg_retrain/<input>_s<seed>.npz. A run is restarted if interrupted."""
import glob, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'ir_vis_fusion')
TRAIN = os.path.join(ROOT, 'tmp_msrs', 'train')
TEST = os.path.join(ROOT, 'tmp_msrs', 'test')
STIM = os.path.join(HERE, 'stim', 'MSRS')
OUT = os.path.join(HERE, 'results', 'seg_retrain')
MODEL = os.path.join(HERE, 'models', 'segformer-b5-finetuned-cityscapes-1024-1024')
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
METHODS += ['Ours'] if 'Ours' in sys.argv[1:2] else []    # addendum 2026-11-16, native only
NCLS, CROP, BATCH, EPOCHS = 9, 384, 4, 30
LR, WD, WARM = 6e-5, .01, 500
which, seed = sys.argv[1], int(sys.argv[2])
BENCH = '--bench' in sys.argv
CONDS = '--conds' in sys.argv    # also keep the weights and test every transformed version (results/seg_retrain_cond)
if CONDS:
    OUT = os.path.join(HERE, 'results', 'seg_retrain_cond')
    assert which in METHODS, which
assert which in METHODS + ['avg', 'vi', 'ir'], which
f_out = os.path.join(OUT, '%s_s%d.npz' % (which, seed))
if os.path.exists(f_out) and not BENCH:
    print('exists', f_out); sys.exit()
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(seed); np.random.seed(seed)
dev = 'cuda'
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(.85)


def to_y8(path):
    """8-bit luminance, the same rule as make_conditions.py."""
    a = Image.open(path)
    if a.mode == 'L':
        return np.asarray(a, np.float32) / 255
    x = np.asarray(a.convert('RGB'), np.float32) / 255
    return np.round((.299 * x[..., 0] + .587 * x[..., 1] + .114 * x[..., 2]) * 255) / 255


def luminance(name):
    """Training luminance of this input, rounded to 8 bits as the test stimuli are."""
    if which in METHODS:
        return to_y8(os.path.join(ROOT, 'results', 'sota', which + '_MSRStrain', name))
    ir, vi = to_y8(os.path.join(TRAIN, 'ir', name)), to_y8(os.path.join(TRAIN, 'vi', name))
    y = {'ir': ir, 'vi': vi, 'avg': (ir + vi) / 2}[which]
    return np.round(y * 255) / 255


def rgb(y, vi_path):
    """Y plus the visible chroma, BT.601, as seg_eval.py."""
    v = np.asarray(Image.open(vi_path).convert('RGB'), np.float32) / 255
    yv = .299 * v[..., 0] + .587 * v[..., 1] + .114 * v[..., 2]
    return np.clip(v - yv[..., None] + y[..., None], 0, 1)


names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TRAIN, 'vi', '*.png')))
assert len(names) == 1083
t = time.time()
Xtr = np.empty((len(names), 480, 640, 3), np.uint8); Ytr = np.empty((len(names), 480, 640), np.uint8)
for i, n in enumerate(names):
    Xtr[i] = np.uint8(np.round(rgb(luminance(n), os.path.join(TRAIN, 'vi', n)) * 255))
    Ytr[i] = np.asarray(Image.open(os.path.join(TRAIN, 'Segmentation_labels', n)))
assert Ytr.max() < NCLS
print('loaded', Xtr.shape, 'in %.0fs' % (time.time() - t), flush=True)

from transformers import SegformerConfig, SegformerForSemanticSegmentation
cfg = SegformerConfig.from_pretrained(MODEL)
cfg.num_labels = NCLS; cfg.id2label = {i: str(i) for i in range(NCLS)}; cfg.label2id = {str(i): i for i in range(NCLS)}
net = SegformerForSemanticSegmentation(cfg)
state = torch.load(os.path.join(MODEL, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
state = {k: v for k, v in state.items() if not k.startswith('decode_head.classifier')}
missing, unexpected = net.load_state_dict(state, strict=False)
assert not unexpected and all('classifier' in k for k in missing), (missing, unexpected)
net = net.to(dev)
head = [p for k, p in net.named_parameters() if k.startswith('decode_head')]
body = [p for k, p in net.named_parameters() if not k.startswith('decode_head')]
opt = torch.optim.AdamW([dict(params=body, lr=LR), dict(params=head, lr=LR * 10)], weight_decay=WD)
scaler = torch.amp.GradScaler('cuda')
mean = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)
steps_per = len(names) // BATCH
TOTAL = steps_per * EPOCHS
BASE = [g['lr'] for g in opt.param_groups]
rng = np.random.default_rng(seed)    # same seed gives every input the same order, crops and flips


def batch(idx):
    x, y = [], []
    for i in idx:
        s = rng.uniform(.75, 1.5)
        H, W = int(480 * s), int(640 * s)
        a = torch.from_numpy(Xtr[i]).permute(2, 0, 1)[None].float().to(dev) / 255
        b = torch.from_numpy(Ytr[i])[None, None].float().to(dev)
        a = torch.nn.functional.interpolate(a, size=(H, W), mode='bilinear', align_corners=False)
        b = torch.nn.functional.interpolate(b, size=(H, W), mode='nearest')
        top, left = rng.integers(0, max(H - CROP, 0) + 1), rng.integers(0, max(W - CROP, 0) + 1)
        a, b = a[..., top:top + CROP, left:left + CROP], b[..., top:top + CROP, left:left + CROP]
        pad = (0, CROP - a.shape[-1], 0, CROP - a.shape[-2])
        a = torch.nn.functional.pad(a, pad); b = torch.nn.functional.pad(b, pad)
        if rng.random() < .5:
            a, b = a.flip(-1), b.flip(-1)
        x.append(a); y.append(b)
    return (torch.cat(x) - mean) / std, torch.cat(y)[:, 0].long()


net.train()
step = 0; t = time.time(); run = 0.
stop = 40 if BENCH else TOTAL
while step < stop:
    order = rng.permutation(len(names))
    for s in range(steps_per):
        if step >= stop:
            break
        f = min(1., (step + 1) / WARM) * (1 - step / TOTAL) ** 1.
        for g, b in zip(opt.param_groups, BASE):
            g['lr'] = b * f
        x, y = batch(order[s * BATCH:(s + 1) * BATCH])
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            lo = net(pixel_values=x).logits
            lo = torch.nn.functional.interpolate(lo, size=y.shape[-2:], mode='bilinear', align_corners=False)
            loss = torch.nn.functional.cross_entropy(lo, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        run += loss.item(); step += 1
        if step % 500 == 0 or (BENCH and step == stop):
            print('%s s%d step %d/%d loss %.4f  %.2f it/s  mem %.1f GB' % (which, seed, step, TOTAL, run / (500 if not BENCH else stop), step / (time.time() - t), torch.cuda.max_memory_allocated() / 1e9), flush=True)
            run = 0.
if BENCH:
    print('bench: %.2f it/s, one run of %d steps = %.0f min' % (step / (time.time() - t), TOTAL, TOTAL / (step / (time.time() - t)) / 60)); sys.exit()

# test only after training ended; no model selection on the test set
del Xtr, Ytr
net.eval()
tests = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TEST, 'vi', '*.png')))
assert len(tests) == 361
N = len(tests)


def evaluate(folder):
    inter = np.zeros((N, NCLS), np.int64); pred = inter.copy(); gt = inter.copy()
    with torch.inference_mode():
        for i, n in enumerate(tests):
            y = np.asarray(Image.open(os.path.join(folder, n)), np.float32) / 255
            a = torch.from_numpy(rgb(y, os.path.join(TEST, 'vi', n))).permute(2, 0, 1)[None].float().to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                lo = net(pixel_values=(a - mean) / std).logits
            m = torch.nn.functional.interpolate(lo.float(), size=a.shape[-2:], mode='bilinear', align_corners=False).argmax(1)[0].cpu().numpy()
            L = np.asarray(Image.open(os.path.join(TEST, 'Segmentation_labels', n)))
            for c in range(NCLS):
                P, G = m == c, L == c
                inter[i, c] = np.sum(P & G); pred[i, c] = P.sum(); gt[i, c] = G.sum()
    return inter, pred, gt


if CONDS:
    os.makedirs(os.path.join(OUT, 'weights'), exist_ok=True)
    torch.save(net.state_dict(), os.path.join(OUT, 'weights', '%s_s%d.pt' % (which, seed)))
    COND = ['native', 'blend25', 'blend50', 'blend75', 'noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
    res = {}
    for c in COND:
        inter, pred, gt = evaluate(os.path.join(STIM, which, c))
        res['inter_' + c] = inter; res['pred_' + c] = pred; res['gt_' + c] = gt
        iou = inter.sum(0) / np.maximum(pred.sum(0) + gt.sum(0) - inter.sum(0), 1)
        print('%s s%d %-8s mIoU9 %.4f mIoU3 %.4f' % (which, seed, c, iou.mean(), iou[[1, 2, 3]].mean()), flush=True)
    np.savez_compressed(f_out, names=np.array(tests), conds=np.array(COND), config=np.array([EPOCHS, BATCH, CROP, LR, WD, TOTAL], float), **res)
    print('DONE %s s%d %d s' % (which, seed, time.time() - t), flush=True)
    sys.exit()
inter, pred, gt = evaluate(os.path.join(STIM, which, 'native') if which in METHODS else os.path.join(STIM, 'base', which))
np.savez_compressed(f_out, names=np.array(tests), inter=inter, pred=pred, gt=gt, config=np.array([EPOCHS, BATCH, CROP, LR, WD, TOTAL], float))
iou = inter.sum(0) / np.maximum(pred.sum(0) + gt.sum(0) - inter.sum(0), 1)
print('DONE %s s%d %d s, mIoU9 %.4f mIoU3 %.4f' % (which, seed, time.time() - t, iou.mean(), iou[[1, 2, 3]].mean()), flush=True)
