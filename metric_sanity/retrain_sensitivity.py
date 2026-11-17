# -*- coding: utf-8 -*-
"""Checks on results/seg_retrain: labels agree, scopes, class support, per-seed differences, seed-aware interval."""
import os, json, hashlib
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
INPUTS = METHODS + ['avg', 'vi', 'ir']
SEEDS = [0, 1, 2]
CL = ['unlab', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'cone', 'bump']
D = {}
for i in INPUTS:
    for s in SEEDS:
        d = np.load(os.path.join(R, 'seg_retrain', '%s_s%d.npz' % (i, s)))
        D[i, s] = (d['inter'].astype(float), d['pred'].astype(float), d['gt'].astype(float)); names = d['names']
        assert list(d['config'][:3]) == [30, 4, 384], d['config']
g0 = D['avg', 0][2]
print('labels identical over 27 runs:', all(np.array_equal(D[k][2], g0) for k in D))
print('prediction arrays distinct over 27 runs:', len({hashlib.md5(D[k][1].tobytes()).hexdigest() for k in D}))
N = len(names); night = np.array([n[-5] == 'N' for n in names])
print('N', N, 'day', (~night).sum(), 'night', night.sum())
SC = {'all': np.ones(N, bool), 'day': ~night, 'night': night}


def iou(x, m):
    I, P, G = x[0][m].sum(0), x[1][m].sum(0), x[2][m].sum(0)
    return I / np.maximum(P + G - I, 1)


print('\nclass support: images containing the class / labelled pixels')
for c, n in enumerate(CL):
    print('%-10s' % n, ' '.join('%s %3d %9d' % (s, (g0[m, c] > 0).sum(), g0[m, c].sum()) for s, m in SC.items()))

print('\nper-class IoU, seed mean (sd)')
for s, m in SC.items():
    print(s)
    print('%-10s' % '', ' '.join('%-11s' % c for c in CL))
    for i in INPUTS:
        v = np.stack([iou(D[i, k], m) for k in SEEDS])
        print('%-10s' % i, ' '.join('%.3f(%.3f)' % (a, b) for a, b in zip(v.mean(0), v.std(0, ddof=1))))

print('\nsame-seed differences, method minus avg')
for s, m in SC.items():
    for name, cl in (('mIoU9', list(range(9))), ('mIoU3', [1, 2, 3])):
        for i in METHODS:
            d = [iou(D[i, k], m)[cl].mean() - iou(D['avg', k], m)[cl].mean() for k in SEEDS]
            print('%-5s %-5s %-9s %+.4f %+.4f %+.4f' % (s, name, i, *d))

# sensitivity: images and seeds both resampled. Seeds are few (3), so this is a rough widening, not an exact interval.
rng = np.random.default_rng(20261102)
B = 10000
print('\nsensitivity: images and seeds resampled together, method and reference share the seed draw')
out = []
for s, m in SC.items():
    idx = np.where(m)[0]; n = len(idx)
    W = rng.multinomial(n, np.ones(n) / n, size=B).astype(float)
    S = rng.integers(0, 3, size=(B, 3))
    bi = {}
    for i in INPUTS:
        arr = []
        for k in SEEDS:
            x = D[i, k]
            I, P, G = W @ x[0][idx], W @ x[1][idx], W @ x[2][idx]
            arr.append(I / np.maximum(P + G - I, 1))
        bi[i] = np.stack(arr)  # seeds x B x classes
    for name, cl in (('mIoU9', list(range(9))), ('mIoU3', [1, 2, 3])):
        sc = {i: bi[i][:, :, cl].mean(2) for i in INPUTS}  # seeds x B
        for ref in ['avg', 'vi', 'ir']:
            for i in (METHODS + ['avg'] if ref != 'avg' else METHODS):
                diff = sc[i] - sc[ref]
                d = diff[S, np.arange(B)[:, None]].mean(1)
                pt = np.mean([iou(D[i, k], m)[cl].mean() - iou(D[ref, k], m)[cl].mean() for k in SEEDS])
                lo, hi = np.percentile(d, [2.5, 97.5])
                out.append(dict(scope=s, score=name, input=i, ref=ref, delta=pt, lo=lo, hi=hi))
                print('%-5s %-5s %-9s - %-3s %+.4f [%+.4f,%+.4f]' % (s, name, i, ref, pt, lo, hi))
json.dump(out, open(os.path.join(R, 'retrain_sensitivity.json'), 'w'), indent=0)
