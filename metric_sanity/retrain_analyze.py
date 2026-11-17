# -*- coding: utf-8 -*-
"""Segmentation retrained on fused images, as fixed in ANALYSIS_PLAN.md. Reads results/seg_retrain/*.npz.
Writes results/retrain_scores.csv and results/retrain_vs_avg.csv."""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
INPUTS = METHODS + ['avg', 'vi', 'ir']
SEEDS = [0, 1, 2]
SCORES = (('mIoU9', list(range(9))), ('mIoU3', [1, 2, 3]))
B = 10000
rng = np.random.default_rng(20261031)
D = {}
for i in INPUTS:
    for s in SEEDS:
        d = np.load(os.path.join(R, 'seg_retrain', '%s_s%d.npz' % (i, s)))
        D[i, s] = (d['inter'].astype(float), d['pred'].astype(float), d['gt'].astype(float)); names = d['names']
N = len(names)
night = np.array([n[-5] == 'N' for n in names])
SCOPES = (('all', np.ones(N, bool)), ('day', ~night), ('night', night))
W = rng.multinomial(N, np.ones(N) / N, size=B).astype(float)


def iou(x, m):
    I, P, G = x[0][m].sum(0), x[1][m].sum(0), x[2][m].sum(0)
    return I / np.maximum(P + G - I, 1)


def biou(x, Wm):
    I, P, G = np.einsum('bn,nc->bc', Wm, x[0]), np.einsum('bn,nc->bc', Wm, x[1]), np.einsum('bn,nc->bc', Wm, x[2])
    return I / np.maximum(P + G - I, 1)


def holm(ps):
    order = np.argsort(ps); run = 0; out = [0.] * len(ps)
    for rank, i in enumerate(order):
        run = max(run, min(1., ps[i] * (len(ps) - rank))); out[i] = run
    return out


scores, cmp_rows = [], []
for sc, mask in SCOPES:
    Wm = W * mask[None, :]
    per = {i: np.stack([iou(D[i, s], mask) for s in SEEDS]) for i in INPUTS}            # seeds x classes
    boot = {i: np.mean([biou(D[i, s], Wm) for s in SEEDS], 0) for i in INPUTS}          # B x classes, seed average
    for name, cl in SCORES:
        for i in INPUTS:
            v = per[i][:, cl].mean(1)
            scores.append(dict(scope=sc, score=name, input=i, mean=v.mean(), seed_sd=v.std(ddof=1), s0=v[0], s1=v[1], s2=v[2], **{'c%d' % c: per[i][:, c].mean() for c in range(9)}))
        fam = []
        for ref in ['avg', 'vi']:
            sub = []
            for m in METHODS:
                d = boot[m][:, cl].mean(1) - boot[ref][:, cl].mean(1)
                sub.append(dict(scope=sc, score=name, reference=ref, method=m, delta=(per[m][:, cl].mean(1) - per[ref][:, cl].mean(1)).mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                                p=min(1., 2 * min(np.mean(d <= 0), np.mean(d >= 0)) + 1 / B)))
            for r, h in zip(sub, holm([r['p'] for r in sub])):
                r['p_holm'] = h
            fam += sub
        cmp_rows += fam
for fn, rows in (('retrain_scores.csv', scores), ('retrain_vs_avg.csv', cmp_rows)):
    with open(os.path.join(R, fn), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (('%.5f' % v) if isinstance(v, float) else v) for k, v in r.items()})
for r in scores:
    print('%-5s %-5s %-9s %.4f sd %.4f' % (r['scope'], r['score'], r['input'], r['mean'], r['seed_sd']))
for r in cmp_rows:
    print('%-5s %-5s %-9s - %-3s %+.4f [%+.4f,%+.4f] holm %.4f %s' % (r['scope'], r['score'], r['method'], r['reference'], r['delta'], r['lo'], r['hi'], r['p_holm'], '*' if r['p_holm'] < .05 else ''))
