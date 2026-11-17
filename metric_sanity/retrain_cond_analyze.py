# -*- coding: utf-8 -*-
"""Transformed images on retrained models, as fixed in ANALYSIS_PLAN.md. Reads results/seg_retrain_cond/*.npz and rank_attack.csv.
Writes results/retrain_cond_pooled.csv, retrain_cond_method.csv, retrain_cond_class.csv, retrain_cond_rank.csv."""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
BLENDS = ['blend25', 'blend50', 'blend75']
ATTACK = ['noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
TRANS = BLENDS + ATTACK
SCORES = (('mIoU3', [1, 2, 3]), ('mIoU9', list(range(9))))
CLASSES = (('car', 1), ('person', 2), ('bike', 3))
B = 10000
rng = np.random.default_rng(20261102)
D = {}
for m in METHODS:
    d = np.load(os.path.join(R, 'seg_retrain_cond', m + '_s0.npz'))
    for c in ['native'] + TRANS:
        D[m, c] = (d['inter_' + c].astype(float), d['pred_' + c].astype(float), d['gt_' + c].astype(float))
    names = d['names']
N = len(names)
W = rng.multinomial(N, np.ones(N) / N, size=B).astype(float)


def iou(x):
    I, P, G = x[0].sum(0), x[1].sum(0), x[2].sum(0)
    return I / np.maximum(P + G - I, 1)


def biou(x):
    I, P, G = W @ x[0], W @ x[1], W @ x[2]
    return I / np.maximum(P + G - I, 1)


def holm(ps):
    order = np.argsort(ps); run = 0; out = [0.] * len(ps)
    for rank, i in enumerate(order):
        run = max(run, min(1., ps[i] * (len(ps) - rank))); out[i] = run
    return out


def save(name, rows):
    with open(os.path.join(R, name), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (('%.5f' % v) if isinstance(v, float) else v) for k, v in r.items()})


print('== Repeatability: native score against the earlier seed-0 run')
for m in METHODS:
    old = np.load(os.path.join(R, 'seg_retrain', m + '_s0.npz'))
    a, b = iou(D[m, 'native']), iou((old['inter'].astype(float), old['pred'].astype(float), old['gt'].astype(float)))
    assert np.array_equal(D[m, 'native'][2], old['gt'])
    print(' %-9s mIoU3 now %.4f earlier %.4f | mIoU9 now %.4f earlier %.4f' % (m, a[[1, 2, 3]].mean(), b[[1, 2, 3]].mean(), a.mean(), b.mean()))

pt = {k: iou(v) for k, v in D.items()}
bt = {k: biou(v) for k, v in D.items()}
per, pooled, cls = [], [], []
for name, cl in SCORES:
    fam = []
    for T in TRANS:
        dd = []
        for m in METHODS:
            d = (bt[m, T] - bt[m, 'native'])[:, cl].mean(1); dd.append(d)
            per.append(dict(score=name, method=m, transform=T, native=pt[m, 'native'][cl].mean(), delta=(pt[m, T] - pt[m, 'native'])[cl].mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
        d = np.mean(dd, 0)
        fam.append(dict(score=name, transform=T, delta=np.mean([(pt[m, T] - pt[m, 'native'])[cl].mean() for m in METHODS]), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                        p=min(1., 2 * min(np.mean(d <= 0), np.mean(d >= 0)) + 1 / B)))
    for r, h in zip(fam, holm([r['p'] for r in fam])):
        r['p_holm'] = h
    pooled += fam
for T in TRANS:
    for cname, c in CLASSES:
        d = np.mean([(bt[m, T] - bt[m, 'native'])[:, c] for m in METHODS], 0)
        cls.append(dict(transform=T, cls=cname, delta=np.mean([(pt[m, T] - pt[m, 'native'])[c] for m in METHODS]), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
save('retrain_cond_pooled.csv', pooled); save('retrain_cond_method.csv', per); save('retrain_cond_class.csv', cls)
print('\n== Six methods pooled, transformed minus native')
for r in pooled:
    print(' %-5s %-9s %+.4f [%+.4f,%+.4f] holm %.4f %s' % (r['score'], r['transform'], r['delta'], r['lo'], r['hi'], r['p_holm'], '*' if r['p_holm'] < .05 else ''))
print('\n== Per method, mIoU3')
for r in per:
    if r['score'] == 'mIoU3':
        print(' %-9s %-9s %+.4f [%+.4f,%+.4f]' % (r['method'], r['transform'], r['delta'], r['lo'], r['hi']))
print('\n== Class, pooled')
for r in cls:
    print(' %-9s %-7s %+.4f [%+.4f,%+.4f]' % (r['transform'], r['cls'], r['delta'], r['lo'], r['hi']))

look = {(r['method'], r['transform']): r for r in per if r['score'] == 'mIoU3'}
rows = []
for r in csv.DictReader(open(os.path.join(R, 'rank_attack.csv'))):
    if r['dataset'] != 'MSRS' or r['kind'] != 'attack':
        continue
    x = look[r['method'], r['transform']]
    rows.append(dict(metric=r['metric'], method=r['method'], transform=r['transform'], rank_native=int(r['rank_native']), rank_after=int(r['rank_after']), counted=r['counted'] == 'True',
                     harmful_pretrained=r['harmful'] == 'True', d_retrained=x['delta'], lo_retrained=x['lo'], hi_retrained=x['hi'],
                     harmful_retrained=r['counted'] == 'True' and x['hi'] <= 0, task_rises_retrained=r['counted'] == 'True' and x['lo'] > 0))
save('retrain_cond_rank.csv', rows)
print('\n== Rank attacks on MSRS: counted, harmful on pretrained, harmful on retrained, task rises on retrained')
for mt in ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'MS_SSIM', 'CC', 'Qabf', 'FMI', 'Nabf', 'SCD', 'VIF']:
    s = [r for r in rows if r['metric'] == mt]
    print(' %-8s %2d %2d %2d %2d' % (mt, sum(r['counted'] for r in s), sum(r['harmful_pretrained'] for r in s), sum(r['harmful_retrained'] for r in s), sum(r['task_rises_retrained'] for r in s)))
print('\n== Attacks reaching first place from last place')
for r in rows:
    if r['counted'] and r['rank_after'] == 1 and r['rank_native'] == 6:
        print(' %-8s %-9s %-8s retrained %+.4f [%+.4f,%+.4f]%s' % (r['metric'], r['method'], r['transform'], r['d_retrained'], r['lo_retrained'], r['hi_retrained'], ' harmful' if r['harmful_retrained'] else ''))
