# -*- coding: utf-8 -*-
"""Addendum 2026-11-16 of ANALYSIS_PLAN.md: the proposed network of the companion paper in the task test.
Statistics as retrain_analyze.py, retrain_sensitivity.py and fmb_analyze.py; not added to the Holm families of the six methods.
Writes results/ours_retrain.csv, ours_pretrained.csv, ours_metrics_rank.csv."""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
SEEDS = [0, 1, 2]
B = 10000
rng = np.random.default_rng(20261116)
METS = ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'MS_SSIM', 'CC', 'Qabf', 'FMI', 'Nabf', 'SCD', 'VIF']


def save(name, rows):
    with open(os.path.join(R, name), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (('%.5f' % v) if isinstance(v, float) else v) for k, v in r.items()})


def iou(x, m=None):
    I, P, G = (a if m is None else a[m] for a in x)
    I, P, G = I.sum(0), P.sum(0), G.sum(0)
    return I / np.maximum(P + G - I, 1)


def biou(x, W):
    I, P, G = W @ x[0], W @ x[1], W @ x[2]
    return I / np.maximum(P + G - I, 1)


def twoside(d):
    return min(1., 2 * min(np.mean(d <= 0), np.mean(d >= 0)) + 1 / len(d))


# ---------- retrained ----------
rows = []
for ds in ('MSRS', 'FMB'):
    D = {}
    for i in ['Ours', 'avg', 'vi', 'ir'] + METHODS:
        for s in SEEDS:
            if ds == 'MSRS':
                d = np.load(os.path.join(R, 'seg_retrain', '%s_s%d.npz' % (i, s)))
                x = (d['inter'].astype(float), d['pred'].astype(float), d['gt'].astype(float))
                D[i, s] = {'full': x, 'g3': tuple(a[:, [1, 2, 3]] for a in x)}
            else:
                d = np.load(os.path.join(R, 'seg_retrain_fmb', '%s_s%d.npz' % (i, s)))
                D[i, s] = {'full': tuple(d[k + '_native'].astype(float)[:, [c for c in range(14) if c != 12]] for k in ('inter', 'pred', 'gt')),
                           'g3': tuple(d[k + '3_native'].astype(float) for k in ('inter', 'pred', 'gt'))}
            names = d['names']
    assert all(np.array_equal(D[k]['full'][2], D['avg', 0]['full'][2]) for k in D), 'labels differ'
    N = len(names)
    scopes = [('all', np.ones(N, bool))]
    if ds == 'MSRS':
        night = np.array([n[-5] == 'N' for n in names]); scopes += [('day', ~night), ('night', night)]
    full = 'mIoU9' if ds == 'MSRS' else 'mIoU13'
    for sc, mask in scopes:
        idx = np.where(mask)[0]; n = len(idx)
        W = rng.multinomial(n, np.ones(n) / n, size=B).astype(float)
        S = rng.integers(0, 3, size=(B, 3))
        for score, part in ((full, 'full'), ('mIoU3', 'g3')):
            pt = {i: np.array([iou(D[i, s][part], mask).mean() for s in SEEDS]) for i in ['Ours', 'avg', 'vi', 'ir'] + METHODS}
            bo = {i: np.stack([biou(tuple(a[idx] for a in D[i, s][part]), W).mean(1) for s in SEEDS]) for i in ['Ours', 'avg', 'vi', 'ir']}   # seeds x B
            others = sorted(METHODS, key=lambda m: -pt[m].mean())
            for ref in ['avg', 'vi', 'ir']:
                d = (bo['Ours'] - bo[ref]).mean(0)
                ds_ = (bo['Ours'] - bo[ref])[S, np.arange(B)[:, None]].mean(1)
                rows.append(dict(dataset=ds, scope=sc, score=score, ours=float(pt['Ours'].mean()), ours_sd=float(pt['Ours'].std(ddof=1)),
                                 reference=ref, ref_score=float(pt[ref].mean()), delta=float((pt['Ours'] - pt[ref]).mean()),
                                 lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p=twoside(d),
                                 lo_seed=float(np.percentile(ds_, 2.5)), hi_seed=float(np.percentile(ds_, 97.5)),
                                 rank_among_7=1 + int(sum(pt[m].mean() > pt['Ours'].mean() for m in METHODS)),
                                 best_other=others[0], best_other_score=float(pt[others[0]].mean())))
save('ours_retrain.csv', rows)
print('== Retrained: proposed network minus reference (image bootstrap; images and seeds)')
for r in rows:
    print(' %-4s %-5s %-6s Ours %.4f sd %.4f - %-3s %.4f = %+.4f [%+.4f,%+.4f] p %.4f seeds [%+.4f,%+.4f]  rank %d/7 (best other %s %.4f)' % (
        r['dataset'], r['scope'], r['score'], r['ours'], r['ours_sd'], r['reference'], r['ref_score'], r['delta'], r['lo'], r['hi'], r['p'], r['lo_seed'], r['hi_seed'], r['rank_among_7'], r['best_other'], r['best_other_score']))

# ---------- pretrained ----------
rows = []
for ds, sub in (('MSRS', 'seg'), ('FMB', 'seg_fmb')):
    for model in ('segformer', 'mask2former'):
        ld = lambda k: tuple(np.load(os.path.join(R, sub, model, k + '.npz'))[a].astype(float) for a in ('inter', 'pred', 'gt'))
        o = ld('Ours_native'); N = o[0].shape[0]
        W = rng.multinomial(N, np.ones(N) / N, size=B).astype(float)
        sc = {m: iou(ld(m + '_native')).mean() for m in METHODS}
        for ref in ('avg', 'vi'):
            x = ld('base_' + ref)
            d = (biou(o, W) - biou(x, W)).mean(1)
            rows.append(dict(dataset=ds, model=model, ours=float(iou(o).mean()), reference=ref, ref_score=float(iou(x).mean()),
                             delta=float(iou(o).mean() - iou(x).mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p=twoside(d),
                             rank_among_7=1 + int(sum(v > iou(o).mean() for v in sc.values())), six_min=float(min(sc.values())), six_max=float(max(sc.values()))))
save('ours_pretrained.csv', rows)
print('\n== Pretrained mIoU3: proposed network minus reference')
for r in rows:
    print(' %-4s %-11s Ours %.4f - %-3s %.4f = %+.4f [%+.4f,%+.4f] p %.4f  rank %d/7 (six %.4f-%.4f)' % (
        r['dataset'], r['model'], r['ours'], r['reference'], r['ref_score'], r['delta'], r['lo'], r['hi'], r['p'], r['rank_among_7'], r['six_min'], r['six_max']))

# ---------- metrics, native, descriptive ----------
M = []
for fn in ('cond_metrics.csv', 'cond_metrics_ext.csv', 'cond_metrics_fmb.csv', 'cond_metrics_ours.csv'):
    M += [r for r in csv.DictReader(open(os.path.join(R, fn))) if r['condition'] in ('native', 'avg')]
rows = []
for ds in ('MSRS', 'RoadScene', 'FMB'):
    ims = sorted({r['image'] for r in M if r['dataset'] == ds and r['method'] == 'Ours'})
    look = {(r['method'], r['image']): r for r in M if r['dataset'] == ds and (r['condition'] == 'native' or r['method'] == 'base')}
    for mt in METS:
        sg = -1 if mt == 'Nabf' else 1
        mean = {m: np.nanmean([float(look[m, i][mt]) for i in ims]) for m in METHODS + ['Ours']}
        avg = np.nanmean([float(look['base', i][mt]) for i in ims])
        rk = 1 + sum(sg * mean[m] > sg * mean['Ours'] for m in METHODS)
        rows.append(dict(dataset=ds, metric=mt, n=len(ims), ours=float(mean['Ours']), avg=float(avg), six_min=float(min(mean[m] for m in METHODS)),
                         six_max=float(max(mean[m] for m in METHODS)), rank_among_7=int(rk), above_avg=bool(sg * mean['Ours'] > sg * avg)))
save('ours_metrics_rank.csv', rows)
print('\n== Metrics on the native images (descriptive): rank of the proposed network among the six methods plus itself')
for ds in ('MSRS', 'RoadScene', 'FMB'):
    print(' %-9s ' % ds + ' '.join('%s %d' % (r['metric'], r['rank_among_7']) for r in rows if r['dataset'] == ds), ' n', [r['n'] for r in rows if r['dataset'] == ds][0])
