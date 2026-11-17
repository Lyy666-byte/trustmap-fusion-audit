# -*- coding: utf-8 -*-
"""Second pass of ANALYSIS_PLAN.md, six methods. Same statistics as cause_analyze.py, which stays the record of the
first pass. Adds each method's native output against the pixel average, per model and scope.
Writes results/cause6_seg.csv, cause6_native_vs_avg.csv, cause6_metrics.csv."""
import csv, os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
TRANS = ['blend25', 'blend50', 'blend75', 'noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
METS = ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'MS_SSIM', 'CC', 'Qabf', 'FMI', 'Nabf', 'SCD', 'VIF']
rng = np.random.default_rng(20261031)
B = 2000


def load(model, key):
    d = np.load(os.path.join(R, 'seg', model, key + '.npz'))
    return d['inter'].astype(float), d['pred'].astype(float), d['gt'].astype(float), d['names']


def miou(x, w):
    I, P, G = (w[:, None] * x[0]).sum(0), (w[:, None] * x[1]).sum(0), (w[:, None] * x[2]).sum(0)
    return (I / np.maximum(P + G - I, 1)).mean()


def boot(x, W):
    I, P, G = np.einsum('bn,nc->bc', W, x[0]), np.einsum('bn,nc->bc', W, x[1]), np.einsum('bn,nc->bc', W, x[2])
    return (I / np.maximum(P + G - I, 1)).mean(1)


seg, nva = [], []
for model in ['segformer', 'mask2former']:
    native = {m: load(model, m + '_native') for m in METHODS}
    avg = load(model, 'base_avg'); vis = load(model, 'base_vi')
    names = avg[3]; N = len(names)
    night = np.array([n[-5] == 'N' for n in names])
    weights = rng.multinomial(N, np.ones(N) / N, size=B).astype(float)
    scopes = (('all', np.ones(N, bool)), ('day', ~night), ('night', night))
    print('\n==', model)
    print('native minus pixel average, mIoU3 [95% CI]; mIoU3 of the native, average and visible in the first column')
    for m in METHODS:
        line = []
        for sc, mask in scopes:
            w0 = mask.astype(float); W = weights * w0
            a, b, v = miou(native[m], w0), miou(avg, w0), miou(vis, w0)
            lo, hi = np.percentile(boot(native[m], W) - boot(avg, W), [2.5, 97.5])
            vlo, vhi = np.percentile(boot(native[m], W) - boot(vis, W), [2.5, 97.5])
            nva.append(dict(model=model, method=m, scope=sc, native=a, avg=b, vis=v, minus_avg=a - b, lo=lo, hi=hi, minus_vis=a - v, vis_lo=vlo, vis_hi=vhi))
            line.append('%s %.3f %+.4f[%+.4f,%+.4f] vs vis %+.4f[%+.4f,%+.4f]' % (sc, a, a - b, lo, hi, a - v, vlo, vhi))
        print(' %-10s' % m + ' | '.join(line))
    print(' average: ' + ' '.join('%s %.3f' % (s, miou(avg, k.astype(float))) for s, k in scopes) + '   visible: ' + ' '.join('%s %.3f' % (s, miou(vis, k.astype(float))) for s, k in scopes))
    for T in TRANS + ['avg']:
        cond = {m: (avg if T == 'avg' else load(model, m + '_' + T)) for m in METHODS}
        for sc, mask in scopes:
            w0 = mask.astype(float); W = weights * w0
            dpm = {m: miou(cond[m], w0) - miou(native[m], w0) for m in METHODS}
            d = np.mean(list(dpm.values()))
            lo, hi = np.percentile(np.mean([boot(cond[m], W) - boot(native[m], W) for m in METHODS], 0), [2.5, 97.5])
            seg.append(dict(model=model, transform=T, scope=sc, delta_mIoU3=d, lo=lo, hi=hi, **{'d_' + m: dpm[m] for m in METHODS}))
            if sc == 'all':
                print(' %-8s %+.4f [%+.4f,%+.4f]  %s' % (T, d, lo, hi, ' '.join('%+.3f' % dpm[m] for m in METHODS)))
for name, rows in (('cause6_seg.csv', seg), ('cause6_native_vs_avg.csv', nva)):
    with open(os.path.join(R, name), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (('%.5f' % v) if isinstance(v, float) else v) for k, v in r.items()})

D = list(csv.DictReader(open(os.path.join(R, 'cond_metrics.csv')))) + list(csv.DictReader(open(os.path.join(R, 'cond_metrics_ext.csv'))))
look = {(r['dataset'], r['method'], r['condition'], r['image']): r for r in D}
out = []
for ds in ['MSRS', 'RoadScene']:
    images = sorted({r['image'] for r in D if r['dataset'] == ds})
    print('\n== metrics', ds, 'images', len(images), '(median change per image, averaged over six methods; * = Holm p<.05)')
    print('%-9s' % '' + ' '.join('%7s' % m for m in METS))
    for T in TRANS + ['avg']:
        ps, meds, ups = [], [], []
        for mt in METS:
            sg = -1 if mt == 'Nabf' else 1
            d = []
            for n in images:
                vals = []
                for m in METHODS:
                    a = float(look[(ds, m, 'native', n)][mt])
                    b = float(look[(ds, 'base', 'avg', n)][mt]) if T == 'avg' else float(look[(ds, m, T, n)][mt])
                    vals.append(sg * (b - a))
                d.append(np.mean(vals))
            d = np.array(d); d = d[np.isfinite(d)]
            ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0); meds.append(np.median(d)); ups.append(np.mean(d > 0))
        order = np.argsort(ps); run = 0; holm = [0] * len(ps)
        for rank, i in enumerate(order):
            run = max(run, min(1, ps[i] * (len(ps) - rank))); holm[i] = run
        for mt, med, up, pr, h in zip(METS, meds, ups, ps, holm):
            out.append(dict(dataset=ds, transform=T, metric=mt, median_change=med, frac_up=up, p=pr, p_holm=h,
                            verdict='up' if (h < .05 and med > 0) else ('down' if (h < .05 and med < 0) else 'ns')))
        print('%-9s' % T + ' '.join('%6.3f%s' % (m, '*' if h < .05 else ' ') for m, h in zip(meds, holm)))
with open(os.path.join(R, 'cause6_metrics.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader()
    for r in out:
        w.writerow({k: (('%.5g' % v) if isinstance(v, float) else v) for k, v in r.items()})

print('\nMetrics up on both datasets, six methods:')
for T in TRANS + ['avg']:
    u = {ds: {r['metric'] for r in out if r['dataset'] == ds and r['transform'] == T and r['verdict'] == 'up'} for ds in ['MSRS', 'RoadScene']}
    print(' %-9s %s' % (T, ','.join(m for m in METS if m in u['MSRS'] and m in u['RoadScene']) or '-'))
