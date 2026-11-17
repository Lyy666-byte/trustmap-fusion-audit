# -*- coding: utf-8 -*-
"""Task half of the decision rule of ANALYSIS_PLAN.md on the six-method tables, as cause_verdict.py on the first pass.
Human data are not in, so a metric that rises significantly while mIoU3 falls in at least one model is a candidate,
not a final verdict. Also repeats the metric test per method, so a pooled rise carried by a few methods is visible.
results/cause6_verdict.csv, cause6_metrics_permethod.csv."""
import csv, os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
seg = {(r['model'], r['transform'], r['scope']): r for r in csv.DictReader(open(os.path.join(R, 'cause6_seg.csv')))}
met = list(csv.DictReader(open(os.path.join(R, 'cause6_metrics.csv'))))
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
TRANS = ['blend25', 'blend50', 'blend75', 'avg', 'noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
FONLY = ['noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
METS = ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'MS_SSIM', 'CC', 'Qabf', 'FMI', 'Nabf', 'SCD', 'VIF']

out = []
for T in TRANS:
    a = seg[('segformer', T, 'all')]; b = seg[('mask2former', T, 'all')]
    la, ha, lb, hb = float(a['lo']), float(a['hi']), float(b['lo']), float(b['hi'])
    if ha <= 0 or hb <= 0:
        task = 'worse' if (ha <= 0 and hb <= 0) else 'mixed-'
    elif la > 0 and lb > 0:
        task = 'better'
    elif la > 0 or lb > 0:
        task = 'mixed+'
    else:
        task = 'flat'
    up = {ds: [m['metric'] for m in met if m['dataset'] == ds and m['transform'] == T and m['verdict'] == 'up'] for ds in ['MSRS', 'RoadScene']}
    both = [m for m in up['MSRS'] if m in up['RoadScene']]
    print('%-9s %+.4f[%+.4f,%+.4f] %+.4f[%+.4f,%+.4f] %-6s| %s | %s' % (T, float(a['delta_mIoU3']), la, ha, float(b['delta_mIoU3']), lb, hb, task, ','.join(up['MSRS']), ','.join(both)))
    for m in METS:
        cand = m in up['MSRS'] and (ha <= 0 or hb <= 0)
        out.append(dict(transform=T, metric=m, up_msrs=m in up['MSRS'], up_roadscene=m in up['RoadScene'], task=task, candidate_inflated=cand))
with open(os.path.join(R, 'cause6_verdict.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print('\nCandidates (up on MSRS while mIoU3 falls in at least one model):')
for m in METS:
    ts = [r['transform'] for r in out if r['metric'] == m and r['candidate_inflated']]
    print('  %-8s %s' % (m, ', '.join(ts) if ts else '-'))
print('\nTransformations of F alone that raise each metric (Holm), MSRS / RoadScene, median change:')
for m in METS:
    line = []
    for ds in ['MSRS', 'RoadScene']:
        line.append(' '.join('%s(%+.4f)' % (r['transform'], float(r['median_change'])) for r in met if r['dataset'] == ds and r['metric'] == m and r['transform'] in FONLY and r['verdict'] == 'up') or '-')
    print('  %-8s %s  /  %s' % (m, line[0], line[1]))

D = list(csv.DictReader(open(os.path.join(R, 'cond_metrics.csv')))) + list(csv.DictReader(open(os.path.join(R, 'cond_metrics_ext.csv'))))
look = {(r['dataset'], r['method'], r['condition'], r['image']): r for r in D}
nan = sum(1 for r in D if r['method'] in ('CDDFuse', 'SeAFusion') and any(not np.isfinite(float(r[m])) for m in METS))
print('\nnon-finite metric rows in the two added methods:', nan)
rows = []
print('Pooled rise on both datasets, number of the six methods that also rise on their own (Holm over 13 metrics), MSRS/RoadScene:')
for ds in ['MSRS', 'RoadScene']:
    images = sorted({r['image'] for r in D if r['dataset'] == ds})
    for T in TRANS:
        for me in METHODS:
            ps, meds = [], []
            for mt in METS:
                sg = -1 if mt == 'Nabf' else 1
                b = [float(look[(ds, 'base', 'avg', n)][mt]) if T == 'avg' else float(look[(ds, me, T, n)][mt]) for n in images]
                d = sg * (np.array(b) - np.array([float(look[(ds, me, 'native', n)][mt]) for n in images]))
                d = d[np.isfinite(d)]
                ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0); meds.append(np.median(d))
            order = np.argsort(ps); run = 0; holm = [0] * len(ps)
            for rank, i in enumerate(order):
                run = max(run, min(1, ps[i] * (len(ps) - rank))); holm[i] = run
            for mt, md, h in zip(METS, meds, holm):
                rows.append(dict(dataset=ds, transform=T, method=me, metric=mt, median_change=md, p_holm=h, up=(h < .05 and md > 0)))
with open(os.path.join(R, 'cause6_metrics_permethod.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows:
        w.writerow({k: (('%.5g' % v) if isinstance(v, float) else v) for k, v in r.items()})
for T in TRANS:
    pooled = [m for m in METS if all(any(r['dataset'] == ds and r['transform'] == T and r['metric'] == m and r['verdict'] == 'up' for r in met) for ds in ['MSRS', 'RoadScene'])]
    cells = []
    for m in pooled:
        c = [sum(r['up'] for r in rows if r['dataset'] == ds and r['transform'] == T and r['metric'] == m) for ds in ['MSRS', 'RoadScene']]
        cells.append('%s %d/%d' % (m, c[0], c[1]))
    print('  %-9s %s' % (T, ', '.join(cells) or '-'))
