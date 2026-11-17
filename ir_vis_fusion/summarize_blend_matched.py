# -*- coding: utf-8 -*-
"""Summarise results/blend_matched.csv (blend_matched.m) over the five training seeds.

For every dataset and competing deep method: the number of seeds for which the MI match was attainable,
which output was blended, the range of the blend weight a, and for SSIM, Qabf, Nabf and VIF the mean of
(ours - other) over the attainable seeds together with the number of seeds in which ours is significantly
better / significantly worse (paired Wilcoxon signed-rank test, p < 0.05; for Nabf lower is better).
Writes results/blend_matched_summary.csv."""
import csv, os
import numpy as np

R = 'results'
rows = list(csv.DictReader(open(os.path.join(R, 'blend_matched.csv'))))
mets = ['SSIM', 'Qabf', 'Nabf', 'VIF']
out = []
for ds in ['MSRS', 'TNO', 'RoadScene']:
    for m in ['DenseFuse', 'SeAFusion', 'CDDFuse', 'EMMA', 'DCEvo', 'MMDRFuse', 'LUTFuse']:
        rs = [r for r in rows if r['dataset'] == ds and r['method'] == m]
        ok = [r for r in rs if r['a'] != 'NaN']
        rec = dict(dataset=ds, method=m, n_seeds=len(rs), n_attainable=len(ok),
                   seeds_attainable=' '.join(r['seed'] for r in ok),
                   blended=' '.join(sorted({r['blended'] for r in ok})),
                   a_min=min((float(r['a']) for r in ok), default=float('nan')),
                   a_max=max((float(r['a']) for r in ok), default=float('nan')))
        for k in mets:
            d = [float(r[f'{k}_ours']) - float(r[f'{k}_other']) for r in ok]
            sgn = -1 if k == 'Nabf' else 1
            better = sum(1 for r, x in zip(ok, d) if float(r[f'{k}_p']) < 0.05 and sgn * x > 0)
            worse = sum(1 for r, x in zip(ok, d) if float(r[f'{k}_p']) < 0.05 and sgn * x < 0)
            rec[f'{k}_diff'] = float(np.mean(d)) if d else float('nan')
            rec[f'{k}_better'] = better; rec[f'{k}_worse'] = worse
        out.append(rec)
keys = list(out[0].keys())
with open(os.path.join(R, 'blend_matched_summary.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
    for rec in out:
        w.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in rec.items()})
for rec in out:
    print(f"{rec['dataset']:9s} {rec['method']:9s} n={rec['n_attainable']} ({rec['seeds_attainable']:9s}) "
          f"blend={rec['blended']:10s} a={rec['a_min']:.2f}-{rec['a_max']:.2f} " +
          ' '.join(f"{k} {rec[k+'_diff']:+.3f} ({rec[k+'_better']}/{rec[k+'_worse']})" for k in mets))
