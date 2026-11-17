# -*- coding: utf-8 -*-
"""Figure of Section 5.4: change of mIoU3 under each transformation, six methods pooled, for the three task judges
(two pretrained Cityscapes models and the SegFormer retrained on fused images, seed 0) on MSRS and FMB.
Reads the result tables of ../metric_sanity/results; nothing is typed by hand. Writes results/audit_forest.png."""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MS = os.path.join('..', 'metric_sanity', 'results')
TR = ['blend25', 'blend50', 'blend75', 'noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
LAB = {'blend25': 'blend a = 0.25', 'blend50': 'blend a = 0.5', 'blend75': 'blend a = 0.75', 'noise1': 'noise σ = 0.01',
       'noise3': 'noise σ = 0.03', 'sharp1': 'unsharp k = 1', 'sharp2': 'unsharp k = 2', 'gamma07': 'gamma 0.7',
       'gamma14': 'gamma 1.4', 'contrast': 'contrast ×1.25', 'blur1': 'blur σ = 1'}
rd = lambda f: list(csv.DictReader(open(os.path.join(MS, f))))
D = {}
for r in rd('seg_holm.csv'):
    if r['family'] == 'transform' and r['scope'] == 'all' and r['item'] in TR:
        D['MSRS', r['model'], r['item']] = (float(r['delta']), float(r['lo']), float(r['hi']), float(r['p_holm']))
for r in rd('fmb_seg_holm.csv'):
    if r['family'] == 'transform' and r['item'] in TR:
        D['FMB', r['model'], r['item']] = (float(r['delta']), float(r['lo']), float(r['hi']), float(r['p_holm']))
for r in rd('retrain_cond_pooled.csv'):
    if r['score'] == 'mIoU3':
        D['MSRS', 'retrained', r['transform']] = (float(r['delta']), float(r['lo']), float(r['hi']), float(r['p_holm']))
for r in rd('fmb_retrain_cond.csv'):
    if r['score'] == 'mIoU3' and r['method'] == 'pooled':
        D['FMB', 'retrained', r['transform']] = (float(r['delta']), float(r['lo']), float(r['hi']), float(r['p_holm']))
assert len(D) == 2 * 3 * len(TR), len(D)

JUDGES = [('segformer', 'SegFormer-B5, pretrained', '#1f77b4', 'o'), ('mask2former', 'Mask2Former, pretrained', '#ff7f0e', 's'),
          ('retrained', 'SegFormer-B5, retrained on fused images', '#2ca02c', 'D')]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
for ax, ds in zip(axes, ['MSRS', 'FMB']):
    for j, (k, lab, col, mk) in enumerate(JUDGES):
        for i, T in enumerate(TR):
            d, lo, hi, ph = D[ds, k, T]
            y = i + (j - 1) * 0.24
            ax.plot([lo, hi], [y, y], color=col, lw=1.4)
            ax.plot(d, y, mk, color=col, ms=5.5, mfc=col if ph < .05 else 'white', mew=1.3)
    ax.axvline(0, color='k', lw=0.8)
    for i in range(len(TR)):
        if i in (2, 6):
            ax.axhline(i + 0.5, color='#bbb', lw=0.6, ls=':')
    ax.set_yticks(range(len(TR))); ax.set_yticklabels([LAB[t] for t in TR], fontsize=10)
    ax.set_xlabel('Change of mIoU3 (transformed − native), six methods pooled', fontsize=10)
    ax.set_title('(%s) %s' % ('a' if ds == 'MSRS' else 'b', ds + (' (361 test images)' if ds == 'MSRS' else ' (280 test images)')), fontsize=11, weight='bold')
    ax.grid(alpha=0.25, axis='x')
axes[0].invert_yaxis()    # shared y axis: invert once
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=c, marker=m, lw=1.4, ms=5.5, label=l) for _, l, c, m in JUDGES]
hs.append(Line2D([], [], color='#555', marker='o', mfc='white', lw=0, ms=5.5, label='open marker: not significant (Holm)'))
axes[0].legend(handles=hs, fontsize=8.5, loc='lower left', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join('results', 'audit_forest.png'), dpi=300, bbox_inches='tight')
print('saved results/audit_forest.png')
