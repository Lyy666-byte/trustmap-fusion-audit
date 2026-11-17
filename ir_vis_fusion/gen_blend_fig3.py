# -*- coding: utf-8 -*-
"""Metric sanity check figure: SSIM, Qabf and Nabf against MI when every deep method's output is blended
with the pixel average of the sources, F_a = (1-a) F + a (I_ir + I_vis)/2, a in {0, 0.15, 0.30, 0.50}.
All values are read from results/blend_frontier.csv (blend_frontier.m); the Average baseline (a = 1)
is read from the per-item caches of Tables 1-3 (results/peritem_<tag>.mat)."""
import os, csv
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = 'results'
rows = list(csv.DictReader(open(os.path.join(d, 'blend_frontier.csv'))))
methods = ['DenseFuse', 'SeAFusion', 'CDDFuse', 'EMMA', 'DCEvo', 'MMDRFuse', 'LUTFuse', 'Ours']
styles = {
    'DenseFuse': dict(color='#ff7f0e', marker='x', lw=1.1),
    'SeAFusion': dict(color='#9467bd', marker='P', lw=1.1),
    'CDDFuse':   dict(color='#17becf', marker='D', lw=1.2),
    'EMMA':      dict(color='#8c564b', marker='*', lw=1.2),
    'DCEvo':     dict(color='#e377c2', marker='h', lw=1.2),
    'MMDRFuse':  dict(color='#2ca02c', marker='s', lw=1.1),
    'LUTFuse':   dict(color='#1f77b4', marker='^', lw=1.1),
    'Ours':      dict(color='#d62728', marker='o', lw=2.2),
}
disp = {m: m for m in methods}; disp['LUTFuse'] = 'LUT-Fuse'; disp['Ours'] = 'Ours'
mets = [('SSIM', 'SSIM'), ('Qabf', 'Qabf'), ('Nabf', 'Nabf (lower is better)')]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4)); axes = axes[None, :]
for r, tag in enumerate(['MSRS']):
    P = sio.loadmat(os.path.join(d, f'peritem_{tag}.mat'), squeeze_me=True)
    pm = [str(x) for x in np.ravel(P['methods'])]; pn = [str(x) for x in np.ravel(P['names'])]
    avg = {k: P['ACC'][pm.index('average'), pn.index(k), :].mean() for k in ['SSIM', 'Qabf', 'Nabf', 'MI']}
    for c, (met, lab) in enumerate(mets):
        ax = axes[r, c]
        for m in methods:
            rs = sorted([x for x in rows if x['dataset'] == tag and x['method'] == m], key=lambda x: float(x['a']))
            mi = [float(x['MI']) for x in rs]; v = [float(x[met]) for x in rs]
            st = styles[m]
            ax.plot(mi, v, '-', color=st['color'], lw=st['lw'], alpha=0.9, zorder=3 if m == 'Ours' else 2)
            ax.plot(mi[1:], v[1:], st['marker'], color=st['color'], ms=5.5, mfc='none', zorder=4)
            ax.plot(mi[0], v[0], st['marker'], color=st['color'], ms=10, label=disp[m], zorder=5)
        ax.plot(avg['MI'], avg[met], marker='*', color='#555555', ms=15, ls='none', label="Pixel average", zorder=5)
        ax.set_xlabel('MI (bits)', fontsize=13); ax.set_ylabel(lab, fontsize=13)
        ax.set_title(f'{met} vs. MI', fontsize=14, weight="bold"); ax.tick_params(labelsize=12); ax.grid(alpha=0.3)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=9, fontsize=12.5, handletextpad=0.3, columnspacing=1.2, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.09, 1, 1])
out = os.path.join(d, 'blend_frontier_msrs.png')
plt.savefig(out, dpi=300, bbox_inches='tight'); print('saved', out)
