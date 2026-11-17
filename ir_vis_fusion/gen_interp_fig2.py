# -*- coding: utf-8 -*-
"""Figure 3 of the Electronics manuscript: attention map of the single-stream predecessor vs. trust map W, from the
exported network outputs in results/maps/*.mat (same data as gen_interp_fig.py, compact layout).
The number in each map is its within-image standard deviation."""
import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scenes = ['00931N', '00571D', '01258N']
mats = {s: sio.loadmat(os.path.join('results', 'maps', s + '_maps.mat')) for s in scenes}
g = lambda s, k: np.asarray(mats[s][k], dtype=float)
H, W0 = g(scenes[0], 'ir').shape
cols = [('ir', 'Infrared'), ('vi', 'Visible'), ('A', 'Predecessor attention'), ('W', 'Trust map $W$ (ours)'),
        ('F', 'Fused (ours)')]
pw = 2.5                                         # panel width in inches
fig = plt.figure(figsize=(5 * pw + 0.7, 3 * pw * H / W0 + 0.55))
gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], wspace=0.04, hspace=0.06,
                      left=0.005, right=0.955, bottom=0.01, top=0.93)
for r, s in enumerate(scenes):
    for c, (k, title) in enumerate(cols):
        ax = fig.add_subplot(gs[r, c])
        if k in ('ir', 'vi'):
            ax.imshow(g(s, k), cmap='gray', vmin=0, vmax=1)
        elif k == 'F':
            F = g(s, 'F'); F = (F - F.min()) / (F.max() - F.min() + 1e-9)
            ax.imshow(F, cmap='gray', vmin=0, vmax=1)
        else:
            im = ax.imshow(g(s, k), cmap='jet', vmin=0, vmax=1)
            ax.text(0.03, 0.95, 'std %.3f' % g(s, k).std(), transform=ax.transAxes, ha='left', va='top', fontsize=10.5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=13, pad=5)
cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 5]))
cb.ax.tick_params(labelsize=10.5)
plt.savefig(os.path.join('results', 'fig_interp_attention_v2.png'), dpi=300, bbox_inches='tight')
print('saved results/fig_interp_attention_v2.png')
