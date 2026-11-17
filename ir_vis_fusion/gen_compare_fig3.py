# -*- coding: utf-8 -*-
"""ROI qualitative comparison with all methods of Table 1 except Laplacian and wavelet: 3 x 4 panels from
results/export2 (export_roi.m + export_light.py).
Usage: python gen_compare_fig3.py TAG OUTNAME RX RY RW RH"""
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

tag = sys.argv[1]; outname = sys.argv[2]
rx, ry, rw, rh = [float(sys.argv[i]) for i in range(3, 7)]
d = os.path.join('results', 'export2')
panels = [('ir', 'Infrared'), ('vis', 'Visible'), ('avg', 'Pixel average'), ('gff', 'GFF'),
          ('densefuse', 'DenseFuse'), ('seafusion', 'SeAFusion'), ('cddfuse', 'CDDFuse'), ('emma', 'EMMA'),
          ('dcevo', 'DCEvo'), ('mmdrfuse', 'MMDRFuse'), ('lutfuse', 'LUT-Fuse'), ('ours', 'Ours')]
fig, axes = plt.subplots(3, 4, figsize=(12.8, 8.3))
for ax, (m, title) in zip(axes.ravel(), panels):
    img = mpimg.imread(os.path.join(d, '%s_%s.png' % (tag, m)))
    if img.ndim == 3:
        img = img[..., 0]
    H, W = img.shape
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, pad=4, fontweight=('bold' if m == 'ours' else 'normal'))
    ax.set_xticks([]); ax.set_yticks([])
    x0, y0, w0, h0 = int(rx * W), int(ry * H), int(rw * W), int(rh * H)
    ax.add_patch(Rectangle((x0, y0), w0, h0, edgecolor='red', facecolor='none', lw=1.6))
    iax = ax.inset_axes([1 - 0.44, 0, 0.44, 0.44])
    iax.imshow(img[y0:y0 + h0, x0:x0 + w0], cmap='gray', vmin=0, vmax=1)
    iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor('red'); s.set_linewidth(1.8)
plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=1.4)
plt.savefig(os.path.join('results', outname), dpi=300, bbox_inches='tight')
print('saved', outname)
