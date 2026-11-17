# -*- coding: utf-8 -*-
"""Add the pixel average, MMDRFuse and LUT-Fuse panels of the Figure-2 scene (MSRS 00706N) to results/export2,
processed like the other panels of export_roi.m: grey conversion with the MATLAB rgb2gray weights, written as 8-bit.
The infrared and visible panels already in export2 are checked against the dataset first."""
import os
import numpy as np
from PIL import Image

NAME, TAG, OUT = '00706N.png', 'MSRS', os.path.join('results', 'export2')


def gray(path):
    a = np.asarray(Image.open(path)).astype(np.float64) / 255.0
    if a.ndim == 3:
        a = 0.298936021293775 * a[..., 0] + 0.587043074451121 * a[..., 1] + 0.114020904255103 * a[..., 2]
    return a


def save(a, path):
    Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(path)


ir, vi = gray(os.path.join('tmp_msrs', 'test', 'ir', NAME)), gray(os.path.join('tmp_msrs', 'test', 'vi', NAME))
ir_e, vi_e = gray(os.path.join(OUT, TAG + '_ir.png')), gray(os.path.join(OUT, TAG + '_vis.png'))
assert np.abs(ir - ir_e).max() < 1 / 255 + 1e-9 and np.abs(vi - vi_e).max() < 1 / 255 + 1e-9, 'export2 is not scene 00706N'
save((ir_e + vi_e) / 2, os.path.join(OUT, TAG + '_avg.png'))
for m, d in (('mmdrfuse', 'MMDRFuse_MSRS'), ('lutfuse', 'LUTFuse_MSRS')):
    f = gray(os.path.join('results', 'sota', d, NAME))
    assert f.shape == ir.shape, (m, f.shape, ir.shape)
    save(f, os.path.join(OUT, '%s_%s.png' % (TAG, m)))
print('wrote', [TAG + s for s in ('_avg.png', '_mmdrfuse.png', '_lutfuse.png')])
