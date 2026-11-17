# -*- coding: utf-8 -*-
"""Complete the qualitative panels of the TNO scene (5.png, exported by export_roi.m) in results/export2: pixel average from the exported sources, and the saved
outputs of the methods run in Python (grey conversion with the MATLAB rgb2gray weights, written as 8-bit).
The exported infrared and visible panels are first checked against the dataset."""
import os
import numpy as np
from PIL import Image

OUT = os.path.join('results', 'export2')


def gray(path):
    a = np.asarray(Image.open(path)).astype(np.float64) / 255.0
    if a.ndim == 3:
        a = 0.298936021293775 * a[..., 0] + 0.587043074451121 * a[..., 1] + 0.114020904255103 * a[..., 2]
    return a


def save(a, path):
    Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(path)


for tag, name, data, ds, methods in (
        ('TNO', '5.png', os.path.join('data', 'TNO'), 'TNO', ('MMDRFuse', 'LUTFuse')),):
    ir, vi = gray(os.path.join(data, 'ir', name)), gray(os.path.join(data, 'vi', name))
    ir_e, vi_e = gray(os.path.join(OUT, tag + '_ir.png')), gray(os.path.join(OUT, tag + '_vis.png'))
    assert ir.shape == ir_e.shape and np.abs(ir - ir_e).max() < 1 / 255 + 1e-9, (tag, 'infrared panel is not ' + name)
    assert vi.shape == vi_e.shape and np.abs(vi - vi_e).max() < 1 / 255 + 1e-9, (tag, 'visible panel is not ' + name)
    save((ir_e + vi_e) / 2, os.path.join(OUT, tag + '_avg.png'))
    for m in methods:
        f = gray(os.path.join('results', 'sota', '%s_%s' % (m, ds), name))
        assert f.shape == ir.shape, (tag, m, f.shape, ir.shape)
        save(f, os.path.join(OUT, '%s_%s.png' % (tag, m.lower())))
    print(tag, name, 'panels written:', ['avg'] + [m.lower() for m in methods])
