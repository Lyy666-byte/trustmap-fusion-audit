# -*- coding: utf-8 -*-
"""FMB, as fixed in ANALYSIS_PLAN.md: render the conditions of the six methods on the 280 test images, without the catch condition.
stim/FMB/<method>/<condition>/<name>.png, stim/FMB/base/{ir,vi,avg}/<name>.png. Also checks every fused image: size and not constant."""
import json, os
import numpy as np
from make_conditions import HERE, ROOT, OUT, conditions, save, to_y

METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
TMP = os.path.join(ROOT, 'tmp_fmb')
names = json.load(open(os.path.join(HERE, 'results', 'image_sets_fmb.json')))['fmb_all']
for m in METHODS:
    for suffix, split in (('_FMBtrain', 'train'), ('_FMB', 'test')):
        d = os.path.join(ROOT, 'results', 'sota', m + suffix)
        files = sorted(os.listdir(d))
        assert files == sorted(os.listdir(os.path.join(TMP, split, 'vi'))), (m, split, len(files))
        for n in files:
            y = to_y(os.path.join(d, n))
            assert y.shape == (600, 800) and y.max() > y.min(), (m, split, n, y.shape)
    print(m, 'fused images checked', flush=True)
for k, name in enumerate(names):
    vi = to_y(os.path.join(TMP, 'test', 'vi', name)); ir = to_y(os.path.join(TMP, 'test', 'ir', name))
    avg = (ir + vi) / 2
    for c, y in (('ir', ir), ('vi', vi), ('avg', avg)):
        save(y, os.path.join(OUT, 'FMB', 'base', c, name))
    for m in METHODS:
        F = to_y(os.path.join(ROOT, 'results', 'sota', m + '_FMB', name))
        for c, y in conditions(F, avg, name).items():
            if c != 'catch':
                save(y, os.path.join(OUT, 'FMB', m, c, name))
    if k % 50 == 0:
        print('FMB', k, len(names), flush=True)
print('DONE', flush=True)
