# -*- coding: utf-8 -*-
"""Addendum 2026-11-16 of ANALYSIS_PLAN.md: native 8-bit luminance of the proposed network of the companion paper, by the
same rule as make_conditions.py. stim/<dataset>/Ours/native/<name>.png for MSRS (361), FMB (280) and RoadScene (30)."""
import glob, json, os
import numpy as np
from make_conditions import HERE, ROOT, OUT, save, to_y

S = json.load(open(os.path.join(HERE, 'results', 'image_sets.json')))
fmb = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'tmp_fmb', 'test', 'vi', '*.png')))
SETS = [('MSRS', S['msrs_all'], 'Ours_MSRS361'), ('RoadScene', S['roadscene'], 'Ours_RoadScene'), ('FMB', fmb, 'Ours_FMB')]
for ds, names, src in SETS:
    have = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'results', 'sota', src, '*.png'))}
    assert set(names) <= have, (ds, len(set(names) - have))
    for name in names:
        vi = to_y(os.path.join(OUT, ds, 'base', 'vi', name))
        F = to_y(os.path.join(ROOT, 'results', 'sota', src, name), vi.shape)
        assert F.shape == vi.shape and F.std() > 0, name
        save(F, os.path.join(OUT, ds, 'Ours', 'native', name))
    print(ds, len(names), flush=True)
print('DONE', flush=True)
