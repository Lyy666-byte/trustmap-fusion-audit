# -*- coding: utf-8 -*-
"""Gate before training: run each fusion script on three MSRS test images and compare with the luminance already used as
the test stimuli, so that train and test images come from one pipeline. Run from ir_vis_fusion. Exit 1 on mismatch."""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image
from make_conditions import to_y

HERE = os.path.dirname(os.path.abspath(__file__))
FUS = os.path.abspath(os.path.join(HERE, '..', 'ir_vis_fusion'))
names = json.load(open(os.path.join(HERE, 'results', 'image_sets.json')))['msrs_all'][::150][:3]
tmp = tempfile.mkdtemp()
for sub in ['ir', 'vi']:
    os.makedirs(os.path.join(tmp, sub))
    for n in names:
        shutil.copy(os.path.join(FUS, 'tmp_msrs', 'test', sub, n), os.path.join(tmp, sub, n))
ok = True
for m, s in [('CDDFuse', 'cddfuse'), ('EMMA', 'emma'), ('LUTFuse', 'lutfuse'), ('MMDRFuse', 'mmdrfuse'), ('DCEvo', 'dcevo'), ('SeAFusion', None)]:
    out = os.path.join(tmp, 'out_' + m)
    if s:
        cmd, cwd = [sys.executable, 'sota_%s_infer.py' % s, os.path.join(tmp, 'ir'), os.path.join(tmp, 'vi'), out], FUS
    else:
        cmd, cwd = [sys.executable, 'test.py', '-M', './model/Fusion/fusionmodel_final.pth', '-ir_dir', os.path.join(tmp, 'ir'), '-vi_dir', os.path.join(tmp, 'vi'), '-save_dir', out, '-j', '0'], os.path.join(FUS, 'sota', 'SeAFusion')
    r = subprocess.run(cmd, cwd=cwd, capture_output=True)
    if r.returncode:
        print(m, 'failed', r.stderr[-400:]); ok = False; continue
    worst = 0.
    for n in names:
        a = to_y(os.path.join(out, n)); b = to_y(os.path.join(HERE, 'stim', 'MSRS', m, 'native', n))
        worst = max(worst, float(np.abs(np.round(a * 255) - np.round(b * 255)).max()))
    print('%-10s largest difference from the test stimuli on 3 images: %.0f gray levels' % (m, worst), flush=True)
    ok &= worst <= 2
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if ok else 1)
