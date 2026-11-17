# -*- coding: utf-8 -*-
"""FMB, as fixed in ANALYSIS_PLAN.md: extract images to ir_vis_fusion/tmp_fmb/{train,test}/{ir,vi,Label}, draw the 100 metric images.
Writes results/image_sets_fmb.json. Safe to run again."""
import json, os, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', 'fmb_raw')
OUT = os.path.join(HERE, '..', 'ir_vis_fusion', 'tmp_fmb')
FOLD = {'Infrared': 'ir', 'Visible': 'vi', 'Label': 'Label'}
SEED = 20261031
for split, n in (('train', 1220), ('test', 280)):
    z = zipfile.ZipFile(os.path.join(RAW, split + '.zip'))
    for src, dst in FOLD.items():
        d = os.path.join(OUT, split, dst); os.makedirs(d, exist_ok=True)
        members = [i for i in z.namelist() if i.startswith('%s/%s/' % (split, src)) and i.endswith('.png')]
        assert len(members) == n, (split, src, len(members))
        for m in members:
            f = os.path.join(d, os.path.basename(m))
            if not os.path.exists(f):
                with open(f, 'wb') as fh:
                    fh.write(z.read(m))
    print(split, 'extracted', flush=True)
names = sorted(os.listdir(os.path.join(OUT, 'test', 'vi')))
g = np.random.default_rng(SEED)
with open(os.path.join(HERE, 'results', 'image_sets_fmb.json'), 'w') as fh:
    json.dump(dict(seed=SEED, fmb_all=names, metric_fmb=sorted(g.choice(names, 100, replace=False).tolist())), fh, indent=0)
print('DONE', flush=True)
