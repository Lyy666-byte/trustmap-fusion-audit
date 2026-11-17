# -*- coding: utf-8 -*-
"""Second pass of ANALYSIS_PLAN.md: render the same conditions for CDDFuse and SeAFusion on the image sets already
fixed in results/image_sets.json. Sources and baselines are not touched; the noise seed depends on the image name only,
so both methods get the same noise as the first four."""
import json, os
from make_conditions import HERE, ROOT, OUT, conditions, save, to_y

S = json.load(open(os.path.join(HERE, 'results', 'image_sets.json')))
SETS = [('RoadScene', S['roadscene'], '_RoadScene'), ('MSRS', S['msrs_all'], '_MSRS361')]
for ds, names, suffix in SETS:
    for k, name in enumerate(names):
        vi = to_y(os.path.join(OUT, ds, 'base', 'vi', name)); ir = to_y(os.path.join(OUT, ds, 'base', 'ir', name))
        avg = (ir + vi) / 2
        for m in ['CDDFuse', 'SeAFusion']:
            p = os.path.join(ROOT, 'results', 'sota', m + suffix, name)
            F = to_y(p, vi.shape)
            for c, y in conditions(F, avg, name).items():
                save(y, os.path.join(OUT, ds, m, c, name))
        if k % 50 == 0:
            print(ds, k, len(names), flush=True)
print('DONE', flush=True)
