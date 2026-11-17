# -*- coding: utf-8 -*-
"""Render every condition of ANALYSIS_PLAN.md as an 8-bit luminance PNG, so that the metrics, the segmentation
models and the human study all read exactly the same pixels.
stim/<dataset>/<method>/<condition>/<name>.png, stim/<dataset>/base/{ir,vi,avg}/<name>.png
Also writes results/image_sets.json with the seeded image selections."""
import glob, hashlib, json, os, zlib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'ir_vis_fusion')
OUT = os.path.join(HERE, 'stim')
METHODS = {'DCEvo': 'DCEvo', 'EMMA': 'EMMA', 'LUTFuse': 'LUTFuse', 'MMDRFuse': 'MMDRFuse'}
SEED = 20261031


def to_y(path, size=None):
    a = Image.open(path)
    if a.mode != 'L':
        a = a.convert('RGB')
        x = np.asarray(a, float) / 255
        y = .299 * x[..., 0] + .587 * x[..., 1] + .114 * x[..., 2]
    else:
        y = np.asarray(a, float) / 255
    if size is not None and y.shape != size:
        y = np.asarray(Image.fromarray(np.uint8(np.round(y * 255))).resize((size[1], size[0]), Image.BICUBIC), float) / 255
    return y


def conditions(F, avg, name):
    """All transformed versions of F. The noise seed depends on the image name only."""
    g = np.random.default_rng(zlib.crc32(name.encode()))
    n = g.standard_normal(F.shape)
    b = gaussian_filter(F, 1.5)
    return {
        'native': F,
        'blend25': .75 * F + .25 * avg, 'blend50': .5 * F + .5 * avg, 'blend75': .25 * F + .75 * avg,
        'noise1': F + .01 * n, 'noise3': F + .03 * n,
        'sharp1': F + (F - b), 'sharp2': F + 2 * (F - b),
        'gamma07': np.clip(F, 0, 1) ** .7, 'gamma14': np.clip(F, 0, 1) ** 1.4,
        'contrast': 1.25 * (F - F.mean()) + F.mean(),
        'blur1': gaussian_filter(F, 1.0),
        'catch': gaussian_filter(F, 3.0) + .1 * n,
    }


def save(y, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(y, 0, 1) * 255))).save(path, compress_level=6)


def render(ds, irdir, vidir, names, suffix):
    for k, name in enumerate(names):
        vi = to_y(os.path.join(vidir, name)); ir = to_y(os.path.join(irdir, name), vi.shape)
        avg = (ir + vi) / 2
        for c, y in (('ir', ir), ('vi', vi), ('avg', avg)):
            save(y, os.path.join(OUT, ds, 'base', c, name))
        for m in METHODS:
            F = to_y(os.path.join(ROOT, 'results', 'sota', m + suffix, name), vi.shape)
            for c, y in conditions(F, avg, name).items():
                save(y, os.path.join(OUT, ds, m, c, name))
        if k % 50 == 0:
            print(ds, k, len(names), flush=True)


if __name__ == '__main__':
    msrs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'tmp_msrs', 'test', 'vi', '*.png')))
    for m in METHODS:
        have = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'results', 'sota', m + '_MSRS', '*'))}
        assert set(msrs) <= have, m
    rs_common = None
    for m in METHODS:
        have = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'results', 'sota', m + '_RoadScene', '*'))}
        rs_common = have if rs_common is None else rs_common & have
    rs_common = sorted(n for n in rs_common if os.path.exists(os.path.join(ROOT, 'data', 'RoadScene', 'vi', n)))
    g = np.random.default_rng(SEED)
    metric_msrs = sorted(g.choice(msrs, 100, replace=False).tolist())
    human_msrs = sorted(g.choice(metric_msrs, 60, replace=False).tolist())
    rs = sorted(g.choice(rs_common, 30, replace=False).tolist())
    with open(os.path.join(HERE, 'results', 'image_sets.json'), 'w') as fh:
        json.dump(dict(seed=SEED, msrs_all=msrs, metric_msrs=metric_msrs, human_msrs=human_msrs, roadscene=rs), fh, indent=0)
    print('MSRS', len(msrs), 'RoadScene common', len(rs_common), 'day/night metric set',
          sum(n[-5] == 'D' for n in metric_msrs), sum(n[-5] == 'N' for n in metric_msrs), flush=True)
    render('RoadScene', os.path.join(ROOT, 'data', 'RoadScene', 'ir'), os.path.join(ROOT, 'data', 'RoadScene', 'vi'), rs, '_RoadScene')
    render('MSRS', os.path.join(ROOT, 'tmp_msrs', 'test', 'ir'), os.path.join(ROOT, 'tmp_msrs', 'test', 'vi'), msrs, '_MSRS')
    print('DONE', flush=True)
