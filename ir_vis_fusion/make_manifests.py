# -*- coding: utf-8 -*-
"""Exact file lists (with checksums) of every image subset used in the paper.

    python make_manifests.py           # write data_manifest/*.csv from the local data
    python make_manifests.py --check   # verify the local data against data_manifest/*.csv

Each row: subset, modality, file, height, width, channels, sha256 of the DECODED
pixel array (robust to PNG re-encoding) and sha256 of the file bytes.

Subset rules (identical to the code that produced the paper's numbers):
  MSRS_train200    first 200 pairs of MSRS train/ in file-name order (prepare_data.m)
  MSRS_test30      every 12th pair of the 361 MSRS test/ pairs in file-name order
                   (indices 0, 12, ..., 348)
  TNO20            data/TNO (the 20 pairs used by DenseFuse)
"""
import os
import sys
import csv
import hashlib
import cv2

OUT = 'data_manifest'


def sorted_png(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.bmp')))


def subsets():
    tr = [f for f in sorted_png('tmp_msrs/train/ir')
          if os.path.exists(os.path.join('tmp_msrs/train/vi', f))][:200]
    te = sorted_png('tmp_msrs/test/ir')[::12][:30]
    return {
        'MSRS_train200':    ('tmp_msrs/train', tr),
        'MSRS_test30':      ('tmp_msrs/test', te),
        'TNO20':            ('data/TNO', sorted_png('data/TNO/ir')),
    }


def row(subset, mod, path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SystemExit('cannot read ' + path)
    h, w = a.shape[:2]
    c = 1 if a.ndim == 2 else a.shape[2]
    pix = hashlib.sha256(a.tobytes()).hexdigest()
    with open(path, 'rb') as f:
        fil = hashlib.sha256(f.read()).hexdigest()
    return [subset, mod, os.path.basename(path), h, w, c, pix, fil]


HDR = ['subset', 'modality', 'file', 'height', 'width', 'channels', 'sha256_pixels', 'sha256_file']


def write():
    os.makedirs(OUT, exist_ok=True)
    for name, (base, files) in subsets().items():
        rows = []
        for f in files:
            for mod in ('ir', 'vi'):
                rows.append(row(name, mod, os.path.join(base, mod, f)))
        with open(os.path.join(OUT, name + '.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(HDR)
            w.writerows(rows)
        print('%-17s %3d pairs -> %s' % (name, len(files), os.path.join(OUT, name + '.csv')))


def check():
    bases = {k: v[0] for k, v in {
        'MSRS_train200': ('tmp_msrs/train', 0), 'MSRS_test30': ('tmp_msrs/test', 0),
        'TNO20': ('data/TNO', 0)}.items()}
    ok_all = True
    for name, base in bases.items():
        fn = os.path.join(OUT, name + '.csv')
        rows = list(csv.DictReader(open(fn)))
        bad = missing = 0
        for r in rows:
            p = os.path.join(base, r['modality'], r['file'])
            if not os.path.exists(p):
                missing += 1
                continue
            if row(name, r['modality'], p)[6] != r['sha256_pixels']:
                bad += 1
        ok = bad == 0 and missing == 0
        ok_all &= ok
        print('%-17s %3d images  missing=%d  pixel-mismatch=%d  %s'
              % (name, len(rows), missing, bad, 'OK' if ok else 'FAIL'))
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    check() if '--check' in sys.argv else write()
