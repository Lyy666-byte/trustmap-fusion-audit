# -*- coding: utf-8 -*-
"""Measure the learnable-parameter count and 640x480 inference time of every
PyTorch competitor directly from its released code and weights, and append the
results to results/efficiency_timing.csv so Table 5 of the paper is read from a
stored artifact instead of being hard-coded.

Each model is instantiated exactly as its own inference script does.
"""
import os, sys, csv, time
import torch, torch.nn as nn

H, W, N = 480, 640, 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = 'results/efficiency_timing.csv'
rows = []


def nparams(*mods):
    return sum(p.numel() for m in mods for p in m.parameters())


def timeit(fn):
    for _ in range(5):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(N):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t) / N * 1000.0


ir = torch.rand(1, 1, H, W, device=device)
vi = torch.rand(1, 1, H, W, device=device)

# ---------------- SeAFusion ----------------
try:
    sys.path.insert(0, 'sota/SeAFusion')
    from FusionNet import FusionNet
    fm = FusionNet(output=1)
    fm.load_state_dict(torch.load('sota/SeAFusion/model/Fusion/fusionmodel_final.pth',
                                  map_location=device))
    fm = fm.to(device).eval()
    with torch.no_grad():
        f = lambda: fm(vi, ir)
        rows.append(['SeAFusion', nparams(fm), round(timeit(f), 2), 'PyTorch'])
    print('SeAFusion ok', rows[-1])
except Exception as e:
    print('SeAFusion FAILED:', e)

# ---------------- CDDFuse ----------------
try:
    sys.path.insert(0, 'sota/MMIF-CDDFuse')
    from net import (Restormer_Encoder, Restormer_Decoder,
                     BaseFeatureExtraction, DetailFeatureExtraction)
    ck = torch.load('sota/MMIF-CDDFuse/models/CDDFuse_IVF.pth', map_location=device)
    Enc = nn.DataParallel(Restormer_Encoder()).to(device)
    Dec = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseF = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetF = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    Enc.load_state_dict(ck['DIDF_Encoder']); Dec.load_state_dict(ck['DIDF_Decoder'])
    BaseF.load_state_dict(ck['BaseFuseLayer']); DetF.load_state_dict(ck['DetailFuseLayer'])
    for m in (Enc, Dec, BaseF, DetF):
        m.eval()

    def cdd():
        with torch.no_grad():
            vB, vD, _ = Enc(vi); iB, iD, _ = Enc(ir)
            Dec(vi, BaseF(vB + iB), DetF(vD + iD))
    rows.append(['CDDFuse', nparams(Enc, Dec, BaseF, DetF), round(timeit(cdd), 2), 'PyTorch'])
    print('CDDFuse ok', rows[-1])
except Exception as e:
    print('CDDFuse FAILED:', e)

# ---------------- EMMA ----------------
try:
    sys.path.insert(0, 'sota/MMIF-EMMA')
    from nets.Ufuser import Ufuser
    em = Ufuser().to(device)
    em.load_state_dict(torch.load('sota/MMIF-EMMA/model/EMMA.pth', map_location=device))
    em.eval()
    with torch.no_grad():
        f = lambda: em(ir, vi)
        rows.append(['EMMA', nparams(em), round(timeit(f), 2), 'PyTorch'])
    print('EMMA ok', rows[-1])
except Exception as e:
    print('EMMA FAILED:', e)
    try:
        rows.append(['EMMA', nparams(em), '', 'PyTorch (params only)'])
        print('EMMA params only', rows[-1])
    except Exception:
        pass

# ---------------- DCEvo ----------------
try:
    sys.path.insert(0, 'sota/DCEvo')
    from sleepnet import DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor
    ck = torch.load('sota/DCEvo/ckpt/DCEvo_fusion.pth', map_location=device)
    Enc2 = nn.DataParallel(DE_Encoder()).to(device)
    Dec2 = nn.DataParallel(DE_Decoder()).to(device)
    LF = nn.DataParallel(LowFreqExtractor(dim=64)).to(device)
    HF = nn.DataParallel(HighFreqExtractor(num_layers=3)).to(device)
    Enc2.load_state_dict(ck['DE_Encoder']); Dec2.load_state_dict(ck['DE_Decoder'])
    LF.load_state_dict(ck['LowFreqExtractor']); HF.load_state_dict(ck['HighFreqExtractor'])
    for m in (Enc2, Dec2, LF, HF):
        m.eval()

    def dce():
        with torch.no_grad():
            vB, vD, _ = Enc2(vi); iB, iD, _ = Enc2(ir)
            Dec2(vi, LF(vB + iB), HF(vD + iD))
    rows.append(['DCEvo', nparams(Enc2, Dec2, LF, HF), round(timeit(dce), 2), 'PyTorch'])
    print('DCEvo ok', rows[-1])
except Exception as e:
    print('DCEvo FAILED:', e)
    try:
        rows.append(['DCEvo', nparams(Enc2, Dec2, LF, HF), '', 'PyTorch (params only)'])
        print('DCEvo params only', rows[-1])
    except Exception:
        pass

# ---------------- merge with any existing rows (MATLAB side writes ours/DenseFuse)
existing = []
if os.path.exists(OUT):
    with open(OUT, newline='') as f:
        existing = [r for r in csv.DictReader(f)]
keep = [r for r in existing if r['method'] not in {x[0] for x in rows}]
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['method', 'params', 'time_ms', 'framework'])
    for r in keep:
        w.writerow([r['method'], r['params'], r['time_ms'], r['framework']])
    for r in rows:
        w.writerow(r)
print('\nwrote', OUT)
for r in rows:
    print('  %-10s %10d params  %8s ms' % (r[0], r[1], r[2]))
