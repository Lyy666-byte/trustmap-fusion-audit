# -*- coding: utf-8 -*-
"""FLOPs and peak GPU memory of the PyTorch competitors for one 640x480 pair.

Each model is instantiated exactly as in measure_sota_params.py (released code and weights).
FLOPs are counted with torch.utils.flop_counter.FlopCounterMode, which counts the convolution
and matrix-multiplication operators (aten conv / mm / bmm / addmm / baddbmm / sdpa), reported as
2 x multiply-accumulates; element-wise operations are not counted. The same convention is used
for the MATLAB networks in count_flops.m.

    python measure_sota_flops.py flops    # on the CPU (no timing involved)
    python measure_sota_flops.py memory   # peak allocated GPU memory of one forward pass

Writes / updates results/efficiency_flops_mem.csv (method, gflops, peak_mem_mb).
"""
import os, sys, csv
import torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

MODE = sys.argv[1] if len(sys.argv) > 1 else 'flops'
H, W = 480, 640
device = 'cpu' if MODE == 'flops' else 'cuda'
OUT = 'results/efficiency_flops_mem.csv'


def build():
    models = {}
    sys.path.insert(0, 'sota/SeAFusion')
    from FusionNet import FusionNet
    fm = FusionNet(output=1)
    fm.load_state_dict(torch.load('sota/SeAFusion/model/Fusion/fusionmodel_final.pth', map_location=device))
    fm = fm.to(device).eval()
    models['SeAFusion'] = lambda ir, vi: fm(vi, ir)

    sys.path.insert(0, 'sota/MMIF-CDDFuse')
    from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
    ck = torch.load('sota/MMIF-CDDFuse/models/CDDFuse_IVF.pth', map_location=device)
    Enc = nn.DataParallel(Restormer_Encoder()).to(device)
    Dec = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseF = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetF = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    Enc.load_state_dict(ck['DIDF_Encoder']); Dec.load_state_dict(ck['DIDF_Decoder'])
    BaseF.load_state_dict(ck['BaseFuseLayer']); DetF.load_state_dict(ck['DetailFuseLayer'])
    Enc, Dec, BaseF, DetF = (m.module.to(device).eval() for m in (Enc, Dec, BaseF, DetF))  # unwrap DataParallel

    def cdd(ir, vi):
        vB, vD, _ = Enc(vi); iB, iD, _ = Enc(ir)
        return Dec(vi, BaseF(vB + iB), DetF(vD + iD))
    models['CDDFuse'] = cdd

    sys.path.insert(0, 'sota/MMIF-EMMA')
    from nets.Ufuser import Ufuser
    em = Ufuser().to(device)
    em.load_state_dict(torch.load('sota/MMIF-EMMA/model/EMMA.pth', map_location=device))
    em.eval()
    models['EMMA'] = lambda ir, vi: em(ir, vi)

    sys.path.insert(0, 'sota/DCEvo')
    from sleepnet import DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor
    ck = torch.load('sota/DCEvo/ckpt/DCEvo_fusion.pth', map_location=device)
    Enc2 = nn.DataParallel(DE_Encoder()).to(device)
    Dec2 = nn.DataParallel(DE_Decoder()).to(device)
    LF = nn.DataParallel(LowFreqExtractor(dim=64)).to(device)
    HF = nn.DataParallel(HighFreqExtractor(num_layers=3)).to(device)
    Enc2.load_state_dict(ck['DE_Encoder']); Dec2.load_state_dict(ck['DE_Decoder'])
    LF.load_state_dict(ck['LowFreqExtractor']); HF.load_state_dict(ck['HighFreqExtractor'])
    Enc2, Dec2, LF, HF = (m.module.to(device).eval() for m in (Enc2, Dec2, LF, HF))

    def dce(ir, vi):
        vB, vD, _ = Enc2(vi); iB, iD, _ = Enc2(ir)
        return Dec2(vi, LF(vB + iB), HF(vD + iD))
    models['DCEvo'] = dce
    return models


rows = {}
if os.path.exists(OUT):
    with open(OUT, newline='') as f:
        rows = {r['method']: r for r in csv.DictReader(f)}

torch.manual_seed(0)
ir = torch.rand(1, 1, H, W, device=device)
vi = torch.rand(1, 1, H, W, device=device)
models = build()
for name, fn in models.items():
    r = rows.setdefault(name, {'method': name, 'gflops': '', 'peak_mem_mb': ''})
    with torch.no_grad():
        if MODE == 'flops':
            fc = FlopCounterMode(display=False)
            with fc:
                fn(ir, vi)
            r['gflops'] = '%.4f' % (fc.get_total_flops() / 1e9)
        else:
            fn(ir, vi)                                   # warm-up (cuDNN workspace selection)
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            fn(ir, vi)
            torch.cuda.synchronize()
            r['peak_mem_mb'] = '%.1f' % (torch.cuda.max_memory_allocated() / 2**20)
    print(name, r, flush=True)

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['method', 'gflops', 'peak_mem_mb'])
    w.writeheader()
    for r in rows.values():
        w.writerow({k: r.get(k, '') for k in ('method', 'gflops', 'peak_mem_mb')})
print('wrote', OUT)
