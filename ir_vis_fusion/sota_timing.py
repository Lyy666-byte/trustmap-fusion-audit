# -*- coding: utf-8 -*-
"""Measure inference time of SeAFusion and CDDFuse on 480x640 (GPU)."""
import os, sys, time, numpy as np, torch, torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
H, W = 480, 640; N = 50

def timeit(fn):
    for _ in range(5): fn()
    if device == 'cuda': torch.cuda.synchronize()
    t = time.time()
    for _ in range(N): fn()
    if device == 'cuda': torch.cuda.synchronize()
    return (time.time() - t) / N * 1000

# ---- CDDFuse ----
sys.path.insert(0, 'sota/MMIF-CDDFuse')
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
ck = 'sota/MMIF-CDDFuse/models/CDDFuse_IVF.pth'
Enc = nn.DataParallel(Restormer_Encoder()).to(device); Dec = nn.DataParallel(Restormer_Decoder()).to(device)
BaseF = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailF = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
sd = torch.load(ck, map_location=device)
Enc.load_state_dict(sd['DIDF_Encoder']); Dec.load_state_dict(sd['DIDF_Decoder'])
BaseF.load_state_dict(sd['BaseFuseLayer']); DetailF.load_state_dict(sd['DetailFuseLayer'])
for m in (Enc, Dec, BaseF, DetailF): m.eval()
ir = torch.rand(1,1,H,W).to(device); vi = torch.rand(1,1,H,W).to(device)
def cdd():
    with torch.no_grad():
        vB,vD,_ = Enc(vi); iB,iD,_ = Enc(ir)
        fB = BaseF(vB+iB); fD = DetailF(vD+iD); Dec(vi, fB, fD)
nparam_cdd = sum(p.numel() for p in list(Enc.parameters())+list(Dec.parameters())+list(BaseF.parameters())+list(DetailF.parameters()))
print('CDDFuse_infer_ms = %.2f' % timeit(cdd))
print('CDDFuse_params = %d' % nparam_cdd)

# ---- SeAFusion ----
sys.path.insert(0, 'sota/SeAFusion')
from FusionNet import FusionNet
fm = FusionNet(output=1); fm.load_state_dict(torch.load('sota/SeAFusion/model/Fusion/fusionmodel_final.pth', map_location=device))
fm = fm.to(device).eval()
viY = torch.rand(1,1,H,W).to(device); ir2 = torch.rand(1,1,H,W).to(device)
def sea():
    with torch.no_grad(): fm(viY, ir2)
print('SeAFusion_infer_ms = %.2f' % timeit(sea))
print('SeAFusion_params = %d' % sum(p.numel() for p in fm.parameters()))
