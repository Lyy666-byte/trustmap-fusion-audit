# -*- coding: utf-8 -*-
"""Run CDDFuse pretrained inference on an ir/vi folder, save fused PNGs."""
import os, sys, numpy as np, cv2, torch, torch.nn as nn
ROOT = 'sota/MMIF-CDDFuse'
sys.path.insert(0, ROOT)
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.img_read_save import image_read_cv2

ir_dir, vi_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
ckpt = os.path.join(ROOT, 'models', 'CDDFuse_IVF.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Enc = nn.DataParallel(Restormer_Encoder()).to(device)
Dec = nn.DataParallel(Restormer_Decoder()).to(device)
BaseF = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailF = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
sd = torch.load(ckpt, map_location=device)
Enc.load_state_dict(sd['DIDF_Encoder']); Dec.load_state_dict(sd['DIDF_Decoder'])
BaseF.load_state_dict(sd['BaseFuseLayer']); DetailF.load_state_dict(sd['DetailFuseLayer'])
for m in (Enc, Dec, BaseF, DetailF): m.eval()

with torch.no_grad():
    for name in os.listdir(os.path.join(ir_dir)):
        if not name.lower().endswith(('.png','.jpg','.bmp')): continue
        ir = image_read_cv2(os.path.join(ir_dir, name), 'GRAY')[np.newaxis, np.newaxis, ...] / 255.0
        vi = image_read_cv2(os.path.join(vi_dir, name), 'GRAY')[np.newaxis, np.newaxis, ...] / 255.0
        ir = torch.FloatTensor(ir).to(device); vi = torch.FloatTensor(vi).to(device)
        vB, vD, _ = Enc(vi); iB, iD, _ = Enc(ir)
        fB = BaseF(vB + iB); fD = DetailF(vD + iD)
        fused, _ = Dec(vi, fB, fD)
        fused = (fused - fused.min()) / (fused.max() - fused.min())
        fi = np.squeeze((fused * 255).cpu().numpy()).clip(0,255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, name.split('.')[0] + '.png'), fi)
print('CDDFuse done ->', out_dir)
