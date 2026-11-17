# -*- coding: utf-8 -*-
"""Run DCEvo (CVPR 2025) pretrained fusion inference on an ir/vi folder, save PNGs.
Uses the repo's sleepnet (timm + einops, no kornia); native resolution with
reflect-padding to a multiple of 16. Usage:
  python sota_dcevo_infer.py <ir_dir> <vi_dir> <out_dir>
"""
import os, sys, numpy as np, cv2, torch, torch.nn as nn
ROOT = 'sota/DCEvo'
sys.path.insert(0, ROOT)
from sleepnet import DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor

ir_dir, vi_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(os.path.join(ROOT, 'ckpt', 'DCEvo_fusion.pth'), map_location=device)

Enc = nn.DataParallel(DE_Encoder()).to(device)
Dec = nn.DataParallel(DE_Decoder()).to(device)
LF  = nn.DataParallel(LowFreqExtractor(dim=64)).to(device)
HF  = nn.DataParallel(HighFreqExtractor(num_layers=3)).to(device)
Enc.load_state_dict(ckpt['DE_Encoder']); Dec.load_state_dict(ckpt['DE_Decoder'])
LF.load_state_dict(ckpt['LowFreqExtractor']); HF.load_state_dict(ckpt['HighFreqExtractor'])
for m in (Enc, Dec, LF, HF): m.eval()

def rd(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

with torch.no_grad():
    n = 0
    for name in os.listdir(ir_dir):
        if not name.lower().endswith(('.png', '.jpg', '.bmp')):
            continue
        ir = rd(os.path.join(ir_dir, name)); vi = rd(os.path.join(vi_dir, name))
        h, w = ir.shape
        ph = (16 - h % 16) % 16; pw = (16 - w % 16) % 16
        irp = np.pad(ir, ((0, ph), (0, pw)), mode='reflect')
        vip = np.pad(vi, ((0, ph), (0, pw)), mode='reflect')
        tI = torch.FloatTensor(irp)[None, None].to(device)
        tV = torch.FloatTensor(vip)[None, None].to(device)
        vB, vD, _ = Enc(tV); iB, iD, _ = Enc(tI)
        fB = LF(iB + vB); fD = HF(iD + vD)
        fuse, _ = Dec(tV * 0.5 + tI * 0.5, fB, fD)
        fi = np.squeeze(fuse.cpu().numpy())[:h, :w]
        fi = (fi - fi.min()) / (fi.max() - fi.min() + 1e-8)
        cv2.imwrite(os.path.join(out_dir, name.split('.')[0] + '.png'),
                    (fi * 255).clip(0, 255).astype(np.uint8))
        n += 1
print('DCEvo done -> %s (%d images)' % (out_dir, n))
