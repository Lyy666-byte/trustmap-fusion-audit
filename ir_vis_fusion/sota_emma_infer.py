# -*- coding: utf-8 -*-
"""Run EMMA (CVPR 2024) pretrained inference on an ir/vi folder, save fused PNGs.
Avoids EMMA's utils.py (which imports kornia); uses cv2 I/O and reflect-padding
to a multiple of 32 so any image size works. Usage:
  python sota_emma_infer.py <ir_dir> <vi_dir> <out_dir>
"""
import os, sys, numpy as np, cv2, torch
ROOT = 'sota/MMIF-EMMA'
sys.path.insert(0, ROOT)
from nets.Ufuser import Ufuser

ir_dir, vi_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Ufuser().to(device)
model.load_state_dict(torch.load(os.path.join(ROOT, 'model', 'EMMA.pth'), map_location=device))
model.eval()

def rd(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

with torch.no_grad():
    n = 0
    for name in os.listdir(ir_dir):
        if not name.lower().endswith(('.png', '.jpg', '.bmp')):
            continue
        ir = rd(os.path.join(ir_dir, name)); vi = rd(os.path.join(vi_dir, name))
        h, w = ir.shape
        ph = (32 - h % 32) % 32; pw = (32 - w % 32) % 32
        irp = np.pad(ir, ((0, ph), (0, pw)), mode='reflect')
        vip = np.pad(vi, ((0, ph), (0, pw)), mode='reflect')
        t_ir = torch.FloatTensor(irp)[None, None].to(device)
        t_vi = torch.FloatTensor(vip)[None, None].to(device)
        f = model(t_ir, t_vi)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        fi = np.squeeze(f.cpu().numpy())[:h, :w]
        cv2.imwrite(os.path.join(out_dir, name.split('.')[0] + '.png'),
                    (fi * 255).clip(0, 255).astype(np.uint8))
        n += 1
print('EMMA done -> %s (%d images)' % (out_dir, n))
