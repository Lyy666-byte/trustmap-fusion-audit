# -*- coding: utf-8 -*-
"""MMDRFuse (ACM MM 2024) inference with the released student model (113 parameters).

    python sota_mmdrfuse_infer.py <ir_dir> <vi_dir> <out_dir>

Follows the official test_student.py (github.com/yanglinDeng/MMDRFuse, commit c9e5014):
grayscale inputs (PIL mode 'L') scaled to [0, 1], concatenated [ir, vi], student encoder/decoder,
tanh output saved with scipy.misc.imsave, which min-max rescales to 0..255 (bytescale);
the same rescaling is applied here explicitly (scipy.misc.imsave no longer exists).
"""
import os, sys
import numpy as np, torch
from PIL import Image

ROOT = 'sota/MMDRFuse'
sys.path.insert(0, ROOT)
from student_net import stu_net

ir_dir, vi_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = stu_net(2, 1)
net.load_state_dict(torch.load(os.path.join(ROOT, 'pretrained_models', 'student.model'), map_location=device))
net = net.to(device).eval()
print('MMDRFuse student parameters:', sum(p.numel() for p in net.parameters()))

n = 0
with torch.no_grad():
    for name in sorted(os.listdir(ir_dir)):
        if not name.lower().endswith(('.png', '.jpg', '.bmp')):
            continue
        ir = np.asarray(Image.open(os.path.join(ir_dir, name)).convert('L'), dtype=np.float64) / 255.0
        vi = np.asarray(Image.open(os.path.join(vi_dir, name)).convert('L'), dtype=np.float64) / 255.0
        x = torch.from_numpy(np.stack([ir, vi])[None]).float().to(device)
        out = net.decoder(net.encoder(x))[0][0, 0].cpu().numpy().astype(np.float64)
        cmin, cmax = out.min(), out.max()                       # scipy bytescale(cmin=min, cmax=max)
        b = ((out - cmin) * (255.0 / max(cmax - cmin, 1e-12))).clip(0, 255) + 0.5
        Image.fromarray(b.astype(np.uint8)).save(os.path.join(out_dir, os.path.splitext(name)[0] + '.png'))
        n += 1
print('MMDRFuse done ->', out_dir, '(%d images)' % n)
