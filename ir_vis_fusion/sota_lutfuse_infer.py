# -*- coding: utf-8 -*-
"""LUT-Fuse (ICCV 2025) inference with the released checkpoints.

    python sota_lutfuse_infer.py <ir_dir> <vi_dir> <out_dir>

Follows the official test_lut.py (github.com/zyb5/LUT-Fuse, commit ab5878e): visible image as RGB,
infrared as L, both ToTensor; ckpts/fine_tuned_lut.npy + ckpts/generator_context.pth;
apply_fusion_4d_with_interpolation(vis*255, ir*255, lut, context generator); output clamped to [0, 1]
and saved as RGB. One deviation: the official script resizes the inputs to a multiple of 16
(T.Resize_16). The context generator is fully convolutional with stride 1, so this is not required;
we process every image at its original size, which leaves the 640x480 MSRS images exactly as in the
official script and avoids an interpolation (and a resize back) for the TNO and RoadScene images.
"""
import os, sys
import numpy as np, torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = os.path.abspath('sota/LUT-Fuse')
sys.path.insert(0, ROOT)
from scripts.calculate import load_lookup_table, Generator_for_info, apply_fusion_4d_with_interpolation

ir_dir, vi_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
lut = load_lookup_table(os.path.join(ROOT, 'ckpts', 'fine_tuned_lut.npy')).to('cuda')
gen = Generator_for_info()
gen.load_state_dict(torch.load(os.path.join(ROOT, 'ckpts', 'generator_context.pth'), map_location='cuda'))
gen = gen.to('cuda').eval()
print('LUT entries:', lut.numel(), ' context-generator parameters:', sum(p.numel() for p in gen.parameters()))

n = 0
with torch.no_grad():
    for name in sorted(os.listdir(ir_dir)):
        if not name.lower().endswith(('.png', '.jpg', '.bmp')):
            continue
        vi = TF.to_tensor(Image.open(os.path.join(vi_dir, name)).convert('RGB'))[None].cuda()
        ir = TF.to_tensor(Image.open(os.path.join(ir_dir, name)).convert('L'))[None].cuda()
        out = apply_fusion_4d_with_interpolation(vi * 255., ir * 255., lut, gen)
        res = out.squeeze(0).clamp(0, 1).cpu()
        TF.to_pil_image(res).save(os.path.join(out_dir, os.path.splitext(name)[0] + '.png'))
        n += 1
print('LUT-Fuse done ->', out_dir, '(%d images)' % n)
