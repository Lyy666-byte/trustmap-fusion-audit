# -*- coding: utf-8 -*-
"""Parameters, FLOPs and 640x480 inference time of the two dedicated lightweight networks,
from their released code and weights (same conventions as measure_sota_params.py and
measure_sota_flops.py).

    python measure_light_efficiency.py flops    # FLOPs -> results/efficiency_flops_mem.csv
    python measure_light_efficiency.py time     # parameters + time -> results/efficiency_timing.csv
                                                # (run with the GPU otherwise idle)

MMDRFuse: student network (2 convolutions). LUT-Fuse: a 4-D look-up table (16^4 = 65,536 learnable
entries, read by interpolation) plus a context-generator CNN. FlopCounterMode counts the
convolutions only, i.e. the table look-up and interpolation of LUT-Fuse are not counted; its
learnable values are reported as generator parameters + table entries.
"""
import os, sys, csv, time
import torch
from torch.utils.flop_counter import FlopCounterMode

MODE = sys.argv[1] if len(sys.argv) > 1 else 'flops'
H, W, N = 480, 640, 50
dev = 'cuda'

sys.path.insert(0, 'sota/MMDRFuse')
from student_net import stu_net
mm = stu_net(2, 1)
mm.load_state_dict(torch.load('sota/MMDRFuse/pretrained_models/student.model', map_location=dev))
mm = mm.to(dev).eval()

sys.path.insert(0, os.path.abspath('sota/LUT-Fuse'))
from scripts.calculate import load_lookup_table, Generator_for_info, apply_fusion_4d_with_interpolation
lut = load_lookup_table('sota/LUT-Fuse/ckpts/fine_tuned_lut.npy').to(dev)
gen = Generator_for_info()
gen.load_state_dict(torch.load('sota/LUT-Fuse/ckpts/generator_context.pth', map_location=dev))
gen = gen.to(dev).eval()

torch.manual_seed(0)
ir1 = torch.rand(1, 1, H, W, device=dev)
vi1 = torch.rand(1, 1, H, W, device=dev)
vi3 = torch.rand(1, 3, H, W, device=dev)
fns = {'MMDRFuse': lambda: mm.decoder(mm.encoder(torch.cat([ir1, vi1], 1))),
       'LUT-Fuse': lambda: apply_fusion_4d_with_interpolation(vi3 * 255., ir1 * 255., lut, gen)}
npar = {'MMDRFuse': sum(p.numel() for p in mm.parameters()),
        'LUT-Fuse': sum(p.numel() for p in gen.parameters()) + lut.numel()}

if MODE == 'flops':
    OUT = 'results/efficiency_flops_mem.csv'
    rows = {r['method']: r for r in csv.DictReader(open(OUT))}
    for name, fn in fns.items():
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                fn()
        rows[name] = {'method': name, 'gflops': '%.4f' % (fc.get_total_flops() / 1e9), 'peak_mem_mb': ''}
        print(name, rows[name])
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'gflops', 'peak_mem_mb'])
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, '') for k in ('method', 'gflops', 'peak_mem_mb')})
else:
    OUT = 'results/efficiency_timing.csv'
    rows = list(csv.DictReader(open(OUT)))
    rows = [r for r in rows if r['method'] not in fns]
    for name, fn in fns.items():
        with torch.no_grad():
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(N):
                fn()
            torch.cuda.synchronize()
            ms = (time.time() - t) / N * 1000.0
        rows.append({'method': name, 'params': str(npar[name]), 'time_ms': '%.2f' % ms, 'framework': 'PyTorch'})
        print(name, npar[name], 'params', '%.2f ms' % ms)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'params', 'time_ms', 'framework'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
print('wrote', OUT)
