# -*- coding: utf-8 -*-
"""Architecture figure (Figure 1 of the Electronics manuscript), redrawn with print-size text.
Layers as in gen_arch_fig.py / build_fusion_net_v7.m (43,356 learnables):
  stem      grouped conv 3x3, 2 groups, 20 filters/group  -> 40 ch, ReLU
  rb1, rb2  grouped residual blocks (3x3, 2 groups, 20/group)
  ca        GAP -> conv1x1 40->10 -> ReLU -> conv1x1 10->40 -> sigmoid -> scale
  D         F_ir - F_vis;  trust head conv3x3 20->8, ReLU, conv3x3 dil. 2 8->8, ReLU, conv3x3 8->1, sigmoid
  fusion    Phi = F_vis + W * D;  refine residual block (3x3, 20);  decoder conv3x3 20->20, ReLU, conv3x3 20->1, sigmoid
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FS = 11.5                       # box text
fig, ax = plt.subplots(figsize=(12.5, 5.0))
ax.set_xlim(0, 29.3);ax.set_ylim(0, 11.2); ax.axis('off')
IRC, VISC, SH, ATT, FUSE, OUT, LOSS = '#f6d5d5', '#d6e4f7', '#e2eede', '#fbf0d9', '#e7dcf2', '#eaf2fb', '#f3dede'


def box(x, y, w, h, text, fc, fs=FS, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', linewidth=1.0, edgecolor='black',
                                facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, linespacing=1.25,
            fontweight=('bold' if bold else 'normal'))


def arr(x1, y1, x2, y2, ls='-', color='black', lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, linewidth=lw,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0))


yIR, yVI, hb = 7.4, 4.0, 1.35
box(0.1, yIR, 2.1, hb, 'Infrared\n$I_{ir}$', IRC)
box(0.1, yVI, 2.1, hb, 'Visible\n$I_{vis}$', VISC)
ax.add_patch(FancyBboxPatch((2.65, 3.3), 8.0, 5.85, boxstyle='round,pad=0.08', linewidth=1.0, edgecolor='#888',
                            facecolor='none', linestyle='--'))
ax.text(6.65, 9.45, 'Modality-specific encoder (grouped convolutions, 2 groups)', ha='center', fontsize=10.5,
        style='italic', color='#334')
for yy, cc in ((yIR, IRC), (yVI, VISC)):
    box(2.9, yy, 2.0, hb, 'GConv\n3×3, 20', cc)
    box(5.15, yy, 2.45, hb, 'GResBlock\n20', cc)
    box(7.85, yy, 2.45, hb, 'GResBlock\n20', cc)
    arr(2.2, yy + hb / 2, 2.9, yy + hb / 2); arr(4.9, yy + hb / 2, 5.15, yy + hb / 2); arr(7.6, yy + hb / 2, 7.85, yy + hb / 2)

box(11.0, 5.2, 2.6, 2.6, 'Cross-modal\nchannel\nattention\nGAP, 1×1, σ', ATT, 10.5)
arr(10.3, yIR + hb / 2, 11.0, 7.4); arr(10.3, yVI + hb / 2, 11.0, 5.6)

box(14.1, 7.4, 2.6, 1.2, '$D=F_{ir}-F_{vis}$', FUSE)
box(14.1, 4.0, 1.9, 1.2, '$F_{vis}$', VISC)
arr(13.6, 7.2, 14.1, 7.75); arr(13.6, 5.8, 14.6, 5.2)

box(17.6, 5.3, 3.6, 1.7, '$\\Phi=F_{vis}+W\\odot D$\nconvex fusion', FUSE)
arr(16.7, 8.0, 18.6, 7.0)                    # D -> fusion
arr(16.0, 4.6, 17.6, 5.6)                    # F_vis -> fusion

box(12.4, 0.55, 7.0, 2.05, 'Trust head\nconv3×3 (8) → conv3×3, dilation 2 (8)\n→ conv3×3 (1) → σ', ATT, 10.5)
ax.plot([16.35, 16.35], [7.4, 2.75], color='#444', lw=1.1, ls=':')
arr(16.35, 2.75, 16.35, 2.6, color='#444', ls=':')
box(20.1, 1.0, 1.5, 1.15, '$W$', ATT, 13, bold=True)
arr(19.4, 1.575, 20.1, 1.575)                # trust head -> W (starts at the box edge)
arr(20.85, 2.15, 19.9, 5.3)                  # W -> fusion

box(21.7, 5.45, 2.3, 1.4, 'Refine\nResBlock', SH)
box(24.4, 5.45, 3.05, 1.4, 'Decoder\n2× conv3×3, σ', SH, 10.5)
box(27.75, 5.45, 1.45, 1.4, 'Fused\n$F$', OUT)
arr(21.2, 6.15, 21.7, 6.15); arr(24.0, 6.15, 24.4, 6.15); arr(27.45, 6.15, 27.75, 6.15)

box(17.6, 9.3, 11.6, 1.3, 'Unsupervised loss  $L=L_{int}(S)+20L_{grad}+3(1-L_{ssim})+0.5L_{tv}$', LOSS, 11)
arr(28.47, 6.85, 28.47, 9.3, ls='--', lw=1.0)
ax.text(25.4, 4.3, '43,356 learnable parameters', ha='center', fontsize=10.5, style='italic', color='#444')

plt.tight_layout(pad=0.2)
plt.savefig('results/fig_arch_v2.png', dpi=300, bbox_inches='tight')
print('saved results/fig_arch_v2.png')
