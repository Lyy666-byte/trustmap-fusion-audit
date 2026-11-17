# -*- coding: utf-8 -*-
"""Regenerate Tables 1-8 of the manuscript from the result files in this repository.

    python make_tables.py            -> tables/Table1.csv ... tables/Table8.csv and tables/tables.md

Only the Python standard library is needed. Values are printed with the precision of the paper; in tables.md the
entries that are bold in the paper are marked **bold** (highest value of a column at the printed precision, lowest
for Nabf, ties all bold).
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'ir_vis_fusion', 'results')        # results of the fusion network and of the comparison
MS = os.path.join(HERE, 'metric_sanity', 'results')       # results of the metric audit and of the segmentation
OUT = os.path.join(HERE, 'tables')

METS = ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'MS_SSIM', 'CC', 'Qabf', 'FMI', 'Nabf', 'SCD', 'VIF']
DISP = {'MS_SSIM': 'MEF-SSIM', 'Nabf': 'Nabf↓'}          # MS_SSIM in the result files is MEF-SSIM (see README)
LOWER = {'Nabf'}
ORDER = ['average', 'laplacian', 'dwt', 'gff', 'DenseFuse', 'SeAFusion', 'CDDFuse', 'EMMA', 'DCEvo', 'MMDRFuse',
         'LUTFuse', 'Ours']
NAME = {'average': 'Pixel average', 'laplacian': 'Laplacian', 'dwt': 'Wavelet', 'gff': 'GFF', 'LUTFuse': 'LUT-Fuse',
        'avg': 'Pixel average', 'vi': 'Visible', 'ir': 'Infrared'}
TRANS = ['blend25', 'blend50', 'blend75', 'noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast',
         'blur1', 'avg']
TNAME = {'blend25': 'Blend, a = 0.25', 'blend50': 'Blend, a = 0.5', 'blend75': 'Blend, a = 0.75',
         'noise1': 'Noise, sigma = 0.01', 'noise3': 'Noise, sigma = 0.03', 'sharp1': 'Unsharp, k = 1',
         'sharp2': 'Unsharp, k = 2', 'gamma07': 'Gamma 0.7', 'gamma14': 'Gamma 1.4', 'contrast': 'Contrast x1.25',
         'blur1': 'Blur, sigma = 1', 'avg': 'Pixel average'}


def rd(*p):
    with open(os.path.join(*p), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fmt(m, v):
    return f'{v:.4f}' if m in ('SF', 'AG') else f'{v:.3f}'


def bold_set(m, vals, fmt_=fmt):
    shown = {k: fmt_(m, v) for k, v in vals.items()}
    best = (min if m in LOWER else max)(float(s) for s in shown.values())
    return {k for k, s in shown.items() if float(s) == best}


TABLES = []                                                  # (number, title, header rows, body rows, note)


def add(no, title, header, rows, note=''):
    TABLES.append((no, title, header, rows, note))


def b(s, on):
    return f'**{s}**' if on else s


# ---------------------------------------------------------------- Tables 1 and 2: metric means
def table_means(no, tag, title, note=''):
    rows = {r['Row']: r for r in rd(R, f'metrics_sota_{tag}.csv')}
    data = {k: {m: float(rows[k][m]) for m in METS} for k in ORDER}
    bold = {m: bold_set(m, {k: data[k][m] for k in ORDER}) for m in METS}
    body = [[NAME.get(k, k)] + [b(fmt(m, data[k][m]), k in bold[m]) for m in METS] for k in ORDER]
    add(no, title, [['Method'] + [DISP.get(m, m) for m in METS]], body,
        'Bold = highest value in each column (lowest for columns marked ↓).' + note)


table_means(1, 'MSRS', 'Quantitative comparison on MSRS (30 test pairs).',
            ' MMDRFuse and LUT-Fuse are lightweight networks; DenseFuse is our MATLAB re-implementation, and the other '
            'deep methods use their released weights.')
table_means(2, 'TNO', 'Quantitative comparison on TNO (20 pairs); all networks are used as trained, ours on MSRS.')

# ---------------------------------------------------------------- Table 3: ablation
with open(os.path.join(R, 'ablation2_MSRS.csv'), newline='') as f:
    ab = list(csv.reader(f))
abn = ab[0][1:]
abd = {r[0]: {abn[i]: float(r[i + 1]) for i in range(len(abn))} for r in ab[1:]}
npar = {r['config']: int(r['params']) for r in rd(R, 'ablation2_params.csv')}
dev = {r['config']: r['trust_dev'] for r in rd(R, 'ablation2_trustdev.csv')}
labs = ['A0_single', 'A1_dual_cat', 'A2_cat_trust', 'A3_diff_trust']
ldisp = {'A0_single': 'A0: single stream', 'A1_dual_cat': 'A1: dual, no trust map',
         'A2_cat_trust': 'A2: trust from concatenation', 'A3_diff_trust': 'A3: trust from difference (full)'}
sel = ['SSIM', 'Qabf', 'Nabf', 'MI', 'VIF']
bold = {m: bold_set(m, {l: abd[l][m] for l in labs}) for m in sel}
body = []
for l in labs:
    dv = dev.get(l, 'NaN')
    body.append([ldisp[l], f'{npar[l]:,}'] + [b(fmt(m, abd[l][m]), l in bold[m]) for m in sel]
                + ['—' if dv in ('NaN', 'nan', '') else f'{float(dv):.3f}'])
add(3, 'Ablation on MSRS (identical data, loss and 40 epochs for every configuration).',
    [['Configuration', 'Parameters'] + [DISP.get(m, m) for m in sel] + ['Trust dev.']], body,
    'Bold = highest value in each metric column (lowest for columns marked ↓). Trust dev. = ||W - mean(W)||/||W||; '
    '— = no trust map.')

# ---------------------------------------------------------------- Table 4: efficiency
meas = {r['method']: (int(r['params']), float(r['time_ms']), r['framework']) for r in rd(R, 'efficiency_timing.csv')}
gfl = {r['method']: float(r['gflops']) for r in rd(R, 'efficiency_flops_mem.csv')}
body = []
for m in ['DenseFuse', 'SeAFusion', 'CDDFuse', 'EMMA', 'DCEvo', 'MMDRFuse', 'LUT-Fuse', 'Ours']:
    pc, ms, fw = meas[m]; g = gfl[m]
    body.append([m, f'{pc:,}' + ('ᵃ' if m == 'LUT-Fuse' else ''),
                 f'{g:.2f}' if g < 1 else f'{g:.1f}' if g < 1000 else f'{g:.0f}',
                 f'{ms:.2f}' if ms < 1 else f'{ms:.1f}', fw])
add(4, 'Parameters, operations and run time for one 640×480 pair (RTX 4060 Ti, mean of 50 runs).',
    [['Method', 'Parameters', 'GFLOPs', 'Time (ms)', 'Framework']], body,
    'ᵃ7825 network parameters plus a 65,536-entry look-up table.')

# ---------------------------------------------------------------- Table 5: training seeds
sr = rd(R, 'seeds5_metrics.csv')
a3 = [r for r in sr if r['config'] == 'A3' and r['seed'].isdigit()]
st = {(r['config'], r['seed']): r for r in sr if not r['seed'].isdigit()}
assert len(a3) == 5
xf = (('trust_dev', '%.3f'), ('loss_rise_pct', '%.1f'), ('person_ratio', '%.2f'))
body = [['A3', r['seed']] + [f'{float(r[m]):.3f}' for m in sel] + [f % float(r[k]) for k, f in xf] for r in a3]
for cfg in ('A3', 'A1'):
    for stat in ('mean', 'std'):
        r = st[cfg, stat]
        body.append([cfg, stat] + [f'{float(r[m]):.3f}' for m in sel]
                    + [(f % float(r[k])) if cfg == 'A3' else '—' for k, f in xf])
add(5, 'Five training seeds of the full model (A3) on MSRS, with the mean and standard deviation of A3 and of the '
       'trust-free model A1.',
    [['Model', 'Seed'] + [DISP.get(m, m) for m in sel] + ['Trust dev.', 'Loss rise (%)', 'Person/bg']], body,
    'Loss rise = increase of the unsupervised loss when W is replaced by its spatial mean; Person/bg = mean of W on '
    'person pixels over its mean on unlabeled pixels; std = standard deviation over the five seeds.')

# ---------------------------------------------------------------- Table 6: direction of change
look = {(x['dataset'], x['transform'], x['metric']): x['verdict'] for x in rd(MS, 'cause6_metrics.csv')}
look.update({('FMB', x['transform'], x['metric']): x['verdict'] for x in rd(MS, 'fmb_metrics.csv')})
sym = {'up': '↑', 'down': '↓', 'ns': '·'}
body = [[TNAME[T]] + [''.join(sym[look[d, T, m]] for d in ('MSRS', 'FMB')) for m in METS] for T in TRANS]
add(6, 'Direction of the change of each metric under each transformation, six methods averaged per image; in each '
       'cell MSRS and FMB, in this order.',
    [['Transformation'] + [DISP.get(m, m).rstrip('↓') for m in METS]], body,
    '↑ / ↓ = significant rise / fall (Wilcoxon signed-rank, Holm over the 13 metrics, p < 0.05); · = no significant '
    'change. For Nabf the sign is reversed (↑ = lower artifact rate). MSRS: 100 images, FMB: 100.')

# ---------------------------------------------------------------- Table 7: rank attacks
RS = {(x['dataset'], x['metric']): x for x in rd(MS, 'rank_summary.csv')}
RC = [x for x in rd(MS, 'retrain_cond_rank.csv')
      if x['transform'] not in ('blend25', 'blend50', 'blend75') and x['counted'] == 'True']
FS = {x['metric']: x for x in rd(MS, 'fmb_rank_summary.csv')}
yes = lambda r: 'yes' if int(r['last_best_rank']) == 1 else 'no'
body = []
for m in METS:
    rc = [x for x in RC if x['metric'] == m]
    a, f = RS['MSRS', m], FS[m]
    assert int(a['counted']) == len(rc), m
    vals = [a['counted'], sum(x['harmful_retrained'] == 'True' for x in rc),
            sum(x['task_rises_retrained'] == 'True' for x in rc), yes(a),
            f['counted'], f['harmful_retrained'], f['rises_retrained'], yes(f)]
    body.append([DISP.get(m, m)] + [b(str(v), j in (0, 4) and int(v) > 0) for j, v in enumerate(vals)])
add(7, 'Rank attacks: of 48 attacks per metric and dataset, the number that move the method up by at least one '
       'place, and how the method\'s own segmentation changes.',
    [['Metric', 'MSRS', '', '', '', 'FMB', '', '', ''],
     ['', 'Counted', 'Falls', 'Rises', 'Last → 1st', 'Counted', 'Falls', 'Rises', 'Last → 1st']], body,
    'Counted = rank gain with a 95% bootstrap interval at or above one place (5000 resamples over images). Falls and '
    'Rises = counted attacks after which the mIoU3 of the attacked method, measured with the SegFormer-B5 retrained '
    'on its fused images (seed 0), falls or rises significantly.')

# ---------------------------------------------------------------- Table 8: retrained segmentation
msc = {(x['score'], x['input']): x for x in rd(MS, 'retrain_scores.csv') if x['scope'] == 'all'}
mvs = {(x['score'], x['method']): x for x in rd(MS, 'retrain_vs_avg.csv') if x['scope'] == 'all' and x['reference'] == 'avg'}
fsc = {(x['score'], x['input']): x for x in rd(MS, 'fmb_retrain_scores.csv')}
fvs = {(x['score'], x['input']): x for x in rd(MS, 'fmb_retrain_vs.csv') if x['reference'] == 'avg'}
ours = {(x['dataset'], x['score']): x for x in rd(MS, 'ours_retrain.csv') if x['scope'] == 'all' and x['reference'] == 'avg'}
cols = [('MSRS', 'mIoU3', msc, mvs), ('MSRS', 'mIoU9', msc, mvs), ('FMB', 'mIoU3', fsc, fvs), ('FMB', 'mIoU13', fsc, fvs)]
inputs = ['avg', 'vi', 'ir', 'SeAFusion', 'CDDFuse', 'EMMA', 'DCEvo', 'MMDRFuse', 'LUTFuse', 'Ours']
score = {}
for k, (d, s, sc, vs) in enumerate(cols):
    score[k] = {i: float(ours[d, s]['ours']) if i == 'Ours' else float(sc[s, i]['mean']) for i in inputs}
best = {k: max(v, key=v.get) for k, v in score.items()}
body = []
for i in inputs:
    row = [NAME.get(i, i)]
    for k, (d, s, sc, vs) in enumerate(cols):
        if i == 'Ours':
            delta, mark = float(ours[d, s]['delta']), ''
        elif i in ('avg', 'vi', 'ir'):
            delta, mark = (None if i == 'avg' else score[k][i] - score[k]['avg']), ''
        else:
            delta, mark = float(vs[s, i]['delta']), ('*' if float(vs[s, i]['p_holm']) < .05 else '')
        row += [b(f'{score[k][i]:.4f}', best[k] == i), '—' if delta is None else f'{delta:+.4f}' + mark]
    body.append(row)
add(8, 'Segmentation by SegFormer-B5 retrained on each input (mean of three seeds) and the difference from the '
       'pixel average.',
    [['Input', 'MSRS, mIoU3', '', 'MSRS, mIoU9', '', 'FMB, mIoU3', '', 'FMB, mIoU13', ''],
     [''] + ['Score', 'Δ vs. avg.'] * 4], body,
    'Bold = highest score in each column. * = significant after Holm correction over the six methods (paired '
    'bootstrap over images, 10,000 resamples). Our network is not part of the Holm family (its intervals are in '
    'metric_sanity/results/ours_retrain.csv); for the visible and infrared inputs Δ is the plain difference.')

# ---------------------------------------------------------------- write
os.makedirs(OUT, exist_ok=True)
md = ['# Tables 1-8, regenerated by make_tables.py from the result files', '']
for no, title, header, rows, note in TABLES:
    with open(os.path.join(OUT, f'Table{no}.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        for h in header:
            w.writerow(h)
        for r in rows:
            w.writerow([c.replace('**', '') for c in r])
    md += [f'## Table {no}. {title}', '']
    hdr = [' / '.join(x for x in col if x) for col in zip(*header)] if len(header) > 1 else header[0]
    if len(header) > 1:                                      # carry group labels over the merged cells
        grp, last = [], ''
        for g in header[0]:
            last = g or last
            grp.append(last)
        hdr = [header[0][0]] + [f'{g} {s}'.strip() for g, s in zip(grp[1:], header[1][1:])]
    md += ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    md += ['| ' + ' | '.join(r) + ' |' for r in rows]
    md += ['', note, '']
with open(os.path.join(OUT, 'tables.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))
print('wrote', len(TABLES), 'tables to', OUT)
