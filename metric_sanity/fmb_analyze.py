# -*- coding: utf-8 -*-
"""FMB, every analysis fixed in ANALYSIS_PLAN.md (addendum on FMB). Statistics as cause_analyze6.py, rank_attack.py, retrain_analyze.py.
Writes results/fmb_seg_holm.csv, fmb_class.csv, fmb_metrics.csv, fmb_rank_attack.csv, fmb_rank_summary.csv, fmb_rank_native.csv,
fmb_retrain_scores.csv, fmb_retrain_vs.csv, fmb_retrain_cond.csv, fmb_replication.csv."""
import csv, os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
ATTACK = ['noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
BLENDS = ['blend25', 'blend50', 'blend75']
TRANS = BLENDS + ATTACK
METS = ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'MS_SSIM', 'CC', 'Qabf', 'FMI', 'Nabf', 'SCD', 'VIF']
MODELS = ['segformer', 'mask2former']
CLASSES = ['person', 'car', 'bike']
INPUTS = METHODS + ['avg', 'vi', 'ir']
SEEDS = [0, 1, 2]
rng = np.random.default_rng(20261102)
BP, BRANK = 10000, 5000


def save(name, rows):
    with open(os.path.join(R, name), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (('%.5f' % v) if isinstance(v, float) else v) for k, v in r.items()})


def holm(ps):
    order = np.argsort(ps); run = 0; out = [0.] * len(ps)
    for rank, i in enumerate(order):
        run = max(run, min(1., ps[i] * (len(ps) - rank))); out[i] = run
    return out


def twoside(d):
    return min(1., 2 * min(np.mean(d <= 0), np.mean(d >= 0)) + 1 / len(d))


def piou(x):
    I, P, G = x[0].sum(0), x[1].sum(0), x[2].sum(0)
    return I / np.maximum(P + G - I, 1)


def biou(x):
    I, P, G = W @ x[0], W @ x[1], W @ x[2]
    return I / np.maximum(P + G - I, 1)


def load(model, key):
    d = np.load(os.path.join(R, 'seg_fmb', model, key + '.npz'))
    return d['inter'].astype(float), d['pred'].astype(float), d['gt'].astype(float), d['names']


names = load('segformer', 'base_avg')[3]; N = len(names)
W = rng.multinomial(N, np.ones(N) / N, size=BP).astype(float)

# ---------- pretrained segmentation ----------
seg_method = {}
rows_holm, rows_class = [], []
print('== Pretrained models: mIoU3 per input')
for model in MODELS:
    nat = {m: load(model, m + '_native') for m in METHODS}
    avg, vi, ir = load(model, 'base_avg'), load(model, 'base_vi'), load(model, 'base_ir')
    natb = {m: biou(nat[m]) for m in METHODS}; avgb = biou(avg); vib = biou(vi)
    print(' %-11s ' % model + ' '.join('%s %.4f' % (m, piou(nat[m]).mean()) for m in METHODS) + ' avg %.4f vi %.4f ir %.4f' % (piou(avg).mean(), piou(vi).mean(), piou(ir).mean()))
    pooled = []
    for T in TRANS + ['avg']:
        cond = {m: (avg if T == 'avg' else load(model, m + '_' + T)) for m in METHODS}
        condb = {m: (avgb if T == 'avg' else biou(cond[m])) for m in METHODS}
        for m in METHODS:
            dm = (condb[m] - natb[m]).mean(1)
            seg_method[model, m, T] = ((piou(cond[m]) - piou(nat[m])).mean(), *np.percentile(dm, [2.5, 97.5]))
        dc = np.mean([condb[m] - natb[m] for m in METHODS], 0)
        pt = np.mean([piou(cond[m]) - piou(nat[m]) for m in METHODS], 0)
        for c, cn in enumerate(CLASSES):
            lo, hi = np.percentile(dc[:, c], [2.5, 97.5])
            rows_class.append(dict(model=model, transform=T, cls=cn, delta_iou=float(pt[c]), lo=float(lo), hi=float(hi)))
        dd = dc.mean(1)
        pooled.append(dict(model=model, family='transform', item=T, delta=float(pt.mean()), lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5)), p=twoside(dd)))
    for fam, ref, refb in (('native_minus_avg', avg, avgb), ('native_minus_vi', vi, vib)):
        sub = []
        for m in METHODS:
            dd = (natb[m] - refb).mean(1)
            sub.append(dict(model=model, family=fam, item=m, delta=float((piou(nat[m]) - piou(ref)).mean()), lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5)), p=twoside(dd)))
        for r, h in zip(sub, holm([r['p'] for r in sub])):
            r['p_holm'] = h
        rows_holm += sub
    for r, h in zip(pooled, holm([r['p'] for r in pooled])):
        r['p_holm'] = h
    rows_holm += pooled
save('fmb_seg_holm.csv', rows_holm); save('fmb_class.csv', rows_class)
for r in rows_holm:
    print(' %-11s %-16s %-9s %+.4f [%+.4f,%+.4f] holm %.4f %s' % (r['model'], r['family'], r['item'], r['delta'], r['lo'], r['hi'], r['p_holm'], '*' if r['p_holm'] < .05 else ''))
print('\n== Pretrained, class effect, pooled')
for T in TRANS + ['avg']:
    print(' %-9s' % T + ' | '.join('%s %s' % (mo[:3], ' '.join('%s %+.3f[%+.3f,%+.3f]' % (r['cls'][0], r['delta_iou'], r['lo'], r['hi']) for r in rows_class if r['model'] == mo and r['transform'] == T)) for mo in MODELS))

# ---------- retrained, native inputs ----------
D = {}
for i in INPUTS:
    for s in SEEDS:
        d = np.load(os.path.join(R, 'seg_retrain_fmb', '%s_s%d.npz' % (i, s)))
        assert list(d['names']) == list(names)
        D[i, s] = {'c': (d['inter_native'].astype(float), d['pred_native'].astype(float), d['gt_native'].astype(float)),
                   'g': (d['inter3_native'].astype(float), d['pred3_native'].astype(float), d['gt3_native'].astype(float))}
assert all(np.array_equal(D[k]['c'][2], D['avg', 0]['c'][2]) for k in D)
print('\ntest pixels per class', D['avg', 0]['c'][2].sum(0).astype(int).tolist())
C13 = [c for c in range(14) if c != 12]
assert D['avg', 0]['c'][2][:, 12].sum() == 0
SC = {'mIoU13': lambda x: x['c'], 'mIoU3': lambda x: x['g']}
IDX = {'mIoU13': C13, 'mIoU3': [0, 1, 2]}
scores, cmp_rows = [], []
for name, get in SC.items():
    cl = IDX[name]
    per = {i: np.stack([piou(get(D[i, s])) for s in SEEDS]) for i in INPUTS}
    bo = {i: np.stack([biou(get(D[i, s]))[:, cl].mean(1) for s in SEEDS]) for i in INPUTS}      # seeds x B
    S = rng.integers(0, 3, size=(BP, 3))
    for i in INPUTS:
        v = per[i][:, cl].mean(1)
        scores.append(dict(score=name, input=i, mean=float(v.mean()), seed_sd=float(v.std(ddof=1)), s0=float(v[0]), s1=float(v[1]), s2=float(v[2]), **{'c%d' % c: float(per[i][:, c].mean()) for c in range(per[i].shape[1])}))
    for ref in ['avg', 'vi']:
        sub = []
        for m in METHODS:
            d = (bo[m] - bo[ref]).mean(0)
            ds = (bo[m] - bo[ref])[S, np.arange(BP)[:, None]].mean(1)
            sub.append(dict(score=name, reference=ref, input=m, delta=float((per[m][:, cl].mean(1) - per[ref][:, cl].mean(1)).mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p=twoside(d),
                            lo_seed=float(np.percentile(ds, 2.5)), hi_seed=float(np.percentile(ds, 97.5))))
        for r, h in zip(sub, holm([r['p'] for r in sub])):
            r['p_holm'] = h
        cmp_rows += sub
    for ref in ['vi', 'ir']:
        d = (bo['avg'] - bo[ref]).mean(0); ds = (bo['avg'] - bo[ref])[S, np.arange(BP)[:, None]].mean(1)
        cmp_rows.append(dict(score=name, reference=ref, input='avg', delta=float((per['avg'][:, cl].mean(1) - per[ref][:, cl].mean(1)).mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p=twoside(d),
                             lo_seed=float(np.percentile(ds, 2.5)), hi_seed=float(np.percentile(ds, 97.5)), p_holm=''))
save('fmb_retrain_scores.csv', scores); save('fmb_retrain_vs.csv', cmp_rows)
print('== Retrained, native inputs')
for r in scores:
    print(' %-6s %-9s %.4f sd %.4f' % (r['score'], r['input'], r['mean'], r['seed_sd']))
for r in cmp_rows:
    print(' %-6s %-9s - %-3s %+.4f [%+.4f,%+.4f] seeds too [%+.4f,%+.4f] holm %s' % (r['score'], r['input'], r['reference'], r['delta'], r['lo'], r['hi'], r['lo_seed'], r['hi_seed'], ('%.4f' % r['p_holm']) if r['p_holm'] != '' else '-'))
print(' per class, 14 values (bicycle has no test pixel):')
for r in scores:
    if r['score'] == 'mIoU13':
        print(' %-9s ' % r['input'] + ' '.join('%.3f' % r['c%d' % c] for c in range(14)))

# ---------- retrained, seed 0, transformed inputs ----------
E = {}
for m in METHODS:
    d = np.load(os.path.join(R, 'seg_retrain_fmb', m + '_s0.npz'))
    for c in ['native'] + TRANS:
        E[m, c] = {'c': (d['inter_' + c].astype(float), d['pred_' + c].astype(float), d['gt_' + c].astype(float)),
                   'g': (d['inter3_' + c].astype(float), d['pred3_' + c].astype(float), d['gt3_' + c].astype(float))}
rows_cond, ret_method = [], {}
for name, get in SC.items():
    cl = IDX[name]
    fam = []
    for T in TRANS:
        dd = []
        for m in METHODS:
            d = (biou(get(E[m, T])) - biou(get(E[m, 'native'])))[:, cl].mean(1); dd.append(d)
            lo, hi = np.percentile(d, [2.5, 97.5]); dl = float((piou(get(E[m, T])) - piou(get(E[m, 'native'])))[cl].mean())
            rows_cond.append(dict(score=name, method=m, transform=T, delta=dl, lo=float(lo), hi=float(hi), p='', p_holm=''))
            if name == 'mIoU3':
                ret_method[m, T] = (dl, float(lo), float(hi))
        d = np.mean(dd, 0)
        fam.append(dict(score=name, method='pooled', transform=T, delta=float(np.mean([(piou(get(E[m, T])) - piou(get(E[m, 'native'])))[cl].mean() for m in METHODS])), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p=twoside(d)))
    for r, h in zip(fam, holm([r['p'] for r in fam])):
        r['p_holm'] = h
    rows_cond += fam
save('fmb_retrain_cond.csv', rows_cond)
print('\n== Retrained, seed 0, transformed minus native, six methods pooled')
for r in rows_cond:
    if r['method'] == 'pooled':
        print(' %-6s %-9s %+.4f [%+.4f,%+.4f] holm %.4f %s' % (r['score'], r['transform'], r['delta'], r['lo'], r['hi'], r['p_holm'], '*' if r['p_holm'] < .05 else ''))

# ---------- metrics ----------
M = list(csv.DictReader(open(os.path.join(R, 'cond_metrics_fmb.csv'))))
look = {(r['method'], r['condition'], r['image']): r for r in M}
images = sorted({r['image'] for r in M}); n = len(images)
assert n == 100
rows_m = []
print('\n== Metrics, median change per image, averaged over six methods; * = Holm p<.05')
print('%-9s' % '' + ' '.join('%7s' % m for m in METS))
for T in TRANS + ['avg']:
    ps, meds, ups = [], [], []
    for mt in METS:
        sg = -1 if mt == 'Nabf' else 1
        d = []
        for im in images:
            vals = []
            for m in METHODS:
                a = float(look[m, 'native', im][mt])
                b = float(look['base', 'avg', im][mt]) if T == 'avg' else float(look[m, T, im][mt])
                vals.append(sg * (b - a))
            d.append(np.mean(vals))
        d = np.array(d); d = d[np.isfinite(d)]
        ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0); meds.append(np.median(d)); ups.append(np.mean(d > 0))
    hs = holm(ps)
    for mt, med, up, pr, h in zip(METS, meds, ups, ps, hs):
        rows_m.append(dict(transform=T, metric=mt, median_change=float(med), frac_up=float(up), p=float(pr), p_holm=float(h), verdict='up' if (h < .05 and med > 0) else ('down' if (h < .05 and med < 0) else 'ns')))
    print('%-9s' % T + ' '.join('%6.3f%s' % (m, '*' if h < .05 else ' ') for m, h in zip(meds, hs)))
save('fmb_metrics.csv', rows_m)

# ---------- rank attack ----------
Wr = rng.multinomial(n, np.ones(n) / n, size=BRANK).astype(float) / n
rows, summ, nat_rows = [], [], []
for mt in METS:
    sg = -1. if mt == 'Nabf' else 1.
    val = lambda m, c: sg * np.array([float(look[m, c, i][mt]) for i in images])
    natv = np.stack([val(m, 'native') for m in METHODS]); nm = natv.mean(1); nb = Wr @ natv.T
    avg_m = val('base', 'avg').mean()
    order = np.argsort(-nm)
    rank_nat = {METHODS[i]: int(np.sum(nm > nm[i])) + 1 for i in range(6)}
    nat_rows.append(dict(metric=mt, order='>'.join(METHODS[i] for i in order), avg_rank_of_7=int(np.sum(nm > avg_m)) + 1, last=METHODS[order[-1]]))
    for i, m in enumerate(METHODS):
        others = np.delete(np.arange(6), i)
        r0b = 1 + (np.delete(nb, i, 1) > nb[:, [i]]).sum(1)
        for T in TRANS:
            v = val(m, T); a = v.mean(); ab = Wr @ v
            r1 = int(np.sum(nm[others] > a)) + 1
            r1b = 1 + (nb[:, others] > ab[:, None]).sum(1)
            lo, hi = np.percentile(r0b - r1b, [2.5, 97.5])
            row = dict(metric=mt, method=m, transform=T, kind='blend' if T in BLENDS else 'attack', rank_native=rank_nat[m], rank_after=r1, gain=rank_nat[m] - r1, gain_lo=float(lo), gain_hi=float(hi), counted=bool(lo >= 1), first=r1 == 1)
            for mo in MODELS:
                row.update({x + '_' + mo: float(y) for x, y in zip(('d', 'lo', 'hi'), seg_method[mo, m, T])})
            row.update(d_retrained=ret_method[m, T][0], lo_retrained=ret_method[m, T][1], hi_retrained=ret_method[m, T][2])
            row['falls_pretrained'] = bool(row['hi_segformer'] <= 0 or row['hi_mask2former'] <= 0)
            row['falls_retrained'] = bool(row['hi_retrained'] <= 0)
            row['rises_retrained'] = bool(row['lo_retrained'] > 0)
            rows.append(row)
    att = [r for r in rows if r['metric'] == mt and r['kind'] == 'attack']; cnt = [r for r in att if r['counted']]
    wins = sorted({r['method'] for r in cnt if r['first'] and r['rank_native'] > 1})
    lastm = METHODS[order[-1]]
    summ.append(dict(metric=mt, attacks=len(att), counted=len(cnt), harmful_pretrained=sum(r['falls_pretrained'] for r in cnt), harmful_retrained=sum(r['falls_retrained'] for r in cnt), rises_retrained=sum(r['rises_retrained'] for r in cnt),
                     reach_first=','.join(wins), last=lastm, last_best_rank=min(r['rank_after'] for r in att if r['method'] == lastm), avg_rank_of_7=nat_rows[-1]['avg_rank_of_7']))
save('fmb_rank_attack.csv', rows); save('fmb_rank_summary.csv', summ); save('fmb_rank_native.csv', nat_rows)
print('\n== Rank attacks, 48 per metric: counted, harmful on pretrained, harmful on retrained, task rises on retrained, last best rank, average rank of 7')
ms = {r['metric']: r for r in csv.DictReader(open(os.path.join(R, 'rank_summary.csv'))) if r['dataset'] == 'MSRS'}
rep = []
for r in summ:
    c0 = int(ms[r['metric']]['counted'])
    rep.append(dict(metric=r['metric'], counted_msrs=c0, counted_fmb=r['counted'], replicated=(r['counted'] * 2 >= c0) if c0 else ''))
    print(' %-8s %2d %2d %2d %2d  last %-9s best %d  avg %d  first: %s' % (r['metric'], r['counted'], r['harmful_pretrained'], r['harmful_retrained'], r['rises_retrained'], r['last'], r['last_best_rank'], r['avg_rank_of_7'], r['reach_first']))
save('fmb_replication.csv', rep)
print('\n== Replication of the rank attack: counted on MSRS, on FMB, at least half as many')
for r in rep:
    print(' %-8s %2d %2d %s' % (r['metric'], r['counted_msrs'], r['counted_fmb'], r['replicated']))
print('DONE')
