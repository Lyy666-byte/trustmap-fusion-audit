# -*- coding: utf-8 -*-
"""Rank and class analyses of the existing data, as fixed in the ANALYSIS_PLAN.md addendum of the same name.
Writes results/rank_attack.csv, rank_summary.csv, rank_native.csv, class_effects.csv, seg_holm.csv."""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
METHODS = ['DCEvo', 'EMMA', 'LUTFuse', 'MMDRFuse', 'CDDFuse', 'SeAFusion']
ATTACK = ['noise1', 'noise3', 'sharp1', 'sharp2', 'gamma07', 'gamma14', 'contrast', 'blur1']
BLENDS = ['blend25', 'blend50', 'blend75']
TRANS = BLENDS + ATTACK
METS = ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'MS_SSIM', 'CC', 'Qabf', 'FMI', 'Nabf', 'SCD', 'VIF']
MODELS = ['segformer', 'mask2former']
CLASSES = ['person', 'car', 'bike']
rng = np.random.default_rng(20261031)
BRANK, BP = 5000, 10000


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


# ---------- segmentation, per image counts ----------
def load(model, key):
    d = np.load(os.path.join(R, 'seg', model, key + '.npz'))
    return d['inter'].astype(float), d['pred'].astype(float), d['gt'].astype(float), d['names']


names = load('segformer', 'base_avg')[3]; N = len(names)
night = np.array([n[-5] == 'N' for n in names])
SCOPES = (('all', np.ones(N, bool)), ('day', ~night), ('night', night))
W = rng.multinomial(N, np.ones(N) / N, size=BP).astype(float)


def biou(x, Wm):
    """Dataset IoU per class for every resample, B x 3."""
    I, P, G = np.einsum('bn,nc->bc', Wm, x[0]), np.einsum('bn,nc->bc', Wm, x[1]), np.einsum('bn,nc->bc', Wm, x[2])
    return I / np.maximum(P + G - I, 1)


def piou(x, m):
    I, P, G = x[0][m].sum(0), x[1][m].sum(0), x[2][m].sum(0)
    return I / np.maximum(P + G - I, 1)


def twoside(d):
    return min(1., 2 * min(np.mean(d <= 0), np.mean(d >= 0)) + 1 / len(d))


seg_method = {}
rows_class, rows_holm = [], []
for model in MODELS:
    nat = {m: load(model, m + '_native') for m in METHODS}
    avg = load(model, 'base_avg')
    natb = {m: biou(nat[m], W) for m in METHODS}
    avgb = biou(avg, W)
    pooled = []
    for T in TRANS + ['avg']:
        cond = {m: (avg if T == 'avg' else load(model, m + '_' + T)) for m in METHODS}
        full = np.ones(N, bool)
        condb = {m: (avgb if T == 'avg' else biou(cond[m], W)) for m in METHODS}
        for m in METHODS:
            d = condb[m] - natb[m]; dm = d.mean(1)[:BRANK]
            seg_method[(model, m, T)] = ((piou(cond[m], full) - piou(nat[m], full)).mean(), *np.percentile(dm, [2.5, 97.5]))
        dc = np.mean([condb[m] - natb[m] for m in METHODS], 0)
        pt = np.mean([piou(cond[m], full) - piou(nat[m], full) for m in METHODS], 0)
        for c, name in enumerate(CLASSES):
            lo, hi = np.percentile(dc[:, c], [2.5, 97.5])
            rows_class.append(dict(model=model, transform=T, cls=name, delta_iou=pt[c], lo=lo, hi=hi))
        dd = dc.mean(1)
        pooled.append(dict(model=model, family='transform', scope='all', item=T, delta=pt.mean(), lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5), p=twoside(dd)))
    for r, h in zip(pooled, holm([r['p'] for r in pooled])):
        r['p_holm'] = h; rows_holm.append(r)
    for sc, mask in SCOPES:
        Wm = W * mask[None, :]
        ab = biou(avg, Wm)
        fam = []
        for m in METHODS:
            dd = (biou(nat[m], Wm) - ab).mean(1)
            fam.append(dict(model=model, family='native_minus_avg', scope=sc, item=m, delta=(piou(nat[m], mask) - piou(avg, mask)).mean(), lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5), p=twoside(dd)))
        for r, h in zip(fam, holm([r['p'] for r in fam])):
            r['p_holm'] = h; rows_holm.append(r)
save('class_effects.csv', rows_class); save('seg_holm.csv', rows_holm)

print('== Task, Holm')
for r in rows_holm:
    print(' %-11s %-16s %-5s %-9s %+.4f [%+.4f,%+.4f] p %.4f holm %.4f %s' % (r['model'], r['family'], r['scope'], r['item'], r['delta'], r['lo'], r['hi'], r['p'], r['p_holm'], '*' if r['p_holm'] < .05 else ''))
print('\n== Class effect, six methods pooled (delta IoU [95% CI])')
for T in TRANS + ['avg']:
    print(' %-9s' % T + ' | '.join('%s %s' % (mo[:3], ' '.join('%s %+.3f[%+.3f,%+.3f]' % (r['cls'][0], r['delta_iou'], r['lo'], r['hi']) for r in rows_class if r['model'] == mo and r['transform'] == T)) for mo in MODELS))

# ---------- metric ranks ----------
D = list(csv.DictReader(open(os.path.join(R, 'cond_metrics.csv')))) + list(csv.DictReader(open(os.path.join(R, 'cond_metrics_ext.csv'))))
look = {(r['dataset'], r['method'], r['condition'], r['image']): r for r in D}
rows, summ, nat_rows = [], [], []
for ds in ['MSRS', 'RoadScene']:
    images = sorted({r['image'] for r in D if r['dataset'] == ds}); n = len(images)
    Wr = rng.multinomial(n, np.ones(n) / n, size=BRANK).astype(float) / n
    for mt in METS:
        sg = -1. if mt == 'Nabf' else 1.
        val = lambda m, c: sg * np.array([float(look[(ds, m, c, i)][mt]) for i in images])
        natv = np.stack([val(m, 'native') for m in METHODS])      # 6 x n
        nm = natv.mean(1)
        nb = Wr @ natv.T                                           # B x 6
        avg_m = val('base', 'avg').mean()
        order = np.argsort(-nm)
        rank_nat = {METHODS[i]: int(np.sum(nm > nm[i])) + 1 for i in range(6)}
        nat_rows.append(dict(dataset=ds, metric=mt, order='>'.join(METHODS[i] for i in order), avg_rank_of_7=int(np.sum(nm > avg_m)) + 1, last=METHODS[order[-1]]))
        for i, m in enumerate(METHODS):
            others = np.delete(np.arange(6), i)
            r0b = 1 + (np.delete(nb, i, 1) > nb[:, [i]]).sum(1)
            for T in TRANS:
                v = val(m, T); a = v.mean(); ab = Wr @ v
                r1 = int(np.sum(nm[others] > a)) + 1
                r1b = 1 + (nb[:, others] > ab[:, None]).sum(1)
                gain = r0b - r1b
                lo, hi = np.percentile(gain, [2.5, 97.5])
                row = dict(dataset=ds, metric=mt, method=m, transform=T, kind='blend' if T in BLENDS else 'attack', rank_native=rank_nat[m], rank_after=r1,
                           gain=rank_nat[m] - r1, gain_lo=float(lo), gain_hi=float(hi), counted=bool(lo >= 1), first=r1 == 1, p_first=float(np.mean(r1b == 1)))
                if ds == 'MSRS':
                    for mo in MODELS:
                        d, l, h = seg_method[(mo, m, T)]
                        row.update({'d_' + mo: float(d), 'lo_' + mo: float(l), 'hi_' + mo: float(h)})
                    row['task_falls'] = bool(row['hi_segformer'] <= 0 or row['hi_mask2former'] <= 0)
                    row['harmful'] = bool(row['counted'] and row['task_falls'])
                else:
                    for mo in MODELS:
                        row.update({'d_' + mo: '', 'lo_' + mo: '', 'hi_' + mo: ''})
                    row['task_falls'] = ''; row['harmful'] = ''
                rows.append(row)
        att = [r for r in rows if r['dataset'] == ds and r['metric'] == mt and r['kind'] == 'attack']
        cnt = [r for r in att if r['counted']]
        wins = sorted({r['method'] for r in att if r['first'] and r['rank_native'] > 1 and r['counted']})
        lastm = METHODS[order[-1]]
        lastbest = min(r['rank_after'] for r in att if r['method'] == lastm)
        summ.append(dict(dataset=ds, metric=mt, attacks=len(att), counted=len(cnt), harmful=sum(r['harmful'] is True for r in att),
                         methods_can_gain=len({r['method'] for r in cnt}), methods_reach_first=len(wins), reach_first=','.join(wins), last=lastm, last_best_rank=lastbest,
                         avg_rank_of_7=nat_rows[-1]['avg_rank_of_7']))
save('rank_attack.csv', rows); save('rank_summary.csv', summ); save('rank_native.csv', nat_rows)

print('\n== Native order and rank of the pixel average among seven')
for r in nat_rows:
    print(' %-9s %-8s %s  avg rank %d' % (r['dataset'], r['metric'], r['order'], r['avg_rank_of_7']))
print('\n== Attacks of F alone: 48 per metric. counted = rank gain with CI lower bound >= 1')
for r in summ:
    print(' %-9s %-8s counted %2d harmful %2d methods gaining %d reach first %d (%s) last %s best rank %d' % (r['dataset'], r['metric'], r['counted'], r['harmful'], r['methods_can_gain'], r['methods_reach_first'], r['reach_first'], r['last'], r['last_best_rank']))
print('\n== Counted attacks by transformation (all metrics), MSRS / RoadScene')
for T in ATTACK:
    print(' %-9s ' % T + ' / '.join(','.join('%s:%d' % (mt, sum(r['counted'] for r in rows if r['dataset'] == ds and r['metric'] == mt and r['transform'] == T)) for mt in METS if any(r['counted'] for r in rows if r['dataset'] == ds and r['metric'] == mt and r['transform'] == T)) or '-' for ds in ['MSRS', 'RoadScene']))
print('\n== Blends: counted gains, MSRS / RoadScene')
for T in BLENDS:
    print(' %-9s ' % T + ' / '.join(','.join('%s:%d' % (mt, sum(r['counted'] for r in rows if r['dataset'] == ds and r['metric'] == mt and r['transform'] == T)) for mt in METS if any(r['counted'] for r in rows if r['dataset'] == ds and r['metric'] == mt and r['transform'] == T)) or '-' for ds in ['MSRS', 'RoadScene']))
print('\n== Largest harmful gains on MSRS')
h = sorted([r for r in rows if r['harmful'] is True], key=lambda r: (-r['gain'], r['metric']))
for r in h[:25]:
    print(' %-8s %-9s %-8s rank %d->%d CI[%g,%g] seg %+.3f m2f %+.3f' % (r['metric'], r['method'], r['transform'], r['rank_native'], r['rank_after'], r['gain_lo'], r['gain_hi'], r['d_segformer'], r['d_mask2former']))
