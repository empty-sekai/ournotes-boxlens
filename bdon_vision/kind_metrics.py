"""Retrieval, acceptance and open-set metrics for gallery encoders (numpy only).

A query is accepted when its best gallery similarity is at least ``t`` and
exceeds the second best by at least ``m``. Open-set behaviour is measured by
removing the query's own card from the gallery ("leave-true-out") and counting
how often the remaining gallery still produces an accepted answer, plus the
accept rate on tiles whose artwork is outside the catalog.
"""
import numpy as np

T_GRID = np.round(np.arange(.30, .96, .01), 2)
M_GRID = np.round(np.arange(0., .31, .01), 2)


def top2(scores):
    order = np.argsort(-scores, axis=1)[:, :2]
    rows = np.arange(len(scores))[:, None]
    values = scores[rows, order]
    return order[:, 0], values[:, 0], values[:, 1]


def retrieval_table(queries, gallery, labels, held):
    """Per-query retrieval facts.

    queries [N, D], gallery [K, D] (L2-normalised), labels [N] gallery index or
    -1 for artwork outside the catalog, held [K] bool marking identities that
    never entered training.
    """
    labels = np.asarray(labels)
    scores = np.asarray(queries, np.float64) @ np.asarray(gallery, np.float64).T
    known = labels >= 0
    pred, s1, s2 = top2(scores)
    loo = scores.copy()
    loo[np.flatnonzero(known), labels[known]] = -np.inf
    loo_pred, loo_s1, loo_s2 = top2(loo)
    true_score = np.full(len(labels), np.nan)
    true_score[known] = scores[np.flatnonzero(known), labels[known]]
    group = np.where(~known, 'foreign', np.where(np.asarray(held)[np.maximum(labels, 0)], 'unseen', 'seen'))
    return {'pred': pred, 's1': s1, 's2': s2, 'correct': known & (pred == labels), 'labels': labels,
            'loo_pred': loo_pred, 'loo_s1': loo_s1, 'loo_s2': loo_s2, 'true_score': true_score, 'group': group}


def accepted(s1, s2, t, m):
    return (s1 >= t) & (s1 - s2 >= m)


def _rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def summarize(table, t, m):
    """Closed-set, acceptance and open-set numbers per identity group."""
    out = {'thresholds': {'similarity': float(t), 'margin': float(m)}}
    acc = accepted(table['s1'], table['s2'], t, m)
    loo_acc = accepted(table['loo_s1'], table['loo_s2'], t, m)
    for name in ('seen', 'unseen', 'known', 'foreign'):
        mask = table['group'] != 'foreign' if name == 'known' else table['group'] == name
        n = int(mask.sum())
        row = {'count': n}
        if name != 'foreign':
            correct = table['correct'][mask]
            a = acc[mask]
            row.update({
                'top1': _rate(int(correct.sum()), n),
                'accept_rate': _rate(int(a.sum()), n),
                'accepted_precision': _rate(int((a & correct).sum()), int(a.sum())),
                'accepted_correct_rate': _rate(int((a & correct).sum()), n),
                'wrong_accepts': int((a & ~correct).sum()),
                'open_set_false_accept': _rate(int(loo_acc[mask].sum()), n),
                'median_true_similarity': None if n == 0 else float(np.median(table['true_score'][mask])),
                'median_margin': None if n == 0 else float(np.median((table['s1'] - table['s2'])[mask])),
            })
        else:
            row['false_accept'] = _rate(int(acc[mask].sum()), n)
        out[name] = row
    return out


def select_thresholds(table, max_false_accept=.01, min_precision=.995, tolerance=.001):
    """Pick (t, m) maximising known accepted-correct rate under open-set limits.

    Limits: unseen leave-true-out false accepts and foreign-artwork accepts each
    at most ``max_false_accept``; accepted precision over known queries at
    least ``min_precision``. Seen and unseen identities are weighted equally.
    Among settings within ``tolerance`` of the best score, those with the
    fewest open-set accepts are kept and the median similarity threshold (then
    the median margin at that threshold) is returned, which stays away from
    the edge of the feasible region.
    """
    group = table['group']
    seen, unseen, foreign = group == 'seen', group == 'unseen', group == 'foreign'
    known = ~foreign
    margin, loo_margin = table['s1'] - table['s2'], table['loo_s1'] - table['loo_s2']
    rows = []
    for t in T_GRID:
        hi = table['s1'] >= t
        loo_hi = table['loo_s1'] >= t
        for m in M_GRID:
            acc = hi & (margin >= m)
            loo = loo_hi & (loo_margin >= m)
            far_unseen = float(loo[unseen].mean()) if unseen.any() else 0.
            far_foreign = float(acc[foreign].mean()) if foreign.any() else 0.
            a = acc[known]
            precision = (a & table['correct'][known]).sum() / max(a.sum(), 1)
            if far_unseen > max_false_accept or far_foreign > max_false_accept or precision < min_precision:
                continue
            good = acc & table['correct']
            parts = [good[g].mean() for g in (seen, unseen) if g.any()]
            rows.append((float(np.mean(parts)) if parts else 0., far_unseen + far_foreign, float(t), float(m)))
    if not rows:
        return {'similarity': None, 'margin': None, 'score': 0., 'feasible': False,
                'max_false_accept': max_false_accept, 'min_precision': min_precision}
    best = max(r[0] for r in rows)
    tied = [r for r in rows if r[0] >= best - tolerance]
    low = min(r[1] for r in tied)
    tied = [r for r in tied if r[1] <= low + 1e-12]
    ts = sorted({r[2] for r in tied})
    t = ts[(len(ts) - 1) // 2]
    ms = sorted(r[3] for r in tied if r[2] == t)
    m = ms[(len(ms) - 1) // 2]
    score = next(r[0] for r in tied if r[2] == t and r[3] == m)
    return {'similarity': t, 'margin': m, 'score': score, 'feasible': True,
            'max_false_accept': max_false_accept, 'min_precision': min_precision,
            'tied_similarity_range': [ts[0], ts[-1]]}


def open_set_curve(table, points=(.005, .01, .02, .05)):
    """Known accepted-correct rate at several open-set false-accept limits."""
    rows = []
    for limit in points:
        choice = select_thresholds(table, max_false_accept=limit)
        row = {'max_false_accept': limit, **choice}
        if choice['feasible']:
            row['summary'] = summarize(table, choice['similarity'], choice['margin'])
        rows.append(row)
    return rows
