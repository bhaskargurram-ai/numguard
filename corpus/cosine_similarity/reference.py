"""Scaled cosine similarity with an explicit zero-vector convention."""

import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    dt = np.result_type(a, b)
    if np.isnan(a).any() or np.isnan(b).any():
        return dt.type(np.nan)
    sa = np.max(np.abs(a), initial=0.0)
    sb = np.max(np.abs(b), initial=0.0)
    if sa == 0 or sb == 0:
        return dt.type(0.0)
    acc = np.float64 if dt.itemsize < 8 else dt.type
    ua = a.astype(acc) / acc(sa)
    ub = b.astype(acc) / acc(sb)
    if not (np.all(np.isfinite(ua)) and np.all(np.isfinite(ub))):
        # infinite components: only their directions matter
        ua = np.where(np.isinf(a), np.sign(a), 0).astype(acc)
        ub = np.where(np.isinf(b), np.sign(b), 0).astype(acc)
    na, nb = np.linalg.norm(ua), np.linalg.norm(ub)
    if na == 0 or nb == 0:
        return dt.type(0.0)
    c = np.dot(ua / na, ub / nb)
    return dt.type(np.clip(c, -1.0, 1.0))
