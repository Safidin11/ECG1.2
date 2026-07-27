"""Связи между отведениями от конечностей: восстановление дыр и проверка раскладки.

Шесть отведений от конечностей (I, II, III, aVR, aVL, aVF) не независимы — они
все выводятся из ДВУХ. Так они и формируются в самом кардиографе:

    III  = II - I                     (закон Эйнтховена)
    aVR  = -(I + II) / 2              (формулы Гольдбергера)
    aVL  = I - II/2
    aVF  = II - I/2

Отсюда два инструмента:

1. ВОССТАНОВЛЕНИЕ. Если в какой-то момент времени известны хотя бы два
   независимых отведения, остальные четыре ВЫЧИСЛЯЮТСЯ точно. Это не
   дорисовывание: значение определено математически, как в самом приборе.

2. ПРОВЕРКА. Если известны три и более, система переопределена, и невязка
   показывает, верна ли оцифровка. Большая невязка = перепутана раскладка,
   неверный масштаб или мусор вместо сигнала.

Важно про время: связи действуют только между отведениями, записанными
ОДНОВРЕМЕННО. В 12x1 это все 10 секунд. В 3x4 колонка 1 (I, II, III) — это
секунды 0-2.5, а колонка 2 (aVR, aVL, aVF) — секунды 2.5-5; связывать их
между собой нельзя. Здесь это учтено само собой: расчёт идёт по каждому
отсчёту отдельно, по тем отведениям, что в этот момент известны.

Грудные (V1-V6) регистрируются независимо — для них связей нет.
"""
from __future__ import annotations

import numpy as np

LIMB = ["I", "II", "III", "aVR", "aVL", "aVF"]

# Строки — отведения из LIMB, столбцы — базис (I, II).
BASIS = np.array([
    [1.0,  0.0],     # I
    [0.0,  1.0],     # II
    [-1.0, 1.0],     # III = II - I
    [-0.5, -0.5],    # aVR = -(I + II)/2
    [1.0,  -0.5],    # aVL = I - II/2
    [-0.5, 1.0],     # aVF = II - I/2
])

MIN_RANK = 2          # меньше двух независимых отведений -> восстановить нечего


def _limb_columns(leads: list[str]) -> dict[str, int]:
    return {n: leads.index(n) for n in LIMB if n in leads}


def reconstruct_limb(sig: np.ndarray, leads: list[str],
                     min_known: int = 2) -> tuple[np.ndarray, dict]:
    """Заполнить пропуски в отведениях от конечностей по их взаимным связям.

    sig — (N, L) с NaN в пропусках, leads — имена столбцов.
    Возвращает (новый массив, отчёт). Исходные значения не трогаются:
    заполняются только NaN.
    """
    out = sig.copy()
    cols = _limb_columns(leads)
    report = {"filled": {}, "skipped": "", "checked": 0}
    if len(cols) < MIN_RANK:
        report["skipped"] = f"известно лишь {len(cols)} отведений от конечностей"
        return out, report

    names = list(cols)                       # какие из шести вообще есть
    idx = [cols[n] for n in names]
    rows = np.array([BASIS[LIMB.index(n)] for n in names])   # (k, 2)
    obs = sig[:, idx]                                        # (N, k)
    known = ~np.isnan(obs)

    # Группируем отсчёты по одинаковому набору известных отведений — для каждой
    # такой группы система решается один раз на всю группу (быстро и устойчиво).
    filled_count = {n: 0 for n in LIMB}
    patterns = np.packbits(known, axis=1)
    uniq, inverse = np.unique(patterns, axis=0, return_inverse=True)
    for u in range(len(uniq)):
        sel = inverse == u
        mask = known[np.flatnonzero(sel)[0]]
        if mask.sum() < min_known:
            continue
        A = rows[mask]                                   # (m, 2)
        if np.linalg.matrix_rank(A) < MIN_RANK:          # напр. только I и III
            continue
        y = obs[np.ix_(sel, mask)].T                     # (m, n_sel)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)        # базис (2, n_sel)
        rec = BASIS @ x                                  # все шесть (6, n_sel)
        for j, name in enumerate(LIMB):
            if name not in cols:
                continue
            col = cols[name]
            gap = sel & np.isnan(sig[:, col])
            if gap.any():
                out[gap, col] = rec[j][np.isnan(obs[sel, names.index(name)])]
                filled_count[name] += int(gap.sum())

    report["filled"] = {k: v for k, v in filled_count.items() if v}
    report["checked"] = int(known.all(axis=1).sum())
    return out, report


def _ptp(x: np.ndarray) -> float:
    """Размах сигнала (робастный: 1-99 процентиль, чтобы выброс не раздул)."""
    v = x[~np.isnan(x)]
    if len(v) < 10:
        return 0.0
    lo, hi = np.percentile(v, [1, 99])
    return float(hi - lo)


def consistency(sig: np.ndarray, leads: list[str]) -> dict:
    """Насколько оцифровка согласуется со связями отведений.

    Считает относительную невязку: насколько наблюдаемые отведения расходятся
    с лучшей подгонкой под связи. ~0 = данные согласованы; заметная величина =
    перепутана раскладка, неверный масштаб или шум вместо сигнала.
    """
    cols = _limb_columns(leads)
    res = {"rank": len(cols), "residual": None, "einthoven": None, "goldberger": None}
    if len(cols) < 3:
        return res

    names = list(cols)
    rows = np.array([BASIS[LIMB.index(n)] for n in names])
    obs = sig[:, [cols[n] for n in names]]
    full = obs[~np.isnan(obs).any(axis=1)]
    if len(full) >= 100:                       # иначе общего окна нет (3x4 колонки)
        x, *_ = np.linalg.lstsq(rows, full.T, rcond=None)
        pred = (rows @ x).T
        scale = np.mean([_ptp(full[:, j]) for j in range(full.shape[1])]) + 1e-9
        res["residual"] = float(np.std(full - pred) / scale)

    if all(n in cols for n in ("I", "II", "III")):      # II = I + III
        a, b, c = (sig[:, cols[n]] for n in ("I", "II", "III"))
        ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c)
        if ok.sum() > 100:
            res["einthoven"] = float(np.std((b - a - c)[ok]) / (_ptp(b[ok]) + 1e-9))
    if all(n in cols for n in ("aVR", "aVL", "aVF")):   # aVR + aVL + aVF = 0
        r, l, f = (sig[:, cols[n]] for n in ("aVR", "aVL", "aVF"))
        ok = ~np.isnan(r) & ~np.isnan(l) & ~np.isnan(f)
        if ok.sum() > 100:
            scale = np.mean([_ptp(r[ok]), _ptp(l[ok]), _ptp(f[ok])]) + 1e-9
            res["goldberger"] = float(np.std((r + l + f)[ok]) / scale)
    return res
