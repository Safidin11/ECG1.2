"""Стадия vectorize: НЕЗАВИСИМАЯ трассировка каждого отведения.

Для каждой клетки (своё окно + своя базовая линия из layout) трассируем сигнал
«следованием»: в каждом столбце берём кластер пикселей, ближайший к предыдущей
точке (устойчиво к смещению соседних отведений, глубоким S-зубцам и толщине
штриха). Где маска дырявая — подхватываем тёмные пиксели полутона рядом с
текущей траекторией. Разрывы интерполируем и продолжаем (не останавливаемся).
Дрейф базовой линии снимаем скользящей медианой (чинит длинную ритм-строку).

Соседние отведения НЕ влияют друг на друга: у каждого своё окно и базовая линия.

Вход:  layout.json (layout.cells с per-lead bbox/baseline + mask_png + core_ready).
Выход: vectorize.json (signal_npy/leads/fs/coverage) + preview.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir, load_ink  # noqa: E402

STAGE = "vectorize"
log = get_logger(STAGE)

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SLEW_PX = 70   # макс. скачок трассы за столбец (px) — против «прямоугольников»


def _clusters(colpix):
    """Кластеры подряд идущих ненулевых пикселей столбца -> (центроид, высота)."""
    ys = np.where(colpix > 0)[0]
    if len(ys) == 0:
        return []
    out, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y - prev > 3:
            out.append(((start + prev) / 2, prev - start + 1))
            start = y
        prev = y
    out.append(((start + prev) / 2, prev - start + 1))
    return out


def _trace_follow(ink, bbox, baseline, slew=SLEW_PX):
    """Трассировка отведения по цветовым чернилам следованием за кластером.

    В каждом столбце берём кластер, ближайший к текущей траектории (при равенстве
    — тоньше, чтобы не липнуть к разделителю/тексту). Скачок > slew запрещён
    (иначе трасса «проваливается» в глубокий S/шум прямоугольником). Разрывы
    интерполируем и продолжаем. slew=None — без ограничения (для чистой изолир.
    компоненты, где нет сетки/соседей и крутой QRS не должен резаться).
    """
    x0, y0, x1, y1 = bbox
    n = x1 - x0
    ys = np.full(n, np.nan)
    prev = baseline - y0
    for i in range(n):
        cl = _clusters(ink[y0:y1, x0 + i])
        if not cl:
            continue
        c, _h = min(cl, key=lambda t: (abs(t[0] - prev), t[1]))
        if slew is None or abs(c - prev) <= slew:
            ys[i] = c
            prev = c
    idx = np.arange(n)
    good = ~np.isnan(ys)
    if good.sum() < 5:
        return None, 0.0
    cov = float(good.mean())
    ys = np.interp(idx, idx[good], ys[good]) + y0
    return ys, cov


def _qmetric(ys, cov, lo, hi, base, ncol):
    """Качество восстановления отведения Q∈[0..1] (по синтезу дизайн-панели).

    Штрафы: пропуски (1−cov), обрезка у края окна, плато на краю (обрезанный QRS),
    baseline прижат к краю (сполз на соседа). Клип/пропуски весомее.
    """
    if ys is None or len(ys) == 0:
        return 0.0
    H = max(1, hi - lo)
    K = max(2, int(round(0.02 * H)))
    clipped = (ys <= lo + K) | (ys >= hi - K)
    clip_frac = float(clipped.mean())
    # самый длинный подряд-«прижатый» участок (плато обрезки)
    run = mx = 0
    for c in clipped:
        run = run + 1 if c else 0
        mx = max(mx, run)
    p_gap = 1.0 - float(cov)
    p_clip = clip_frac
    p_plateau = min(1.0, mx / (0.10 * max(1, ncol)))
    edge_dist = min(base - lo, hi - base)
    p_base = min(1.0, max(0.0, 1.0 - edge_dist / (0.25 * H)))
    # штраф за скачки/ступеньки: доля соседних отсчётов с большим скачком
    diffs = np.abs(np.diff(ys)) if len(ys) > 1 else np.array([0.0])
    p_jump = min(1.0, float(np.mean(diffs > 0.18 * H)) / 0.03)
    q = 1.0 - (0.30 * p_gap + 0.25 * p_clip + 0.15 * p_plateau
               + 0.15 * p_base + 0.15 * p_jump)
    return min(1.0, max(0.0, q))


def _isolate_lead_ink(mask, x0, x1, ytop, ybot, seed_ys, base_y):
    """ROI-компоненты ink, связные с трассой отведения или его baseline.

    Гейт связности: высокий QRS — одна компонента, прикреплённая к изолинии
    отведения (заявляем её), а отдельная компонента соседа отбрасывается.
    """
    sub = (mask[ytop:ybot, x0:x1] > 0).astype(np.uint8)
    n, lbl = cv2.connectedComponents(sub, 8)
    seed = np.zeros_like(sub)
    if seed_ys is not None:
        for i, y in enumerate(seed_ys):
            yy = int(round(y)) - ytop
            if 0 <= yy < sub.shape[0]:
                seed[yy, i] = 1
    br = base_y - ytop
    if 0 <= br < sub.shape[0]:
        seed[br, :] = 1
    labels = set(np.unique(lbl[(seed > 0) & (sub > 0)]).tolist()) - {0}
    keep = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)   # полный размер
    kk = np.isin(lbl, list(labels)) if labels else np.zeros_like(sub, bool)
    keep[ytop:ybot, x0:x1] = kk.astype(np.uint8)
    return keep


def _walk_extent(h, start, limit):
    """От строки start идём наружу, пока не встретим пустой прогон (≥limit строк ink≤1)."""
    out = {}
    for d in (-1, 1):
        empty = 0
        y = start
        last = 0
        while 0 <= y + d < len(h):
            y += d
            if h[y] <= 1:
                empty += 1
                if empty >= limit:
                    break
            else:
                empty = 0
                last = abs(y - start)
        out[d] = last
    return out[-1], out[1]


def _layer2_recut(mask, x0, x1, hard_top, hard_bot, seed_ys):
    """Слой 2: baseline по x-взвешенному профилю + перекрой полосы по краю чернил.

    Широкая изолиния даёт максимум покрытия по строке (узкий QRS — нет), поэтому
    argmax профиля — это настоящая базовая линия, не зубец и не сосед. Реальные
    границы находим «прогулкой» до пустых промежутков вверх/вниз.
    """
    ncol = max(1, x1 - x0)
    c = mask[hard_top:hard_bot, x0:x1].sum(1).astype(np.float32) / ncol
    c = gaussian_filter1d(c, 2.0)
    if c.max() <= 0:
        return None
    y_base = hard_top + int(np.argmax(c))
    lead = _isolate_lead_ink(mask, x0, x1, hard_top, hard_bot, seed_ys, y_base)
    h = lead.sum(1)                                   # полноразмерный, индекс абсолютный
    H = hard_bot - hard_top
    up, down = _walk_extent(h, y_base, max(3, int(0.02 * H)))
    pad = max(6, int(0.06 * (up + down)))
    lo = max(hard_top, y_base - up - pad)
    hi = min(hard_bot, y_base + down + pad)
    # Трассируем по ИЗОЛИРОВАННОЙ компоненте без ограничения скачка: нет прыжков
    # на сетку/соседа, крутой QRS не режется.
    bridged = cv2.morphologyEx(lead, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    ys, cov = _trace_follow(bridged, [x0, lo, x1, hi], y_base, slew=None)
    return ys, cov, lo, hi, y_base


def _trace_cascade(ink, x0, x1, lo, hi, base, top_lim, bot_lim):
    """Каскад: слой 1 (окно из layout) -> при плохом Q слой 2 (перекрой по чернилам).
    Держим лучший по Q результат; ранний выход при Q≥0.9.
    """
    ncol = x1 - x0
    ys, cov = _trace_follow(ink, [x0, lo, x1, hi], base)
    best = (ys, cov, _qmetric(ys, cov, lo, hi, base, ncol), lo, hi)
    # Пытаемся улучшить любое неидеальное отведение (keep-best -> без регресса).
    if best[2] < 0.93:
        try:
            r = _layer2_recut(ink, x0, x1, top_lim, bot_lim, ys)
        except Exception:
            r = None
        if r is not None:
            ys2, cov2, lo2, hi2, base2 = r
            q2 = _qmetric(ys2, cov2, lo2, hi2, base2, ncol)
            if q2 > best[2]:
                best = (ys2, cov2, q2, lo2, hi2)
    return best


def _to_mv(ys, mm_px, seconds, fs, clip):
    mV = -(ys - np.median(ys)) / (10.0 * mm_px)
    # снятие дрейфа базовой линии скользящей медианой (~0.6с)
    win = int(0.6 * fs)
    win = min(win if win % 2 else win + 1, (len(mV) // 2) * 2 - 1)
    if 3 <= win < len(mV):
        mV = mV - median_filter(mV, size=win)
    mV = np.clip(mV, -clip, clip)
    target = int(fs * seconds)
    return np.interp(np.linspace(0, 1, target), np.linspace(0, 1, len(mV)), mV)


def _render_geometry(shape, px_traces, mm_px):
    """Реконструкция в исходной геометрии: холст РАЗМЕРА картинки, светлая
    ЭКГ-сетка и восстановленные трассы в тех же пиксельных координатах."""
    H, W = shape[:2]
    canvas = np.full((H, W, 3), 255, np.uint8)
    minor = max(2, int(round(mm_px)))          # 1 мм
    major = minor * 5                          # 5 мм
    c_minor = (205, 200, 248)                  # BGR: светло-розовая сетка 1 мм
    c_major = (170, 165, 240)                  # 5 мм — насыщеннее
    for x in range(0, W, minor):
        canvas[:, x] = c_minor
    for y in range(0, H, minor):
        canvas[y, :] = c_minor
    for x in range(0, W, major):
        canvas[:, x] = c_major
    for y in range(0, H, major):
        canvas[y, :] = c_major
    for x0, ys in px_traces:
        pts = np.array([[x0 + i, int(round(y))] for i, y in enumerate(ys)], np.int32)
        cv2.polylines(canvas, [pts], False, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    clip_mV = float(config.get("_stage_params", {}).get("clip_mV", 3.0))

    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    layout = manifest.get("layout")
    if not layout:
        log.warning("STAGE %s: нет layout — пропуск (оставляю сигнал ядра)", STAGE)
        out_path = out_dir / "vectorize.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return str(out_path)

    core_img = manifest.get("core_ready_image")
    ink = load_ink(core_img) if core_img and Path(core_img).exists() else \
        (cv2.imread(manifest["mask_png"], cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8)
    mm_px = layout["mm_per_px"]
    fs = manifest.get("fs", 500)
    n_full = int(fs * 10)

    # Блочные клетки (короткие отведения). px_traces — трассы в ИСХОДНЫХ
    # пиксельных координатах (для геометрически идентичной реконструкции).
    # Группируем по КОЛОНКЕ и сортируем по y, чтобы каскад знал границы соседей.
    signals, coverage, px_traces = {}, {}, []
    H = ink.shape[0]
    by_col = {}
    for lead, cell in layout["cells"].items():
        by_col.setdefault(cell["col"], []).append((lead, cell))
    for col, items in by_col.items():
        items.sort(key=lambda t: t[1]["bbox"][1])   # по верхней границе окна
        for k, (lead, cell) in enumerate(items):
            x0, lo, x1, hi = cell["bbox"]
            top_lim = items[k - 1][1]["bbox"][3] if k > 0 else 0
            bot_lim = items[k + 1][1]["bbox"][1] if k < len(items) - 1 else H
            ys, cov, q, lo_u, hi_u = _trace_cascade(
                ink, x0, x1, lo, hi, cell["baseline"], top_lim, bot_lim)
            if ys is not None:
                signals[lead] = _to_mv(ys, mm_px, cell["seconds"], fs, clip_mV)
                coverage[lead] = round(cov, 3)
                px_traces.append((x0, ys))
                cell["bbox"] = [x0, lo_u, x1, hi_u]   # обновим окно для overlay
    # Ритм-строки (полные 10с) — их может быть несколько (напр. V1/II/V5).
    rhythm_sigs = {}
    for rs in layout.get("rhythm_strips", []):
        ys, rcov = _trace_follow(ink, rs["bbox"], rs["baseline"])
        if ys is not None:
            rhythm_sigs[rs["lead"]] = _to_mv(ys, mm_px, rs["seconds"], fs, clip_mV)
            coverage[rs["lead"] + "_rhythm"] = round(rcov, 3)
            px_traces.append((rs["bbox"][0], ys))

    # Геометрически идентичная реконструкция: те же трассы на холсте РАЗМЕРА
    # исходной картинки, в тех же координатах и масштабе (только чистые линии).
    recon = _render_geometry(ink.shape, px_traces, mm_px)
    recon_path = out_dir / "reconstruction.png"
    cv2.imwrite(str(recon_path), recon)

    # Матрица 12×10с: для отведения берём его ритм-строку (10с) если есть,
    # иначе блочную клетку (короче, дополняем NaN).
    mat = np.full((n_full, len(LEAD_ORDER)), np.nan, dtype=np.float32)
    for j, lead in enumerate(LEAD_ORDER):
        if lead in rhythm_sigs:
            mat[:, j] = rhythm_sigs[lead][:n_full]
        elif lead in signals:
            s = signals[lead]
            mat[: len(s), j] = s
    signal_npy = out_dir / "signal.npy"
    np.save(signal_npy, mat)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(12, 1, figsize=(12, 16))
    for ax, lead, j in zip(axs, LEAD_ORDER, range(12)):
        s = mat[:, j]
        ax.plot(np.arange(len(s)) / fs, np.nan_to_num(s), lw=0.7, color="black")
        cov = coverage.get(lead + "_rhythm", coverage.get(lead, 0.0))
        ax.set_ylabel(f"{lead}\ncov={cov:.0%}", rotation=0, labelpad=32, fontsize=9, va="center")
        ax.set_ylim(-2, 2.5)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("сек")
    fig.suptitle("ECG1.2 — независимая реконструкция по отведениям (demo, не медизделие)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    preview = out_dir / "preview.png"
    plt.savefig(preview, dpi=110)
    plt.close()

    manifest["signal_npy"] = str(signal_npy)
    manifest["preview"] = str(preview)
    manifest["reconstruction"] = str(recon_path)
    manifest["leads"] = LEAD_ORDER
    manifest["coverage"] = coverage
    manifest["vectorizer"] = "per-lead independent trace-following (own baseline/ROI, drift removal)"

    out_path = out_dir / "vectorize.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    lowcov = [l for l, c in coverage.items() if c < 0.5]
    log.info("STAGE %s: 12 отведений (независимо), низкое покрытие=%s", STAGE, lowcov or "нет")
    return str(out_path)
