"""Стадия preprocess: фото/скан/фото-экрана -> вырезанная область ЭКГ + чистая
бинарная маска трассы (работает на разных входах, в т.ч. бледных фото экрана).

Конвейер (стандарт из обработки документов/OCR):
  1. ЛОКАЛИЗАЦИЯ области ЭКГ: отделяем светлую ЭКГ-бумагу от тёмного окружения
     (рамка окна, меню, обои, поля) — Otsu + крупнейшая светлая компонента.
     Если тёмного окружения мало (скан) — берём весь кадр.
  2. Перспектива/поворот: если найден чёткий 4-угольник листа — warpPerspective.
  3. Свет/контраст: деление на размытый фон (убирает тени/градиент). CLAHE —
     АДАПТИВНО: только если трасса бледная (иначе на чистых раздувает сетку).
  4. Апскейл под рабочее разрешение.
  5. Маска трассы: color_ink (яркость + двухуровневый Otsu, цветоустойчиво).
     Сохраняем ink.png рядом с core_ready.

Выход: core_ready.png (цветной кроп — для mm/px и геометрии) и рядом ink.png
(бинарная трасса). Панель before_after.png — для глаза.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir, color_ink  # noqa: E402

MIN_INK_COV = 0.013   # ниже -> трасса бледная, включаем CLAHE

STAGE = "preprocess"
log = get_logger(STAGE)


def _order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def localize_ecg(bgr):
    """Вырезать область ЭКГ-бумаги (светлую) из тёмного окружения.

    Otsu делит кадр на тёмное (рамка/меню/обои) и светлое (бумага). Если тёмного
    заметно (>15% кадра) — берём крупнейшую светлую компоненту (это и есть ЭКГ).
    Иначе (скан/чистое фото) — весь кадр.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_frac = float((g < t).mean())
    if dark_frac < 0.15:
        return bgr, (0, 0, W, H)
    m = (g > t).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, W // 120),) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return bgr, (0, 0, W, H)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = st[big]
    if a < 0.2 * H * W:                      # компонента подозрительно мала — весь кадр
        return bgr, (0, 0, W, H)
    pad = int(0.004 * max(H, W))             # чуть внутрь, чтобы не тянуть кромку окна
    x0, y0 = max(0, x + pad), max(0, y + pad)
    x1, y1 = min(W, x + w - pad), min(H, y + h - pad)
    return bgr[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def _find_document_quad(bgr, min_area_ratio=0.5):
    h, w = bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((5, 5), np.uint8), 2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area_ratio * small.shape[0] * small.shape[1]:
            q = approx.reshape(4, 2).astype(np.float32)
            return q / scale if scale < 1 else q
    return None


def _warp(bgr, quad):
    rect = _order_points(quad)
    (tl, tr, br, bl) = rect
    mw = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    mh = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype=np.float32)
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(rect, dst), (mw, mh))


def _remove_shadows_color(bgr):
    """Выровнять освещение делением на размытый фон (фон — на уменьшенной копии)."""
    h, w = bgr.shape[:2]
    scale = min(1.0, 700.0 / max(h, w))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    sigma = max(small.shape[:2]) / 12.0
    out = np.zeros_like(bgr)
    for ch in range(3):
        b = cv2.GaussianBlur(small[:, :, ch], (0, 0), sigmaX=sigma, sigmaY=sigma)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR) if scale < 1 else b
        out[:, :, ch] = cv2.divide(bgr[:, :, ch], b, scale=255)
    return out


def _clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def sauvola_ink(gray, w=35, k=0.3, R=128.0):
    """Адаптивная бинаризация Sauvola: локальный порог на пиксель — плотно ловит
    БЛЕДНУЮ трассу (фото экрана/скан), где color_ink берёт слишком редко."""
    im = gray.astype(np.float32)
    w = w if w % 2 else w + 1
    mean = cv2.boxFilter(im, -1, (w, w))
    sq = cv2.boxFilter(im * im, -1, (w, w))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    T = mean * (1 + k * (std / R - 1))
    ink = (im < T).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return ink


def _upscale_to_width(bgr, target_w):
    h, w = bgr.shape[:2]
    if abs(w - target_w) < 2:
        return bgr
    return cv2.resize(bgr, (target_w, int(round(h * target_w / w))),
                      interpolation=cv2.INTER_CUBIC if w < target_w else cv2.INTER_AREA)


def _panel(stages, out_path):
    width = 900
    tiles = []
    for title, img in stages:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        tile = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        bar = np.full((30, width, 3), 255, np.uint8)
        cv2.putText(bar, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        tiles.append(np.vstack([bar, tile]))
    cv2.imwrite(str(out_path), np.vstack(tiles))


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    params = config.get("_stage_params", {})
    target_w = int(params.get("target_width", 2200))

    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"не удалось прочитать картинку: {input_path}")
    original = bgr.copy()

    # 1) локализация области ЭКГ (отрезаем тёмное окружение / хром окна)
    region, (rx, ry, rw, rh) = localize_ecg(bgr)
    log.info("STAGE %s: область ЭКГ %s из кадра %s", STAGE, (rw, rh), bgr.shape[:2])

    # 2) перспектива (если в РЕГИОНЕ есть чёткий лист)
    if bool(params.get("perspective", True)):
        quad = _find_document_quad(region)
        if quad is not None:
            region = _warp(region, quad)

    # 3) свет/контраст. Определяем «бледность» по покрытию color_ink на deshadow.
    desh = _remove_shadows_color(region)
    faint = float(color_ink(desh).mean()) < MIN_INK_COV
    final = _clahe(desh) if faint else desh    # CLAHE — только для бледных
    log.info("STAGE %s: трасса %s", STAGE, "бледная (Sauvola+CLAHE)" if faint else "чёткая (color_ink)")

    # 4) апскейл
    core = _upscale_to_width(final, target_w)
    core_path = out_dir / "core_ready.png"
    cv2.imwrite(str(core_path), core)

    # 5) маска трассы: бледная -> Sauvola (плотно), чёткая -> color_ink (чисто).
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    ink = sauvola_ink(gray) if faint else color_ink(core)
    cv2.imwrite(str(out_dir / "ink.png"), ink * 255)

    _panel([("1. original", original),
            ("2. region", region),
            ("3. contrast" + (" +CLAHE" if faint else ""), final),
            ("4. ink " + ("(Sauvola)" if faint else "(color)"), ink * 255)],
           out_dir / "before_after.png")
    log.info("STAGE %s: core_ready %s, ink=%.1f%% -> %s", STAGE, core.shape[:2],
             100 * float(ink.mean()), core_path)
    return str(core_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out_dir", default="/tmp/pp2_out")
    a = ap.parse_args()
    print(run(a.input, {"_run_dir": a.out_dir, "_stage_params": {}}))
