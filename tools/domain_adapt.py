"""Домен-адаптация: реальное фото ЭКГ -> «синтетический вид» ecg-image-kit.

Цель — привести вход к распределению, на котором обучена nnU-Net felixkrones:
светлая розоватая бумага + КРАСНАЯ сетка (1мм тонкая + 5мм жирная) + тонкая
ЧЁРНАЯ трасса, ~2200px по ширине, плоский свет.

Ключевой момент: НЕ бинаризуем трассу (иначе теряем бледные грудные). Считаем
непрерывную «прозрачность чернил» alpha из нормализованной яркости и
композитим чёрную трассу поверх синтетической сетки. Так модель видит
естественные тонкие линии, включая слабые.

Вход: core_ready.png (наш deshadow+CLAHE кроп). Выход: PNG «под синтетику».
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def _norm_gray(bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Яркость + робастные уровни бумаги (светлое) и трассы (тёмное)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper = float(np.percentile(gray, 90))     # фон бумаги
    dark = float(np.percentile(gray, 2))       # самые тёмные чернила
    return gray, paper, dark


def _ink_alpha(gray: np.ndarray, paper: float, dark: float, gamma: float = 1.4) -> np.ndarray:
    """Непрерывная прозрачность чернил 0..1 (1 = чёрная трасса)."""
    span = max(paper - dark, 1.0)
    a = (paper - gray) / span
    a = np.clip(a, 0.0, 1.0) ** gamma          # gamma>1 подавляет слабый фон-шум сетки
    return a


def _draw_grid(shape, mm_px: float) -> np.ndarray:
    """Синтетическая красная сетка на розоватой бумаге (BGR)."""
    h, w = shape
    img = np.full((h, w, 3), (238, 238, 255), np.uint8)   # розовато-белая бумага
    fine = (205, 205, 255)     # 1мм — светло-красная
    bold = (150, 150, 255)     # 5мм — насыщенная красная
    step = max(mm_px, 3.0)
    # тонкая сетка
    x = 0.0
    while x < w:
        cv2.line(img, (int(round(x)), 0), (int(round(x)), h), fine, 1)
        x += step
    y = 0.0
    while y < h:
        cv2.line(img, (0, int(round(y))), (w, int(round(y))), fine, 1)
        y += step
    # жирная каждые 5мм
    x = 0.0
    while x < w:
        cv2.line(img, (int(round(x)), 0), (int(round(x)), h), bold, 1)
        x += step * 5
    y = 0.0
    while y < h:
        cv2.line(img, (0, int(round(y))), (w, int(round(y))), bold, 1)
        y += step * 5
    return img


def _clean_ink(ink: np.ndarray) -> np.ndarray:
    """Из сырой бинарной маски -> чистая тонкая трасса: убрать точки сетки/мелочь,
    оставить вытянутые компоненты (трасса), слегка утоньшить."""
    m = (ink > 0).astype(np.uint8)
    # закрыть разрывы вдоль трассы (горизонтально), убрать одиночные точки сетки
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        w_i, h_i, a_i = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        if a_i >= 40 and w_i >= 8:             # вытянутые/крупные -> трасса и подписи
            keep[lbl == i] = 1
    return keep


def adapt(bgr: np.ndarray, mm_px: float | None = None, target_w: int = 2200,
          ink: np.ndarray | None = None) -> np.ndarray:
    """Реальное фото -> синтетический вид для nnU-Net.

    ink задан -> рисуем криспную ЧЁРНУЮ трассу из бинарной маски (устойчиво к
    бледным фото). Иначе -> непрерывная alpha из яркости.
    """
    if ink is not None and ink.shape[:2] != bgr.shape[:2]:
        ink = cv2.resize(ink, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if abs(bgr.shape[1] - target_w) > 2:
        h = int(round(bgr.shape[0] * target_w / bgr.shape[1]))
        interp = cv2.INTER_CUBIC if bgr.shape[1] < target_w else cv2.INTER_AREA
        if ink is not None:
            ink = cv2.resize(ink, (target_w, h), interpolation=cv2.INTER_NEAREST)
        bgr = cv2.resize(bgr, (target_w, h), interpolation=interp)
    h, w = bgr.shape[:2]
    if mm_px is None:
        mm_px = w / 250.0                      # 10с * 25мм/с = 250мм на всю ширину контента

    if ink is not None:
        trace = _clean_ink(ink).astype(np.float32)
        trace = cv2.GaussianBlur(trace, (0, 0), 0.7)     # антиалиасинг как у синтетики
        alpha = np.clip(trace, 0, 1)[..., None]
    else:
        gray, paper, dark = _norm_gray(bgr)
        alpha = _ink_alpha(gray, paper, dark)[..., None]

    grid = _draw_grid((h, w), mm_px).astype(np.float32)
    black = np.zeros_like(grid)
    out = grid * (1.0 - alpha) + black * alpha           # чёрная трасса поверх сетки
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="core_ready.png")
    ap.add_argument("-o", "--output", required=True, help="синтетический PNG")
    ap.add_argument("--ink", default=None, help="ink.png (криспная трасса из маски)")
    ap.add_argument("--mm_px", type=float, default=None)
    a = ap.parse_args()
    bgr = cv2.imread(a.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"не прочитать: {a.input}")
    ink = cv2.imread(a.ink, cv2.IMREAD_GRAYSCALE) if a.ink else None
    out = adapt(bgr, mm_px=a.mm_px, ink=ink)
    cv2.imwrite(a.output, out)
    print(f"[domain_adapt] {bgr.shape[:2]} -> {out.shape[:2]} -> {a.output}")


if __name__ == "__main__":
    main()
