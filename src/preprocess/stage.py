"""Стадия preprocess: реальное фото/скан ЭКГ -> нормализованная картинка под ядро.

Цель: привести реальное фото к виду, максимально близкому к синтетике, на
которой обучены веса felixkrones (RGB-распечатка: белый фон, цветная сетка,
чёрная кривая, ~25 мм/с, 10 мм/мВ, достаточное разрешение).

Шаги (OpenCV):
  1. градации серого для детекции границ листа;
  2. поиск 4 углов листа + коррекция перспективы
     (getPerspectiveTransform + warpPerspective);
  3. выравнивание освещения / удаление теней — деление на размытый фон
     (cv2.divide), поканально, чтобы сохранить цвет сетки; adaptiveThreshold
     используется для отладочной панели «до/после»;
  4. лёгкая обрезка белых полей;
  5. апскейл до целевой ширины (nnU-Net обучен на большем разрешении).

Вход:  путь к фото (png/jpg).
Выход: путь к core_ready.png (RGB) — его получает segment. Рядом кладётся
       панель before_after.png для глазной проверки.

Мягкая деградация: если 4 угла не найдены — работаем по полному кадру.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir  # noqa: E402

STAGE = "preprocess"
log = get_logger(STAGE)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочить 4 точки: tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left (min x+y)
    rect[2] = pts[np.argmax(s)]      # bottom-right (max x+y)
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]      # top-right (min y-x)
    rect[3] = pts[np.argmax(d)]      # bottom-left
    return rect


def _find_document_quad(bgr: np.ndarray, min_area_ratio: float = 0.35):
    """Найти четырёхугольник листа. -> (4,2) в координатах оригинала или None."""
    h, w = bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    small_area = small.shape[0] * small.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area_ratio * small_area:
            quad = approx.reshape(4, 2).astype(np.float32)
            if scale < 1:
                quad /= scale
            return quad
    return None


def _four_point_transform(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = _order_points(quad)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (maxW, maxH))


def _remove_shadows_color(bgr: np.ndarray) -> np.ndarray:
    """Выровнять освещение поканально делением на размытый фон (сохраняет цвет)."""
    h, w = bgr.shape[:2]
    sigma = max(h, w) / 12.0
    out = np.zeros_like(bgr)
    for ch in range(3):
        blur = cv2.GaussianBlur(bgr[:, :, ch], (0, 0), sigmaX=sigma, sigmaY=sigma)
        out[:, :, ch] = cv2.divide(bgr[:, :, ch], blur, scale=255)
    return out


def _adaptive_binary(bgr: np.ndarray) -> np.ndarray:
    """Отладочная бинаризация (adaptiveThreshold) — только для панели «до/после»."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)


def _autocrop_white(bgr: np.ndarray, thr: int = 245, pad: int = 8) -> np.ndarray:
    """Обрезать однородные белые поля вокруг контента."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray < thr
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return bgr
    x, y, w, h = cv2.boundingRect(coords)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
    return bgr[y0:y1, x0:x1]


def _upscale_to_width(bgr: np.ndarray, target_w: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w >= target_w:
        return bgr
    scale = target_w / w
    return cv2.resize(bgr, (target_w, int(round(h * scale))), interpolation=cv2.INTER_CUBIC)


def _make_panel(stages: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    """Собрать вертикальную панель «до/после» с подписями."""
    width = 900
    tiles = []
    for title, img in stages:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        tile = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        bar = np.full((32, width, 3), 255, np.uint8)
        cv2.putText(bar, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        tiles.append(np.vstack([bar, tile]))
    cv2.imwrite(str(out_path), np.vstack(tiles))


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    params = config.get("_stage_params", {})
    target_w = int(params.get("target_width", 2200))
    do_perspective = bool(params.get("perspective", True))

    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"не удалось прочитать картинку: {input_path}")
    original = bgr.copy()

    # 2) перспектива
    warped = bgr
    if do_perspective:
        quad = _find_document_quad(bgr)
        if quad is not None:
            warped = _four_point_transform(bgr, quad)
            log.info("STAGE %s: найден лист, коррекция перспективы %s -> %s",
                     STAGE, bgr.shape[:2], warped.shape[:2])
        else:
            log.info("STAGE %s: чёткий 4-угольник не найден — работаю по полному кадру", STAGE)

    # 3) удаление теней (цветное) + отладочная бинаризация
    deshadow = _remove_shadows_color(warped)
    binary = _adaptive_binary(warped)

    # 4) обрезка полей, 5) апскейл
    cropped = _autocrop_white(deshadow)
    core_ready = _upscale_to_width(cropped, target_w)

    core_path = out_dir / "core_ready.png"
    cv2.imwrite(str(core_path), core_ready)

    _make_panel(
        [("1. original", original),
         ("2. deshadow (cv2.divide)", deshadow),
         ("3. adaptiveThreshold (debug)", binary),
         ("4. core_ready (upscaled)", core_ready)],
        out_dir / "before_after.png",
    )
    log.info("STAGE %s: core_ready %s -> %s", STAGE, core_ready.shape[:2], core_path)
    return str(core_path)


if __name__ == "__main__":
    # Быстрый автономный прогон без пайплайна (для визуальной проверки).
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out_dir", default="/tmp/pp_out")
    ap.add_argument("--target_width", type=int, default=2200)
    a = ap.parse_args()
    cfg = {"_run_dir": a.out_dir, "_stage_params": {"target_width": a.target_width}}
    print(run(a.input, cfg))
