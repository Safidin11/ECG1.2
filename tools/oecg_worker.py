"""Обработчик поверх Open-ECG-Digitizer: их пайплайн + цифровой двойник снимка.

Их `src.digitize` сохраняет только сигнал и диагностическую картинку из 4
панелей. Нам нужно ещё одно: ЦИФРОВАЯ КОПИЯ обработанной фотографии — тот же
размер, те же места линий, та же высота зубцов, но линия нарисована по
оцифрованным данным. Такую копию можно наложить на снимок, и всё совпадёт.

Всё берётся из ОДНОГО прогона сети: их `forward()` уже возвращает
  aligned.image   — снимок после исправления перспективы и обрезки,
  signal.raw_lines — линии в ПИКСЕЛЬНЫХ координатах этого снимка,
  pixel_spacing_mm — масштаб.
Поэтому второй прогон (и лишние полторы минуты) не нужен.

Мы НЕ правим их код: импортируем их же классы и повторяем их логику сохранения
CSV дословно.

Запускать интерпретатором движка с cwd = каталог движка.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from src.config.default import get_cfg                          # noqa: E402
from src.utils import find_config_path, import_class_from_path  # noqa: E402
from torchvision.io import decode_image                         # noqa: E402

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Цвета копии: белая бумага, розовая сетка, чёрная трасса — как на распечатке.
# OpenCV работает в BGR, поэтому порядок каналов обратный привычному RGB.
PAPER = (252, 250, 250)
GRID_FINE = (196, 186, 240)      # розовый, 1 мм
GRID_BOLD = (140, 120, 224)      # насыщенный розовый, 5 мм
TRACE = (12, 12, 12)
LABEL = (40, 30, 25)             # подписи отведений
SEP = (150, 140, 150)            # границы между отведениями


def save_csv(canonical, base: str) -> None:
    """Дословно их формат: (отсчёты, отведения), заголовок — имена отведений."""
    if canonical is None:
        return
    data = canonical.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    header = ",".join(LEAD_NAMES[:data.shape[0]])
    np.savetxt(base + "_timeseries_canonical.csv", data.T, delimiter=",",
               header=header, comments="")


def lead_grid(layout_name: str, layouts_path: str):
    """Сетка отведений выбранной раскладки: список строк + число колонок."""
    import yaml
    try:
        layouts = yaml.safe_load(open(layouts_path, encoding="utf-8")) or {}
    except Exception:
        return None, 1, 0
    lay = layouts.get(layout_name)
    if not lay:
        return None, 1, 0
    grid = [[r] if isinstance(r, str) else list(r) for r in lay.get("leads", [])]
    cols = int(lay.get("layout", {}).get("cols", 1))
    return grid, cols, len(lay.get("rhythm_leads") or [])


def _sorted_lines(lines) -> np.ndarray:
    """Линии сверху вниз. Экстрактор отдаёт их в произвольном порядке, а подписи
    раскладки идут сверху вниз."""
    arr = lines.cpu().numpy() if hasattr(lines, "cpu") else np.asarray(lines)
    if arr.ndim == 1:
        arr = arr[None, :]
    order = np.argsort([np.nanmedian(r) if np.any(~np.isnan(r)) else np.inf for r in arr])
    return arr[order]


def content_box(arr: np.ndarray, shape, mm_px_x: float, mm_px_y: float):
    """Прямоугольник, который реально занят трассами, плюс поля под оформление.

    Нужен потому, что после исправления перспективы по краям холста остаются
    пустые чёрные поля, и картинка выглядит смещённой. Обрезаем и снимок, и
    копию ОДИНАКОВО — взаимная геометрия не меняется, наложение по-прежнему
    совпадает, просто уходит перекошенная пустота.
    """
    H, W = shape
    xs = np.flatnonzero(np.any(~np.isnan(arr), axis=0))
    ys = arr[~np.isnan(arr)]
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, W, H
    # Поля считаем в СВОИХ миллиметрах по каждой оси: масштаб по x и y разный,
    # и калибр-импульс шириной 9 мм не влезал, когда поле мерили по вертикали.
    mmx = (1.0 / mm_px_x) if mm_px_x > 0 else W / 250.0
    mm = (1.0 / mm_px_y) if mm_px_y > 0 else H / 60.0
    pad_left = int(round(14 * mmx))                       # место под калибр-импульс
    pad_side = int(round(5 * mmx))
    pad_top = int(round(9 * mm))                          # место под подписи
    # Границы НЕ ограничиваем размером снимка: если содержимое начинается у
    # самого края, поле под калибр-импульс просто добирается пустым — иначе
    # импульс некуда рисовать.
    return (int(xs[0]) - pad_left, int(ys.min()) - pad_top,
            int(xs[-1]) + pad_side, int(ys.max()) + pad_side)


def draw_twin(arr: np.ndarray, box, shape, mm_px_x: float, mm_px_y: float,
              out_path: str, lead_rows=None, rhythm_name: str = "II") -> None:
    """Нарисовать цифровую копию: трассы ровно на своих координатах + оформление.

    Координаты трасс НЕ меняются — только вычитается смещение обрезки, которое
    точно так же применяется к снимку. Всё остальное (сетка, подписи, границы
    колонок, калибр-импульс) — оформление поверх.
    """
    import cv2
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    canvas = np.full((H, W, 3), PAPER, np.uint8)
    mm_x = (1.0 / mm_px_x) if mm_px_x > 0 else 0.0        # 1 мм по горизонтали, px
    mm_y = (1.0 / mm_px_y) if mm_px_y > 0 else 0.0

    # --- миллиметровка ---
    if 2 <= mm_x <= 60 and 2 <= mm_y <= 60:
        for k in range(int(W / mm_x) + 1):
            x = int(round(k * mm_x))
            bold = k % 5 == 0
            cv2.line(canvas, (x, 0), (x, H), GRID_BOLD if bold else GRID_FINE, 2 if bold else 1)
        for k in range(int(H / mm_y) + 1):
            y = int(round(k * mm_y))
            bold = k % 5 == 0
            cv2.line(canvas, (0, y), (W, y), GRID_BOLD if bold else GRID_FINE, 2 if bold else 1)

    # --- где начинается и кончается содержимое (для колонок и импульса) ---
    xs = np.flatnonzero(np.any(~np.isnan(arr), axis=0))
    cx0, cx1 = (int(xs[0]) - x0, int(xs[-1]) - x0) if len(xs) else (0, W)

    ncols = max((len(r) for r in lead_rows), default=1) if lead_rows else 1

    def runs_of(row):
        """Непрерывные куски линии: на плёнке между отведениями есть разрыв,
        поэтому куски и есть колонки. Мелкие обрывки от сбоев распознавания
        отбрасываем по длине."""
        ok = ~np.isnan(row)
        if not ok.any():
            return []
        d = np.diff(ok.astype(np.int8))
        starts = list(np.flatnonzero(d == 1) + 1)
        ends = list(np.flatnonzero(d == -1) + 1)
        if ok[0]:
            starts = [0] + starts
        if ok[-1]:
            ends = ends + [len(row)]
        span = max(1, len(row) // (ncols * 6))
        return [(a, b) for a, b in zip(starts, ends) if b - a >= span]

    col_w = (cx1 - cx0) / max(ncols, 1)      # запасной вариант

    # --- границы между отведениями: по разрывам, усреднённым по строкам ---
    seps = []
    if ncols > 1:
        per_row = [runs_of(r) for r in arr]
        good = [rr for rr in per_row if len(rr) == ncols]
        if good:
            for c in range(1, ncols):
                seps.append(int(round(np.mean([(rr[c - 1][1] + rr[c][0]) / 2
                                               for rr in good]) - x0)))
        else:
            seps = [int(round(cx0 + c * col_w)) for c in range(1, ncols)]
        for x in seps:                      # пунктир, как на настоящей плёнке
            y = int(0.02 * H)
            dash = max(6, int(H / 90))
            while y < H - int(0.02 * H):
                cv2.line(canvas, (x, y), (x, min(y + dash, H)), SEP, 2, cv2.LINE_AA)
                y += dash * 2

    # --- трассы: ровно там, где они на снимке ---
    for row in arr:
        pts, run = [], []
        for x in range(len(row)):
            xc = x - x0
            if not (0 <= xc < W):
                continue
            y = row[x]
            if np.isnan(y):
                if len(run) > 1:
                    pts.append(run)
                run = []
            else:
                run.append((xc, int(round(float(y))) - y0))
        if len(run) > 1:
            pts.append(run)
        for seg in pts:                                    # разрывы не соединяем
            cv2.polylines(canvas, [np.array(seg, np.int32)], False, TRACE, 2, cv2.LINE_AA)

    # --- калибр-импульс 1 мВ (10 мм) в начале каждой строки ---
    if mm_x > 0 and mm_y > 0:
        for row in arr:
            good = row[~np.isnan(row)]
            if len(good) < 20:
                continue
            base = int(np.median(good)) - y0
            top = base - int(round(10 * mm_y))             # 1 мВ при 10 мм/мВ
            xa = cx0 - int(round(9 * mm_x))
            xb = cx0 - int(round(4 * mm_x))
            if xa < 1 or not (0 < top < H):
                continue
            cv2.polylines(canvas, [np.array(
                [(xa - int(2 * mm_x), base), (xa, base), (xa, top), (xb, top),
                 (xb, base), (xb + int(2 * mm_x), base)], np.int32)],
                False, TRACE, 2, cv2.LINE_AA)

    # --- подписи отведений над началом своего отрезка ---
    if lead_rows:
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = max(0.55, H / 1000)
        th = max(1, int(round(scale * 1.4)))
        for i, row in enumerate(arr):
            names = lead_rows[i] if i < len(lead_rows) else [rhythm_name]
            n = len(names)
            rr = runs_of(row)
            if len(rr) == n:                       # куски совпали с числом отведений
                bounds = [(a - x0, b - x0) for a, b in rr]
            else:                                  # иначе — по общим границам
                edges = [cx0] + seps + [cx1] if len(seps) == n - 1 else \
                    [int(round(cx0 + k * (cx1 - cx0) / n)) for k in range(n + 1)]
                bounds = [(edges[k], edges[k + 1]) for k in range(n)]
            for c, nm in enumerate(names):
                xs0, xs1 = bounds[c]
                seg = row[max(xs0 + x0, 0):min(xs1 + x0, len(row))]
                good = seg[~np.isnan(seg)]
                if len(good) < 5:
                    continue
                # высота подписи — от базовой линии ВСЕЙ строки, чтобы имена
                # шли ровно, а не прыгали за каждым отрезком
                base_row = np.nanmedian(row)
                y = int(base_row) - y0 - int(round(9 * mm_y if mm_y else 0.06 * H))
                y = max(int(scale * 22), min(H - 4, y))
                org = (xs0 + max(4, int(1.5 * mm_x)), y)
                cv2.putText(canvas, nm, org, font, scale, PAPER, th + 3, cv2.LINE_AA)
                cv2.putText(canvas, nm, org, font, scale, LABEL, th, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def crop_pad(img: np.ndarray, box, fill=(250, 250, 250)) -> np.ndarray:
    """Вырезать прямоугольник, который может выходить за границы картинки:
    недостающие поля добираются заливкой. Так снимок и копия остаются одного
    размера и совмещаются."""
    x0, y0, x1, y1 = box
    out = np.full((y1 - y0, x1 - x0, 3), fill, np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def aligned_bgr(img_tensor) -> np.ndarray:
    """Обработанный снимок (после исправления перспективы) в BGR."""
    import cv2
    arr = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    if arr.max() <= 1.001:
        arr = arr * 255.0
    return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-o", "--out_base", required=True, help="префикс выходных файлов")
    ap.add_argument("--layouts", default=None, help="наш configs/oecg_layouts.yml для подписей")
    a = ap.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(find_config_path(a.config))
    wrapper = import_class_from_path(cfg.MODEL.class_path)(**cfg.MODEL.KWARGS)

    image = decode_image(a.image, mode="RGB").unsqueeze(0)
    got = wrapper(image, layout_should_include_substring=None)

    sig = got.get("signal", {})
    save_csv(sig.get("canonical_lines"), a.out_base)

    # метаданные раскладки — в их же формате
    name = os.path.basename(a.out_base)
    with open(os.path.join(os.path.dirname(a.out_base), "digitization_metadata.csv"),
              "w", encoding="utf-8") as f:
        f.write("file_path,matching_cost,is_flipped,lead_layout\n")
        f.write(f'{name},{sig.get("layout_matching_cost", 1.0)},'
                f'{sig.get("layout_is_flipped", "False")},{got.get("layout_name", "Unknown layout")}\n')

    photo = aligned_bgr(got["aligned"]["image"])
    H, W = photo.shape[:2]

    # Их же диагностическая картинка из 4 панелей — «все этапы движка».
    try:
        from src.digitize import save_png_plot
        save_png_plot(got, sig.get("canonical_lines"), a.out_base + "_stages")
    except Exception as exc:
        print(f"[twin] панель этапов не построена: {exc}")

    # ВАЖНО: берём raw_lines — это выход экстрактора в ПИКСЕЛЬНЫХ координатах
    # обработанного снимка. Поле lines после привязки к отведениям уже
    # пересчитано в другие единицы и для геометрии не годится.
    import cv2
    lines = sig.get("raw_lines")
    if lines is not None:
        ps = got.get("pixel_spacing_mm", {})
        mmx, mmy = float(ps.get("x", 0) or 0), float(ps.get("y", 0) or 0)
        rows, _, _ = lead_grid(got.get("layout_name", ""), a.layouts) if a.layouts \
            else (None, 1, 0)
        arr = _sorted_lines(lines)
        box = content_box(arr, (H, W), mmx, mmy)
        draw_twin(arr, box, (H, W), mmx, mmy, a.out_base + "_twin.png", lead_rows=rows)
        cv2.imwrite(a.out_base + "_aligned.png", crop_pad(photo, box))
        print(f"[twin] копия {box[2]-box[0]}x{box[3]-box[1]}, линий={len(arr)}, "
              f"подписи={'да' if rows else 'нет'}")
    else:
        cv2.imwrite(a.out_base + "_aligned.png", photo)
        print("[twin] линий нет — копия не построена")


if __name__ == "__main__":
    main()
