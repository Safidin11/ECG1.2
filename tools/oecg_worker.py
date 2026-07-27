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
PAPER = (255, 255, 255)
GRID_FINE = (213, 205, 247)      # светло-розовый (1 мм)
GRID_BOLD = (170, 156, 233)      # насыщенный розовый (5 мм)
TRACE = (17, 17, 17)


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


def draw_twin(lines, shape, mm_px_x: float, mm_px_y: float, out_path: str,
              grid: bool = True, lead_rows=None, cols: int = 1,
              rhythm_name: str = "II") -> None:
    """Нарисовать линии на холсте того же размера, что обработанный снимок.

    lines — (n_линий, ширина): для каждого столбца x положение линии y в
    пикселях (NaN там, где линия не найдена). Рисуем ровно по этим координатам,
    ничего не масштабируя и не раздвигая.
    lead_rows — сетка имён отведений (строки × колонки) для подписей.
    """
    import cv2
    H, W = shape
    canvas = np.full((H, W, 3), PAPER, np.uint8)

    if grid and mm_px_x > 0 and mm_px_y > 0:
        step_x, step_y = 1.0 / mm_px_x, 1.0 / mm_px_y      # 1 мм в пикселях
        if 2 <= step_x <= 60 and 2 <= step_y <= 60:        # иначе масштаб не распознан
            for k in range(int(W / step_x) + 1):
                x = int(round(k * step_x))
                cv2.line(canvas, (x, 0), (x, H), GRID_BOLD if k % 5 == 0 else GRID_FINE, 1)
            for k in range(int(H / step_y) + 1):
                y = int(round(k * step_y))
                cv2.line(canvas, (0, y), (W, y), GRID_BOLD if k % 5 == 0 else GRID_FINE, 1)

    arr = lines.cpu().numpy() if hasattr(lines, "cpu") else np.asarray(lines)
    if arr.ndim == 1:
        arr = arr[None, :]
    # Экстрактор отдаёт линии в произвольном порядке (первой может прийти нижняя),
    # а подписи раскладки идут сверху вниз — сортируем по высоте.
    order = np.argsort([np.nanmedian(r) if np.any(~np.isnan(r)) else np.inf for r in arr])
    arr = arr[order]
    for row in arr:
        pts, run = [], []
        for x in range(min(len(row), W)):
            y = row[x]
            if np.isnan(y):
                if len(run) > 1:
                    pts.append(run)
                run = []
            else:
                run.append((x, int(round(float(y)))))
        if len(run) > 1:
            pts.append(run)
        for seg in pts:                                    # разрывы не соединяем
            cv2.polylines(canvas, [np.array(seg, np.int32)], False, TRACE, 2, cv2.LINE_AA)

    # Подписи отведений — как на распечатке: имя над началом своего отрезка.
    if lead_rows:
        font, scale, th = cv2.FONT_HERSHEY_DUPLEX, max(0.45, H / 1400), 1
        for i, row in enumerate(arr):
            names = lead_rows[i] if i < len(lead_rows) else [rhythm_name]
            ncol = len(names) if i < len(lead_rows) else 1
            for c, nm in enumerate(names):
                x0, x1 = int(W * c / ncol), int(W * (c + 1) / ncol)
                seg = row[x0:min(x1, len(row))]
                good = seg[~np.isnan(seg)]
                if len(good) < 5:
                    continue
                y = int(np.median(good)) - int(0.035 * H)
                y = max(int(0.02 * H), min(H - 5, y))
                cv2.putText(canvas, nm, (x0 + max(6, int(0.006 * W)), y),
                            font, scale, TRACE, th, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def save_aligned(img_tensor, out_path: str) -> tuple[int, int]:
    """Обработанный снимок (после перспективы и обрезки) — как эталон для сравнения."""
    import cv2
    arr = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    if arr.max() <= 1.001:
        arr = arr * 255.0
    bgr = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    return bgr.shape[0], bgr.shape[1]


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

    H, W = save_aligned(got["aligned"]["image"], a.out_base + "_aligned.png")

    # Их же диагностическая картинка из 4 панелей — «все этапы движка».
    try:
        from src.digitize import save_png_plot
        save_png_plot(got, sig.get("canonical_lines"), a.out_base + "_stages")
    except Exception as exc:
        print(f"[twin] панель этапов не построена: {exc}")

    # ВАЖНО: берём raw_lines — это выход экстрактора в ПИКСЕЛЬНЫХ координатах
    # обработанного снимка. Поле lines после привязки к отведениям уже
    # пересчитано в другие единицы и для геометрии не годится.
    lines = sig.get("raw_lines")
    if lines is not None:
        ps = got.get("pixel_spacing_mm", {})
        rows, cols, _ = lead_grid(got.get("layout_name", ""), a.layouts) if a.layouts \
            else (None, 1, 0)
        draw_twin(lines, (H, W), float(ps.get("x", 0) or 0), float(ps.get("y", 0) or 0),
                  a.out_base + "_twin.png", lead_rows=rows, cols=cols)
        print(f"[twin] цифровая копия {W}x{H}, линий={len(lines)}, подписи={'да' if rows else 'нет'}")
    else:
        print("[twin] линий нет — копия не построена")


if __name__ == "__main__":
    main()
