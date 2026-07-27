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


def draw_twin(lines, shape, mm_px_x: float, mm_px_y: float, out_path: str,
              grid: bool = True) -> None:
    """Нарисовать линии на холсте того же размера, что обработанный снимок.

    lines — (n_линий, ширина): для каждого столбца x положение линии y в
    пикселях (NaN там, где линия не найдена). Рисуем ровно по этим координатам,
    ничего не масштабируя и не раздвигая.
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

    # ВАЖНО: берём raw_lines — это выход экстрактора в ПИКСЕЛЬНЫХ координатах
    # обработанного снимка. Поле lines после привязки к отведениям уже
    # пересчитано в другие единицы и для геометрии не годится.
    lines = sig.get("raw_lines")
    if lines is not None:
        ps = got.get("pixel_spacing_mm", {})
        draw_twin(lines, (H, W), float(ps.get("x", 0) or 0), float(ps.get("y", 0) or 0),
                  a.out_base + "_twin.png")
        print(f"[twin] цифровая копия {W}x{H}, линий={len(lines)}")
    else:
        print("[twin] линий нет — копия не построена")


if __name__ == "__main__":
    main()
