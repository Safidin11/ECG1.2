"""Безопасный драйвер поверх felixkrones/ECG-Digitiser (без правки их кода).

Проблема: их `src/run/digitize.py` вычисляет угол поворота через Hough
(`get_rotation_angle`) и, если сетка не даёт >=1200 голосов (типично для
реальных сканов низкого разрешения), возвращает NaN. Далее
`torchvision.rotate(image, NaN)` ОБНУЛЯЕТ картинку -> сегментация пустая ->
"Signal is empty". Для векторной синтетики Hough даёт ~199 линий и угол 0,
поэтому там всё работает, а на реальных фото — нет.

Решение: мы НЕ модифицируем их репозиторий, а импортируем их же функции и
повторяем их пайплайн один-в-один, меняя ровно одно: если угол NaN — берём
0.0 (фото мы уже выпрямляем в стадии preprocess). Опционально угол можно
задать жёстко через --force_angle.

Запускать интерпретатором ecgdig с cwd = каталог репо ECG-Digitiser
(нужно для `from config import ...` и `from src.run.digitize import ...`).
Логика векторизации/масштабирования/записи WFDB — их, дословно.
"""
import argparse
import os
import sys

import numpy as np
import torch
import wfdb
from torchvision.io.image import read_image
from torchvision.transforms.functional import rotate

sys.path.insert(0, os.getcwd())  # каталог репо ECG-Digitiser (cwd)

from config import (  # noqa: E402
    DATASET_NAME, IMAGE_TYPE, FREQUENCY, LONG_SIGNAL_LENGTH_SEC,
    SHORT_SIGNAL_LENGTH_SEC, Y_SHIFT_RATIO, SIGNAL_UNITS, LEAD_LABEL_MAPPING,
    FMT, ADC_GAIN, BASELINE,
)
from src.run.digitize import (  # noqa: E402
    get_rotation_angle, predict_mask_nnunet, cut_binary, vectorise,
    save_plot_masks_and_signals,
)


def digitize_one(image_path, model_folder, output_folder, force_angle=None,
                 show_image=True):
    os.makedirs(output_folder, exist_ok=True)
    record = os.path.basename(image_path).replace(f".{IMAGE_TYPE}", "")

    image = read_image(image_path)[:3]

    # --- ЕДИНСТВЕННОЕ отличие от их run(): защита от NaN-угла ---
    if force_angle is not None:
        rot_angle = float(force_angle)
    else:
        rot_angle = get_rotation_angle(image.permute(1, 2, 0).numpy().astype(np.uint8))
        if rot_angle is None or (isinstance(rot_angle, float) and np.isnan(rot_angle)):
            print("[safe] Hough не нашёл угол (NaN) -> использую 0.0 (фото уже выпрямлено)")
            rot_angle = 0.0
    image_rotated = rotate(image, rot_angle)

    # --- дальше их логика дословно ---
    mask_to_use = predict_mask_nnunet(image_rotated, DATASET_NAME, model_folder)

    # Дополнительно сохраняем СЫРУЮ маску nnU-Net (метки 0..12) выровненную с
    # входной картинкой. Наш пайплайн использует её как бинарную трассу + свою
    # раскладку (метки felixkrones на реальных фото ненадёжны).
    import cv2 as _cv2
    _m = mask_to_use.numpy()
    if _m.ndim == 3:
        _m = _m[0]
    _cv2.imwrite(os.path.join(output_folder, f"{record}_mask.png"), _m.astype(np.uint8))

    signal_masks_cropped, signal_positions_cropped, _ = cut_binary(mask_to_use, image_rotated)

    x_pixel_list = [v.shape[2] for v in signal_masks_cropped.values() if v is not None]
    if len(x_pixel_list) == 0:
        raise ValueError(f"Signal is empty for record {record} (маска пуста даже после фикса угла)")
    x_pixel_list_median = np.median(x_pixel_list)
    x_pixel_list_below_2x_median_mean = np.mean(
        [v for v in x_pixel_list if v < 2 * x_pixel_list_median]
    )
    sec_per_pixel = 2.5 / x_pixel_list_below_2x_median_mean
    mm_per_pixel = 25 * sec_per_pixel
    sec_per_pixel = mm_per_pixel / 25
    mV_per_pixel = mm_per_pixel / 10

    signals_predicted = {}
    for lead, mask in signal_masks_cropped.items():
        if mask is not None:
            signals_predicted[lead] = vectorise(
                image_rotated, mask, signal_positions_cropped[lead]["y1"],
                sec_per_pixel, mV_per_pixel, Y_SHIFT_RATIO, lead,
            )
        else:
            signals_predicted[lead] = None

    signals = {
        name: signals_predicted[name].numpy()
        for name in LEAD_LABEL_MAPPING.keys()
        if signals_predicted[name] is not None
    }
    num_samples = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)
    signal_list = []
    for signal in signals.values():
        if len(signal) < num_samples:
            nan_signal = np.empty(num_samples)
            nan_signal[:] = np.nan
            nan_signal[: int(len(signal))] = signal
            signal_list.append(nan_signal)
        else:
            signal_list.append(signal)
    sig_names = list(signals.keys())
    signals = np.array(signal_list).T

    if signals.shape[0] == 0:
        raise ValueError(f"Signal is empty for record {record}.")

    if show_image:
        save_plot_masks_and_signals(
            image_rotated, signal_masks_cropped, signal_positions_cropped,
            signals, sig_names, output_folder, f"{record}.png",
        )

    if (np.nanmax(signals) > 10) or (np.nanmin(signals) < -10):
        max_val, min_val = np.nanmax(signals), np.nanmin(signals)
        signals = (signals - min_val) / (max_val - min_val) * 2 - 1

    wfdb.wrsamp(
        record, fs=FREQUENCY, units=[SIGNAL_UNITS] * signals.shape[1],
        sig_name=sig_names, p_signal=np.nan_to_num(signals),
        write_dir=output_folder, fmt=[FMT] * signals.shape[1],
        adc_gain=[ADC_GAIN] * signals.shape[1], baseline=[BASELINE] * signals.shape[1],
    )
    print(f"[safe] сигнал {signals.shape}, отведения={sig_names}, угол={rot_angle}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_folder", required=True)
    ap.add_argument("-m", "--model_folder", required=True)
    ap.add_argument("-o", "--output_folder", required=True)
    ap.add_argument("--force_angle", type=float, default=None,
                    help="жёстко задать угол поворота (град); по умолчанию Hough с фиксом NaN->0")
    args = ap.parse_args()

    imgs = [f for f in os.listdir(args.data_folder) if f.endswith(f".{IMAGE_TYPE}")]
    for f in imgs:
        digitize_one(os.path.join(args.data_folder, f), args.model_folder,
                     args.output_folder, force_angle=args.force_angle)


if __name__ == "__main__":
    main()
