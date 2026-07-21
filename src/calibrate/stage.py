"""Стадия calibrate: поднять/зафиксировать калибровку, применённую ядром.

felixkrones/ECG-Digitiser выполняет калибровку ВНУТРИ digitize.py:
  - поворот изображения по Hough Transform (выравнивание сетки),
  - пересчёт пикселей в мВ/секунды по стандарту 25 мм/с и 10 мм/мВ,
  - ресемплинг до FREQUENCY=500 Гц.
Эти параметры не выводятся отдельным файлом, поэтому здесь мы фиксируем
номинальную калибровку ядра и статистику сигнала в манифесте для последующих
стадий (vectorize/reconstruct). Это НЕ повторный вызов ядра — работа лёгкая,
поэтому стадия выполняется в core-venv (не подпроцесс).

Вход:  манифест segment.json.
Выход: манифест calibrate.json (segment + блок calibration).

Источник калибровочной логики: https://github.com/felixkrones/ECG-Digitiser
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir  # noqa: E402

STAGE = "calibrate"
log = get_logger(STAGE)

# Стандарт бумажной ЭКГ, на который опирается ядро felixkrones.
PAPER_MM_PER_SEC = 25.0
PAPER_MM_PER_MV = 10.0


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)

    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Если предыдущая стадия деградировала и передала не манифест, а картинку —
    # мягко выходим passthrough-логикой (пусть решает оркестратор).
    if "signal_npy" not in manifest:
        raise RuntimeError("calibrate ожидал манифест segment.json, получил другое")

    signal = np.load(manifest["signal_npy"])
    fs = manifest.get("fs", 500)

    calibration = {
        "mm_per_sec": PAPER_MM_PER_SEC,
        "mm_per_mV": PAPER_MM_PER_MV,
        "fs_hz": fs,
        "units": "mV",
        "applied_by": "felixkrones core (Hough rotation + fixed paper scale)",
        "signal_stats": {
            "min": float(np.nanmin(signal)),
            "max": float(np.nanmax(signal)),
            "mean": float(np.nanmean(signal)),
        },
    }
    manifest["calibration"] = calibration

    out_path = out_dir / "calibrate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("STAGE %s: fs=%dГц, 25мм/с, 10мм/мВ, диапазон [%.2f..%.2f] мВ -> %s",
             STAGE, fs, calibration["signal_stats"]["min"],
             calibration["signal_stats"]["max"], out_path)
    return str(out_path)
