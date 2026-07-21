"""Стадия segment: картинка -> сегментация + векторизация ядром felixkrones.

Ядро felixkrones/ECG-Digitiser монолитно (сегментация nnU-Net + Hough-поворот +
векторизация + запись WFDB идут одним CLI-вызовом). Поэтому «тяжёлую» работу
делает эта стадия: она вызывает ядро как ИЗОЛИРОВАННЫЙ подпроцесс (venv ecgdig)
и читает результат из файлов (WFDB). Стадия calibrate ниже по цепочке лишь
поднимает метаданные калибровки, которые ядро применило внутри.

Вход:  путь к картинке (png).
Выход: путь к манифесту segment.json (пути к signal.npy, preview.png, WFDB, ...).

Источник: https://github.com/felixkrones/ECG-Digitiser
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir  # noqa: E402
from wrappers.felixkrones_core import run_core  # noqa: E402

STAGE = "segment"
log = get_logger(STAGE)


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    manifest = run_core(input_path, str(out_dir), config)

    manifest_path = out_dir / "segment.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("STAGE %s: сигнал %d×%d -> %s",
             STAGE, manifest["n_leads"], manifest["n_samples"], manifest_path)
    return str(manifest_path)
