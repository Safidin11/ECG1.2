"""Общие утилиты для стадий пайплайна ECG1.2.

Каждая стадия — отдельный модуль с функцией:
    run(input_path: str, config: dict) -> str   # возвращает путь к своему выходу

Здесь лежат хелперы, которыми пользуются все стадии: логирование,
создание рабочих папок и passthrough-заглушка (копирование входа в выход).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-11s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(stage: str) -> logging.Logger:
    return logging.getLogger(stage)


def stage_dir(config: dict, stage: str) -> Path:
    """Папка для артефактов конкретной стадии внутри текущего run-каталога."""
    run_dir = Path(config["_run_dir"])
    d = run_dir / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def color_ink(bgr: np.ndarray, thr: int | None = None) -> np.ndarray:
    """Извлечь «чернила» ЭКГ по ЯРКОСТИ с адаптивным порогом (двухуровневый Otsu).

    Трасса — самая ТЁМНАЯ структура, любого цвета (чёрная, синяя, фиолетовая).
    Порог подбирается сам под каждую картинку: сначала Otsu отделяет фон от
    не-фона (сетка+трасса), затем второй Otsu среди не-фона отделяет более тёмную
    трассу от более светлой сетки. Так работает и на цветных распечатках, где
    «тёмное во всех каналах» теряло синюю трассу (у неё высокий канал B).

    thr — можно задать порог яркости вручную (иначе адаптивно).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if thr is None:
        t1, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = gray[gray < t1]
        if dark.size > 100:
            thr, _ = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thr = t1
    return (gray < thr).astype(np.uint8)


def load_ink(core_ready_path: str) -> np.ndarray:
    """Бинарная маска трассы: готовая ink.png рядом с core_ready (Sauvola из
    preprocess) — иначе fallback на color_ink по самой картинке."""
    p = Path(core_ready_path)
    ink_p = p.parent / "ink.png"
    if ink_p.exists():
        return (cv2.imread(str(ink_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    return color_ink(cv2.imread(str(p)))


def passthrough(input_path: str, config: dict, stage: str) -> str:
    """Заглушка стадии: копирует вход в выход и логирует passthrough.

    Используется, пока реальная логика стадии не реализована. Позволяет
    прогнать весь пайплайн end-to-end ещё на этапе каркаса (Фаза 0).
    """
    log = get_logger(stage)
    src = Path(input_path)
    dst = stage_dir(config, stage) / src.name
    shutil.copy2(src, dst)
    log.info("STAGE %s: passthrough  %s -> %s", stage, src.name, dst)
    return str(dst)
