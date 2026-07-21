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
