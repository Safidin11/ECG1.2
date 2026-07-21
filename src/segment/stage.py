"""Стадия segment: картинка -> бинарная маска кривой ЭКГ (сигнал vs сетка/фон).

Целевое поведение (Фаза 1): обёртка nnU-Net сегментации из
felixkrones/ECG-Digitiser (победитель PhysioNet Challenge 2024).
Вызывается как ИЗОЛИРОВАННЫЙ подпроцесс со своим venv; обмен данными —
через файлы (.npy/.png), НЕ через общие импорты.

Пока (Фаза 0) — passthrough.
"""
from common import passthrough

STAGE = "segment"


def run(input_path: str, config: dict) -> str:
    # TODO(Фаза 1): subprocess -> external/ECG-Digitiser CLI, чтение маски.
    return passthrough(input_path, config, STAGE)
