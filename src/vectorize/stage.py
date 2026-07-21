"""Стадия vectorize: маска + калибровка -> сырой сигнал по каждому отведению.

Целевое поведение: трассировка кривой по маске в числовой ряд с учётом
калибровки (мВ/с). При наложениях трасс (Фаза 6, опц.) — подход разделения
сигналов из masoudrahimi39/ECG-code.

Пока (Фаза 0) — passthrough.
"""
from common import passthrough

STAGE = "vectorize"


def run(input_path: str, config: dict) -> str:
    # TODO: трассировка маски -> raw-сигнал (.npy) с учётом scale.
    return passthrough(input_path, config, STAGE)
