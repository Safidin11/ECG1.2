"""Стадия calibrate: маска + картинка -> масштаб + угол наклона.

Целевое поведение (Фаза 1): калибровка через Hough Transform из
felixkrones/ECG-Digitiser — определение шага сетки, пересчёт
пиксели->мВ и пиксели->секунды (25мм/с, 10мм/мВ) и угла наклона.
Тот же изолированный подпроцесс, что и segment; результат — JSON со scale.

Пока (Фаза 0) — passthrough.
"""
from common import passthrough

STAGE = "calibrate"


def run(input_path: str, config: dict) -> str:
    # TODO(Фаза 1): Hough-калибровка из felixkrones, вывод scale.json.
    return passthrough(input_path, config, STAGE)
