"""Стадия preprocess: фото/скан -> выровненная нормализованная картинка.

Целевое поведение (Фаза 2): OpenCV-препроцессор.
  - перевод в градации серого
  - детекция 4 углов листа + коррекция перспективы (getPerspectiveTransform)
  - выравнивание освещения / удаление теней (adaptiveThreshold)
  - обрезка полей, нормализация под формат ядра felixkrones/ECG-Digitiser

Пока (Фаза 0) — passthrough.
"""
from common import passthrough

STAGE = "preprocess"


def run(input_path: str, config: dict) -> str:
    # TODO(Фаза 2): реальный OpenCV-препроцессинг.
    return passthrough(input_path, config, STAGE)
