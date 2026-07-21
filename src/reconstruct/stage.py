"""Стадия reconstruct: сырой сигнал -> полные 10с × 12 отведений.

Целевое поведение (Фаза 4): достройка обрезанных (2.5с/5с) отведений до
полных 10 секунд моделью из UMMISCO/ecgtizer (режимы full/lazy/fragmented).
Изолированный подпроцесс со своим venv; обмен через файлы (.npy/WFDB).

Пока (Фаза 0) — passthrough.
"""
from common import passthrough

STAGE = "reconstruct"


def run(input_path: str, config: dict) -> str:
    # TODO(Фаза 4): subprocess -> external/ecgtizer, достройка до 10с.
    return passthrough(input_path, config, STAGE)
