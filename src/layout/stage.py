"""Стадия layout: определение раскладки отведений на картинке.

Целевое поведение (Фаза 3): сопоставление регионов картинки с известными
раскладками по шаблонам. Подход из Ahus-AIM/Open-ECG-Digitizer
(src/config/lead_layouts_*.yml). Модуль сообщает остальному пайплайну,
где какое отведение (bbox -> имя отведения).

Пока (Фаза 0) — passthrough.
"""
from common import passthrough

STAGE = "layout"


def run(input_path: str, config: dict) -> str:
    # TODO(Фаза 3): матчинг раскладки по шаблонам lead_layouts_*.yml.
    return passthrough(input_path, config, STAGE)
