# ECG1.2 — гибридный дигитайзер ЭКГ

Модульный пайплайн дигитизации ЭКГ: **картинка бумажной распечатки → цифровой
сигнал 12×N**. Собирается из лучших открытых решений, каждое подключается как
изолированный модуль.

> [!WARNING]
> **⚠️ Это learning/demo-инструмент, НЕ медицинское изделие.**
> Результаты не предназначены для диагностики, лечения или любых клинических
> решений. Не используйте на реальных пациентских данных для медицинских целей.

---

## Архитектура

Пайплайн — цепочка логических стадий. Каждая стадия — отдельный модуль в `src/`
с единой сигнатурой:

```python
run(input_path: str, config: dict) -> output_path: str
```

Выход одной стадии = вход следующей. Оркестратор `src/pipeline.py` читает
`configs/pipeline.yml` и вызывает активные стадии по очереди.

| # | Стадия        | Вход → Выход                                        | Источник подхода |
|---|---------------|-----------------------------------------------------|------------------|
| 1 | `preprocess`  | фото/скан → выровненная нормализованная картинка    | OpenCV (свой)    |
| 2 | `layout`      | картинка → раскладка отведений (где какое)          | Ahus-AIM/Open-ECG-Digitizer |
| 3 | `segment`     | картинка → бинарная маска кривой ЭКГ                | felixkrones/ECG-Digitiser (nnU-Net) |
| 4 | `calibrate`   | маска+картинка → масштаб (px→мВ, px→с) + угол        | felixkrones/ECG-Digitiser (Hough) |
| 5 | `vectorize`   | маска+калибровка → сырой сигнал по отведениям        | свой (+ masoudrahimi39 для наложений) |
| 6 | `reconstruct` | сырой сигнал → полные 10с × 12 отведений            | UMMISCO/ecgtizer |
| 7 | `output`      | сигнал → WFDB/.npy + превью-график                  | свой             |

### Принципы

- **Изоляция зависимостей.** Внешние модели с конфликтующими требованиями
  подключаются как **отдельные подпроцессы**, каждый со своим venv. Обмен
  данными — через **файлы** (`.npy` / WFDB / JSON), а не через общие импорты.
- **Мягкая деградация.** Падение одной стадии не роняет весь пайплайн: ошибка
  логируется, вход прокидывается дальше без изменений.
- **Конфиг рулит.** Какие стадии активны, пути и параметры — в
  `configs/pipeline.yml` (флаг `enabled` у каждой стадии).

---

## Структура

```
src/
  preprocess/ layout/ segment/ calibrate/ vectorize/ reconstruct/ output/
  pipeline.py        # оркестратор
  common.py          # общие утилиты (логирование, passthrough)
external/            # клоны внешних репозиториев (gitignored)
weights/             # веса моделей (gitignored, git-lfs)
data/samples/        # тестовые картинки
configs/pipeline.yml # пути, параметры, флаги enabled
tools/               # вспомогательные скрипты (генератор синтетики и т.п.)
tests/
```

---

## Установка (ядро)

Ядро (оркестратор + препроцессор) использует собственный venv:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install pyyaml numpy opencv-python-headless matplotlib
```

Внешние модели (felixkrones, ecgtizer, …) ставятся в свои venv в соответствующих
фазах — см. ниже. Они **намеренно** не в одном окружении с ядром.

### Ядро felixkrones (Фаза 1)

- Окружение: conda-env `ecgdig` (Python 3.11 + torch + nnU-Net), путь прописан в
  `configs/pipeline.yml → interpreters.segment_calibrate`. Вызывается **подпроцессом**
  (`python -m src.run.digitize`), обмен через WFDB-файлы, без общих импортов.
- Веса: M3 (`fold_all/checkpoint_final.pth`, ~453 МБ). На GitHub у репо felixkrones
  **исчерпан LFS-бюджет**, поэтому веса берутся из локальной копии
  `~/ECG-Digitiser/models/M3` (симлинк в `external/`), и в `weights/`-политику не
  попадают. Если у тебя весов нет — их нужно получить из релиза/зеркала проекта.
- Синтетика в «родном» формате генерируется встроенным `ecg-image-generator`
  (см. `tools/make_native_wfdb.py` + `tools/gen_native_image.py`).
- nnU-Net на CPU ≈ 8 мин/картинка, поэтому результат **кэшируется** по содержимому
  входа (`output/cache/<hash>/`).

---

## Запуск

```bash
# сгенерировать тестовую синтетическую картинку (если нужно)
./.venv/bin/python tools/make_synthetic_ecg.py

# прогнать пайплайн
./.venv/bin/python src/pipeline.py --input data/samples/synthetic_ecg.png
```

Артефакты каждого запуска складываются в `output/runs/run_<timestamp>/` — по
подпапке на стадию (gitignored).

---

## Дорожная карта (фазы)

- **Фаза 0** — каркас: структура, оркестратор, стадии-заглушки (passthrough). ✅
- **Фаза 1** — ядро felixkrones/ECG-Digitiser (segment + calibrate), прогон на синтетике. ✅
- **Фаза 2** — OpenCV-препроцессор реального фото (перспектива, тени, обрезка).
- **Фаза 3** — шаблоны раскладок отведений (Open-ECG-Digitizer).
- **Фаза 4** — достройка отведений до 10с (UMMISCO/ecgtizer).
- **Фаза 5** — валидация на PTB-Image (MSE, корреляция Пирсона).
- **Фаза 6** (опц.) — разделение наложенных трасс (masoudrahimi39/ECG-code).

Текущий статус — см. коммиты; каждая фаза завершается отдельным коммитом.

---

## Источники

- felixkrones/ECG-Digitiser — https://github.com/felixkrones/ECG-Digitiser
- UMMISCO/ecgtizer — https://github.com/UMMISCO/ecgtizer
- Ahus-AIM/Open-ECG-Digitizer — https://github.com/Ahus-AIM/Open-ECG-Digitizer
- masoudrahimi39/ECG-code — https://github.com/masoudrahimi39/ECG-code
- adofersan/ecg-miner, Tereshchenkolab/paper-ecg — референс/GUI для сверки.

Лицензии внешних проектов остаются за их авторами; здесь они лишь оркеструются.
