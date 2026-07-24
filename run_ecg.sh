#!/usr/bin/env bash
# Прогнать ЭКГ-фото через пайплайн и открыть результат.
# Использование:  ./run_ecg.sh /путь/к/фото.jpg
set -e
cd "$(dirname "$0")"

if [ -z "$1" ]; then
  echo "Использование: ./run_ecg.sh /путь/к/фото.(jpg|png) [формат]"
  echo "  формат (необязательно): 3x4 3x4_1R 3x4_3R 6x2 6x2_1R 12x1 (по умолч. auto)"
  exit 1
fi
if [ ! -f "$1" ]; then
  echo "Файл не найден: $1"; exit 1
fi

TPL_ARG=""
[ -n "$2" ] && TPL_ARG="--template $2" && echo "Формат задан явно: $2"

echo "Запускаю пайплайн на: $1 (быстрый путь, без nnU-Net — секунды)"
./.venv/bin/python src/pipeline.py --input "$1" --fast $TPL_ARG 2>&1 | grep -vE "nnUNet_raw|nnUNet_preprocessed"

RUN=$(ls -td output/runs/2026* | head -1)
echo ""
echo "Готово. Результаты в: $RUN"
echo "  раскладка: $RUN/layout/overlay.png"
echo "  сигнал:    $RUN/vectorize/preview.png"
echo "  цифровая ЭКГ: $RUN/output/digital_ecg.png"
open "$RUN/output/digital_ecg.png" "$RUN/layout/overlay.png" 2>/dev/null || true
