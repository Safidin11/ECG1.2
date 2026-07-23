#!/usr/bin/env bash
# Прогнать ЭКГ-фото через пайплайн и открыть результат.
# Использование:  ./run_ecg.sh /путь/к/фото.jpg
set -e
cd "$(dirname "$0")"

if [ -z "$1" ]; then
  echo "Использование: ./run_ecg.sh /путь/к/фото.(jpg|png)"
  exit 1
fi
if [ ! -f "$1" ]; then
  echo "Файл не найден: $1"; exit 1
fi

echo "Запускаю пайплайн на: $1"
echo "(первый прогон новой картинки ~5-8 мин — nnU-Net на CPU; потом мгновенно из кэша)"
./.venv/bin/python src/pipeline.py --input "$1" 2>&1 | grep -vE "nnUNet_raw|nnUNet_preprocessed"

RUN=$(ls -td output/runs/2026* | head -1)
echo ""
echo "Готово. Результаты в: $RUN"
echo "  раскладка: $RUN/layout/overlay.png"
echo "  сигнал:    $RUN/vectorize/preview.png"
echo "  цифровая ЭКГ: $RUN/output/digital_ecg.png"
open "$RUN/output/digital_ecg.png" "$RUN/layout/overlay.png" 2>/dev/null || true
