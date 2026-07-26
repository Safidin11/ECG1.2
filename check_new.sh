#!/usr/bin/env bash
# Проверить фотку НОВЫМ движком (Open-ECG-Digitizer, обучен на реальных фото).
# Использование:  ./check_new.sh /путь/к/фото.jpg
set -e
cd "$(dirname "$0")"

if [ -z "$1" ]; then
  echo "Использование: ./check_new.sh /путь/к/фото.(jpg|png)"
  exit 1
fi
if [ ! -f "$1" ]; then
  echo "Файл не найден: $1"; exit 1
fi

NAME=$(basename "$1"); NAME="${NAME%.*}"
OUT="output/oecg/$NAME"

echo "Оцифровываю новым движком: $1"
echo "(крупное фото ужимается автоматически; обычно 1-3 минуты)"
./.venv/bin/python tools/oecg_digitize.py -i "$1" -o "$OUT"

echo ""
echo "Результат: $OUT"
open "$OUT/$NAME.png" 2>/dev/null || true
