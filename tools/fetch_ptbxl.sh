#!/usr/bin/env bash
# Скачать данные для проверки точности: PTB-XL (сигналы) и PTB-XL+ (разметка).
#
# PTB-XL      — записи с известным сигналом, из них печатаем плёнку.
# PTB-XL+     — интервалы для ТЕХ ЖЕ записей, посчитанные двумя медицинскими
#               анализаторами (University of Glasgow и GE 12SL). Два, а не один,
#               потому что их расхождение между собой и есть масштаб, с которым
#               имеет смысл сравнивать нашу ошибку.
#
# Обе базы открытые, регистрации не требуют. В репозиторий не кладутся
# (см. .gitignore) — файлы разметки весят по 90 МБ.
#
#   tools/fetch_ptbxl.sh          # 40 записей — хватает для стенда
#   tools/fetch_ptbxl.sh 100      # больше
set -euo pipefail

N="${1:-40}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIG="$ROOT/data/ptbxl"
FEAT="$ROOT/data/ptbxl_plus"
BASE="https://physionet.org/files/ptb-xl/1.0.3/records500/00000"
PLUS="https://physionet.org/files/ptb-xl-plus/1.0.1/features"

mkdir -p "$SIG" "$FEAT"

echo "сигналы: $N записей -> $SIG"
for i in $(seq 1 "$N"); do
  name=$(printf "%05d_hr" "$i")
  for ext in hea dat; do
    [ -s "$SIG/$name.$ext" ] && continue
    # -C - докачивает оборванное: файлы разметки большие, и обрыв на середине
    # выглядит как нормальный CSV — потом молча теряются записи.
    curl -fsSL --retry 5 -C - -o "$SIG/$name.$ext" "$BASE/$name.$ext" \
      || echo "  пропуск $name.$ext"
  done
done

echo "разметка анализаторов -> $FEAT"
for f in unig_features.csv 12sl_features.csv feature_description.csv; do
  curl -fL --retry 5 -C - -o "$FEAT/$f" "$PLUS/$f" && echo "  $f: $(wc -c <"$FEAT/$f") байт"
done

echo
echo "готово. Проверка:"
echo "  .venv/bin/python tools/validate_ptbxl.py -n $N      # точность оцифровки"
echo "  .venv/bin/python tools/validate_measure.py -n $N    # точность измерений"
