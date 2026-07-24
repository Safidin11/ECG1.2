"""ECG1.2 — локальный веб-сервис дигитизации ЭКГ.

Загрузи картинку ЭКГ, выбери формат раскладки (или «авто») — получишь
цифровую ЭКГ, разметку отведений и сигналы. Работает по цветовым чернилам,
без 8-минутного nnU-Net (быстро, секунды).

Запуск:
    ./.venv/bin/python service/app.py
    открой http://127.0.0.1:5000

Learning/demo-инструмент. НЕ медицинское изделие.
"""
import json
import sys
import uuid
from pathlib import Path

from flask import Flask, request, send_file, abort, render_template_string

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import run_pipeline  # noqa: E402

UPLOADS = ROOT / "output" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

FORMATS = [
    ("auto", "Авто (определить самому)"),
    ("3x4_1R", "3×4 + ритм II (стандарт)"),
    ("3x4", "3×4 (без ритма)"),
    ("3x4_3R", "3×4 + 3 ритма (V1, II, V5)"),
    ("6x2_1R", "6×2 + ритм II"),
    ("6x2", "6×2 (без ритма)"),
    ("12x1", "12×1 (каждое отведение 10с)"),
]

app = Flask(__name__)

PAGE = """
<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>ECG1.2 — дигитайзер ЭКГ</title>
<style>
:root{--bg:#0f1420;--card:#182233;--ink:#e8eef7;--mut:#93a1b5;--acc:#4f9dff;--line:#26344a}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{max-width:1000px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--mut);margin:0 0 20px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:18px}
label{display:block;font-weight:600;margin:0 0 6px;font-size:13px;color:var(--mut)}
input[type=file],select{width:100%;padding:11px;border-radius:9px;border:1px solid var(--line);
  background:#0e1626;color:var(--ink);font-size:14px}
.row{display:flex;gap:16px;flex-wrap:wrap}.row>div{flex:1;min-width:220px}
button{margin-top:16px;background:var(--acc);color:#04122b;border:0;border-radius:9px;
  padding:12px 22px;font-weight:700;font-size:15px;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
.imgs{display:grid;grid-template-columns:1fr;gap:16px}
.imgs figure{margin:0;background:#0e1626;border:1px solid var(--line);border-radius:10px;padding:10px}
.imgs figcaption{color:var(--mut);font-size:13px;margin-bottom:8px}
.imgs img{width:100%;border-radius:6px;display:block}
.cov{font-size:13px;color:var(--mut);margin-top:6px}
.err{background:#3a1620;border:1px solid #6b2233;color:#ffb3c0;padding:12px;border-radius:9px}
.tag{display:inline-block;background:#0e1626;border:1px solid var(--line);border-radius:6px;
  padding:2px 8px;font-size:12px;color:var(--acc);margin-left:6px}
.warn{color:#e0a44a;font-size:12px;margin-top:14px}
a{color:var(--acc)}
</style></head><body><div class=wrap>
<h1>ECG1.2 — дигитайзер ЭКГ <span class=tag>demo</span></h1>
<p class=sub>Картинка бумажной ЭКГ → цифровой сигнал. Не медицинское изделие.</p>

<form class=card method=post action="/digitize" enctype="multipart/form-data" onsubmit="go(this)">
  <div class=row>
    <div><label>Картинка ЭКГ (jpg/png)</label>
      <input type=file name=image accept="image/*" required></div>
    <div><label>Формат раскладки</label>
      <select name=template>
        {% for val,txt in formats %}<option value="{{val}}">{{txt}}</option>{% endfor %}
      </select></div>
  </div>
  <button type=submit>Оцифровать</button>
  <div class=warn>Первый прогон — несколько секунд (препроцессинг). Если авто-формат
    промахнётся — выбери формат вручную.</div>
</form>

{% if error %}<div class="card err">{{error}}</div>{% endif %}

{% if result %}
<div class=card>
  <h3 style="margin:0 0 12px">Результат <span class=tag>{{result.template}}</span></h3>
  <div class=imgs>
    <figure><figcaption>Цифровая ЭКГ (реконструкция)</figcaption>
      <img src="/img/{{result.run}}/output/digital_ecg.png"></figure>
    <figure><figcaption>Разметка отведений (overlay)</figcaption>
      <img src="/img/{{result.run}}/layout/overlay.png"></figure>
    <figure><figcaption>Сигналы по отведениям</figcaption>
      <img src="/img/{{result.run}}/vectorize/preview.png"></figure>
  </div>
  <div class=cov>Покрытие: {{result.cov}}</div>
  <div class=cov>WFDB: <code>{{result.run}}/output/ecg_reconstructed.dat</code></div>
</div>
{% endif %}

<script>
function go(f){var b=f.querySelector('button');b.disabled=true;b.textContent='Обрабатываю…';}
</script>
</div></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, formats=FORMATS, result=None, error=None)


@app.route("/digitize", methods=["POST"])
def digitize():
    f = request.files.get("image")
    template = request.form.get("template", "auto")
    if not f or f.filename == "":
        return render_template_string(PAGE, formats=FORMATS, result=None,
                                      error="Файл не выбран")
    ext = Path(f.filename).suffix.lower() or ".png"
    up = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    f.save(str(up))
    # Проверим, что картинку вообще можно прочитать (HEIC/иные форматы OpenCV не берёт).
    import cv2
    if cv2.imread(str(up)) is None:
        return render_template_string(
            PAGE, formats=FORMATS, result=None,
            error="Не удалось прочитать картинку. Используй JPG или PNG "
                  "(HEIC/иные форматы не поддерживаются — сконвертируй в JPG).")
    try:
        out_json = run_pipeline(str(up), str(ROOT / "configs" / "pipeline.yml"),
                                template=(None if template == "auto" else template),
                                fast=True)
        d = {}
        try:
            with open(out_json, encoding="utf-8") as fp:
                d = json.load(fp)
        except Exception:
            d = {}                          # пайплайн вернул не JSON (все стадии деградировали)
        if not d.get("digital_ecg"):
            hint = "Попробуй выбрать формат вручную из списка" if template == "auto" \
                else "Проверь, что выбран правильный формат, или попробуй другое фото"
            raise RuntimeError("не удалось распознать раскладку ЭКГ. " + hint)
        run = Path(out_json).parent.parent.name
        lay = d.get("layout", {})
        cov = d.get("coverage", {})
        cov_txt = ", ".join(f"{k} {int(v*100)}%" for k, v in cov.items()) or "—"
        result = {"run": run, "template": lay.get("template", template), "cov": cov_txt}
        return render_template_string(PAGE, formats=FORMATS, result=result, error=None)
    except Exception as exc:
        return render_template_string(PAGE, formats=FORMATS, result=None,
                                      error=f"{exc}")


@app.route("/img/<run>/<stage>/<name>")
def img(run, stage, name):
    # Отдаём картинки из output/runs с защитой от выхода за пределы каталога.
    p = (ROOT / "output" / "runs" / run / stage / name).resolve()
    base = (ROOT / "output" / "runs").resolve()
    if base not in p.parents or not p.exists():
        abort(404)
    return send_file(str(p))


if __name__ == "__main__":
    print("ECG1.2 service -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
