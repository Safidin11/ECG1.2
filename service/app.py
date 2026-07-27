"""ECG1.2 — веб-сервис оцифровки ЭКГ.

Фото бумажной ЭКГ -> цифровой сигнал 12 отведений + реконструкция на
миллиметровке. Движок: Open-ECG-Digitizer (U-Net, обучен на реальных фото;
см. external/Open-ECG-Digitizer) — запускается изолированным субпроцессом
со своим venv, как и другие внешние решения в проекте.

Запуск:
    ./.venv/bin/python service/app.py     ->  http://127.0.0.1:5000

Learning/demo-инструмент. НЕ медицинское изделие.
"""
from __future__ import annotations

import json
import shutil
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, render_template_string, request, send_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from oecg_digitize import digitize                      # noqa: E402
from oecg_render import coverage, load_csv, render      # noqa: E402
from ecg_leads import consistency, reconstruct_limb   # noqa: E402

# Формат ЭКГ: имя раскладки движка -> понятная подпись. Первый пункт — авто.
FORMATS = [
    ("", "Определить автоматически"),
    ("standard_3x4_with_r1", "3×4 + ритм II  (стандарт)"),
    ("standard_3x4", "3×4  (без ритм-строки)"),
    ("standard_3x4_with_r2", "3×4 + 2 ритм-строки"),
    ("standard_3x4_with_r3", "3×4 + 3 ритм-строки"),
    ("standard_12x1", "12×1  (каждое отведение 10 с)"),
    ("standard_6x2_with_r1", "6×2 + ритм"),
    ("standard_6x2", "6×2  (без ритм-строки)"),
    ("precordial_6x1", "6×1  грудные V1–V6"),
    ("standard_6x1_limb", "6×1  от конечностей"),
    ("precordial_3x2", "3×2  грудные"),
    ("standard_3x1", "3×1"),
    ("cabrera_12x1", "12×1  Кабрера"),
    ("cabrera_6x1_limb", "6×1  Кабрера, от конечностей"),
]
FORMAT_LABEL = dict(FORMATS)

RUNS = ROOT / "output" / "web"
RUNS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024     # 60 МБ на файл

PAGE = """
<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>ECG1.2 — оцифровка ЭКГ</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0f17; --bg2:#111827; --card:#151d2b; --line:#243044;
  --ink:#e8eef7; --mut:#8b9ab1; --acc:#22d3a5; --acc2:#38bdf8;
  --warn:#fbbf24; --err:#f87171;
}
@media (prefers-color-scheme: light){
  :root{--bg:#f5f7fa;--bg2:#fff;--card:#fff;--line:#e3e8f0;--ink:#0f172a;--mut:#64748b}
}
body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--ink);min-height:100vh}
.wrap{max-width:1560px;margin:0 auto;padding:28px 24px 56px}

header{display:flex;align-items:center;gap:14px;margin-bottom:6px}
.logo{width:42px;height:42px;border-radius:11px;flex:0 0 42px;
  background:linear-gradient(135deg,var(--acc),var(--acc2));
  display:grid;place-items:center;color:#04121b}
h1{font-size:23px;font-weight:700;letter-spacing:-.02em}
.tag{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;color:var(--acc);border:1px solid var(--acc);
  border-radius:5px;padding:1px 7px;vertical-align:3px;margin-left:9px}
.sub{color:var(--mut);font-size:14px;margin:0 0 26px 56px}

.card{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:22px;margin-bottom:20px}

.bar-form{display:grid;grid-template-columns:minmax(280px,1fr) 320px 190px;
  gap:16px;align-items:end}
@media (max-width:900px){.bar-form{grid-template-columns:1fr}}
.drop{display:flex;align-items:center;gap:14px;border:2px dashed var(--line);
  border-radius:12px;padding:15px 18px;cursor:pointer;transition:.15s;background:var(--bg2);
  min-height:64px}
.drop:hover,.drop.over{border-color:var(--acc)}
.drop svg{opacity:.45;flex:0 0 26px}
.drop b{display:block;font-size:14.5px;margin-bottom:2px}
.drop span{color:var(--mut);font-size:12.5px}
.drop.has{border-style:solid;border-color:var(--acc)}
input[type=file]{display:none}
.field{margin:0}
.field label{display:block;font-size:12.5px;font-weight:600;color:var(--mut);
  margin-bottom:7px}
select{width:100%;padding:12px 13px;border-radius:10px;border:1px solid var(--line);
  background:var(--bg2);color:var(--ink);font-size:14.5px;font-family:inherit;
  cursor:pointer;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='none' stroke='%238b9ab1' stroke-width='1.8' d='M1 1.5 6 6.5 11 1.5'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 14px center}
select:focus{outline:none;border-color:var(--acc)}
button{width:100%;background:var(--acc);color:#04121b;border:0;
  border-radius:10px;padding:13px;font-size:15px;font-weight:700;cursor:pointer;
  transition:.15s;font-family:inherit}
button:hover{filter:brightness(1.08)}
button:disabled{opacity:.55;cursor:wait}
.hint{color:var(--mut);font-size:12.5px;margin-top:12px;text-align:center}
.busy{display:none;grid-column:1/-1;text-align:center;padding:8px 0 0;
  color:var(--mut);font-size:13px}
.busy.on{display:block}
.spin{width:26px;height:26px;border:3px solid var(--line);border-top-color:var(--acc);
  border-radius:50%;margin:0 auto 10px;animation:s .8s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}

.head{display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:12px;margin-bottom:18px}
h2{font-size:18px;font-weight:700}
.badges{display:flex;gap:8px;flex-wrap:wrap}
.badge{font-size:12px;font-weight:600;padding:4px 11px;border-radius:99px;
  background:var(--bg2);border:1px solid var(--line);color:var(--mut)}
.badge.ok{color:var(--acc);border-color:color-mix(in srgb,var(--acc) 40%,transparent)}
.badge.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 40%,transparent)}

figure{margin-bottom:20px}
figcaption{font-size:13px;color:var(--mut);margin-bottom:9px}
figcaption b{color:var(--ink);font-weight:600;font-size:13.5px}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}
@media (max-width:1100px){.compare{grid-template-columns:1fr}}
.compare figure{margin:0}
.imgbox{background:#fff;border:1px solid var(--line);border-radius:11px;
  padding:8px;overflow:auto;max-height:78vh}
.imgbox img{width:100%;display:block;border-radius:5px}

.leads{display:grid;grid-template-columns:repeat(12,1fr);gap:8px}
@media (max-width:1100px){.leads{grid-template-columns:repeat(6,1fr)}}
@media (max-width:640px){.leads{grid-template-columns:repeat(3,1fr)}}
.lead{background:var(--bg2);border:1px solid var(--line);border-radius:9px;padding:9px 11px}
.lead .n{font-size:12px;font-weight:700}
.lead .v{font-size:11px;color:var(--mut);margin-top:1px}
.bar{height:3px;border-radius:2px;background:var(--line);margin-top:6px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--acc);border-radius:2px}
.lead.low .bar i{background:var(--warn)}
.note{background:var(--bg2);border:1px solid var(--line);border-left:3px solid var(--acc);
  border-radius:8px;padding:11px 14px;margin-bottom:18px;font-size:13px;color:var(--mut)}
.note b{color:var(--ink)}
.preview{display:none;grid-column:1/-1;margin-top:6px}
.preview.on{display:block}
.preview .imgbox{max-height:52vh}

details{margin-top:8px}
summary{cursor:pointer;color:var(--mut);font-size:13px;padding:9px 0;
  list-style:none;user-select:none}
summary::-webkit-details-marker{display:none}
summary:before{content:"▸";display:inline-block;margin-right:6px;transition:.15s}
details[open] summary:before{transform:rotate(90deg)}
summary:hover{color:var(--ink)}

.err{background:color-mix(in srgb,var(--err) 12%,var(--card));
  border-color:color-mix(in srgb,var(--err) 35%,transparent)}
.err b{display:block;margin-bottom:5px;font-size:15px;color:var(--err)}
.files{font-size:12.5px;color:var(--mut);margin-top:16px;
  padding-top:15px;border-top:1px solid var(--line)}
.files code{background:var(--bg2);padding:2px 7px;border-radius:5px;
  font-size:12px;color:var(--ink)}
footer{text-align:center;color:var(--mut);font-size:12.5px;margin-top:32px;line-height:1.8}
</style></head><body><div class=wrap>

<header>
  <div class=logo><svg width=24 height=24 viewBox="0 0 24 24" fill=none
    stroke=currentColor stroke-width=2.2 stroke-linecap=round stroke-linejoin=round>
    <path d="M2 12h4l2-7 4 14 3-9 2 2h5"/></svg></div>
  <h1>ECG1.2<span class=tag>demo</span></h1>
</header>
<p class=sub>Фотография бумажной ЭКГ → цифровой сигнал 12 отведений</p>

<form class="card bar-form" method=post action="/digitize" enctype="multipart/form-data" id=f>
  <label class=drop id=drop>
    <input type=file name=image accept="image/*" required id=file>
    <svg width=26 height=26 viewBox="0 0 24 24" fill=none stroke=currentColor
      stroke-width=1.6 stroke-linecap=round><path d="M12 16V4m0 0L7 9m5-5 5 5"/>
      <path d="M3 15v4a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4"/></svg>
    <div><b id=dropTitle>Выбери фото ЭКГ</b>
      <span id=dropSub>или перетащи сюда · JPG, PNG</span></div>
  </label>

  <div class=field>
    <label for=fmt>Формат ЭКГ</label>
    <select name=layout id=fmt>
      {% for val, txt in formats %}
      <option value="{{val}}" {{'selected' if val == sel else ''}}>{{txt}}</option>
      {% endfor %}
    </select>
  </div>

  <div class=field>
    <label>&nbsp;</label>
    <button type=submit id=go>Оцифровать</button>
  </div>
  <div class=busy id=busy><div class=spin></div>
    <span>Обрабатываю — обычно 1–2 минуты…</span></div>

  <div class=preview id=preview>
    <figcaption><b>Выбранный снимок</b> — посмотри и определи формат, если авто ошибётся</figcaption>
    <div class=imgbox><img id=previewImg alt="выбранный снимок"></div>
  </div>
</form>

{% if error %}<div class="card err"><b>Не получилось</b>{{error}}</div>{% endif %}

{% if r %}
<div class=card>
  <div class=head>
    <h2>Результат</h2>
    <div class=badges>
      <span class="badge {{'ok' if r.manual or r.cost < 0.3 else 'warn'}}">{{r.layout}}{{'  · вручную' if r.manual else ''}}</span>
      <span class="badge {{'ok' if r.n_leads == 12 else 'warn'}}">{{r.n_leads}} из 12 отведений</span>
      <span class="badge {{'ok' if r.resid_ok else 'warn'}}"
        title="Отведения от конечностей связаны формулами (II = I + III и др.). Насколько оцифровка им противоречит: чем меньше, тем достовернее.">сходимость {{r.resid_txt}}</span>
      <span class=badge>{{r.secs}} с</span>
    </div>
  </div>

  <figure>
    <figcaption><b>Цифровая копия</b> — та же геометрия, что на снимке; линия построена по данным</figcaption>
    <div class=imgbox><img src="/img/{{r.run}}/{{'twin.png' if r.twin else 'digital_ecg.png'}}" alt="цифровая копия"></div>
  </figure>

  <details>
    <summary>Показать стандартную раскладку (25 мм/с · 10 мм/мВ)</summary>
    <div class=imgbox style="margin-top:10px">
      <img src="/img/{{r.run}}/digital_ecg.png" alt="стандартная раскладка"></div>
  </details>

  {% if r.filled %}
  <div class=note>Восстановлено <b>{{r.filled_sec}} с</b> сигнала в отведениях от конечностей —
    вычислено по связям между ними (II = I + III, aVR + aVL + aVF = 0), а не дорисовано.</div>
  {% endif %}

  <figcaption style="margin-bottom:10px"><b>Качество по отведениям</b>
    — какую долю сигнала удалось прочитать</figcaption>
  <div class=leads>
    {% for name, pct in r.leads %}
    <div class="lead {{'low' if pct < 60 else ''}}">
      <div class=n>{{name}}</div><div class=v>{{pct}}%</div>
      <div class=bar><i style="width:{{pct}}%"></i></div>
    </div>{% endfor %}
  </div>

  {% if r.stages %}
  <details>
    <summary>Показать все этапы Open ECG Digitizer</summary>
    <div class=imgbox style="margin-top:10px">
      <img src="/img/{{r.run}}/stages.png" alt="этапы движка"></div>
  </details>
  {% endif %}

  <div class=files>
    Файлы: <code>output/web/{{r.run}}/</code> — сигнал <code>signal.csv</code>,
    картинка <code>digital_ecg.png</code>
  </div>
</div>
{% endif %}

<footer>
  Движок: <b>Open-ECG-Digitizer</b> — U-Net, обучен на реальных фото ЭКГ<br>
  Учебно-демонстрационный инструмент. <b>Не медицинское изделие</b> — не для диагностики.
</footer>

<script>
const file=document.getElementById('file'), drop=document.getElementById('drop');
file.onchange=()=>{const f=file.files[0]; if(!f)return;
  drop.classList.add('has');
  document.getElementById('dropTitle').textContent=f.name;
  document.getElementById('dropSub').textContent='готово к оцифровке';
  const rd=new FileReader();
  rd.onload=e=>{document.getElementById('previewImg').src=e.target.result;
    document.getElementById('preview').classList.add('on');};
  rd.readAsDataURL(f);};
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{
  ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{
  ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>{file.files=ev.dataTransfer.files;
  file.dispatchEvent(new Event('change'))});
document.getElementById('f').onsubmit=()=>{
  const b=document.getElementById('go');
  b.disabled=true;b.textContent='Обрабатываю…';
  document.getElementById('busy').classList.add('on')};
</script>
</div></body></html>
"""


def _page(error=None, r=None, sel=""):
    return render_template_string(PAGE, error=error, r=r, formats=FORMATS, sel=sel)


@app.route("/")
def index():
    return _page()


@app.route("/digitize", methods=["POST"])
def digitize_route():
    f = request.files.get("image")
    chosen = (request.form.get("layout") or "").strip()
    if chosen and chosen not in FORMAT_LABEL:
        return _page(error="Неизвестный формат ЭКГ.", sel="")
    if not f or not f.filename:
        return _page(error="Файл не выбран.", sel=chosen)

    run = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
    run_dir = RUNS / run
    run_dir.mkdir(parents=True, exist_ok=True)
    src = run_dir / ("upload" + (Path(f.filename).suffix.lower() or ".png"))
    f.save(str(src))

    try:
        import cv2
        if cv2.imread(str(src), cv2.IMREAD_UNCHANGED) is None:
            raise RuntimeError("Не удалось прочитать картинку. Нужен JPG или PNG "
                               "(HEIC не поддерживается — сконвертируй в JPG).")

        t0 = datetime.now()
        engine_out = run_dir / "engine"
        digitize(str(src), str(engine_out), layout=chosen or None)
        secs = int((datetime.now() - t0).total_seconds())

        csvs = list(engine_out.glob("*_timeseries_canonical.csv"))
        if not csvs:
            raise RuntimeError("Не удалось прочитать ЭКГ на этом снимке. Попробуй кадр, "
                               "где плёнка видна целиком, без сильного наклона и смаза.")
        shutil.copy2(csvs[0], run_dir / "signal.csv")

        for src_name, dst_name in (("*_twin.png", "twin.png"),
                                   ("*_aligned.png", "aligned.png"),
                                   ("*_stages.png", "stages.png")):
            found = list(engine_out.glob(src_name))
            if found:
                shutil.copy2(found[0], run_dir / dst_name)
        has_twin = (run_dir / "twin.png").exists()
        has_stages = (run_dir / "stages.png").exists()

        layout, cost, layout_key = "формат не определён", 1.0, None
        meta = engine_out / "digitization_metadata.csv"
        if meta.exists():
            last = meta.read_text(encoding="utf-8").strip().split("\n")[-1].split(",")
            if len(last) >= 4:
                cost = float(last[1])
                layout_key = None if last[3] == "Unknown layout" else last[3]
                layout = layout_key or "формат не определён"
        if chosen:                       # формат задан вручную — показываем его подпись
            layout, cost = FORMAT_LABEL[chosen], 0.0

        sig, names = load_csv(str(csvs[0]))
        cov_before = coverage(sig, names)
        check = consistency(sig, names)
        sig, rep = reconstruct_limb(sig, names)      # дыры в отв. от конечностей
        filled = sum(rep["filled"].values())

        render(str(csvs[0]), str(run_dir / "digital_ecg.png"),
               layout=(chosen or layout_key), sig=sig, leads=names)

        cov = coverage(sig, names)
        leads = [(n, int(round(100 * c))) for n, c in cov.items()]
        n_leads = sum(1 for c in cov.values() if c > 0.05)

        (run_dir / "result.json").write_text(json.dumps(
            {"layout": layout, "chosen": chosen or None, "cost": cost,
             "coverage": cov, "coverage_before": cov_before, "seconds": secs,
             "consistency": check, "reconstructed": rep["filled"]},
            ensure_ascii=False, indent=2), encoding="utf-8")

        resid = check.get("residual")
        return _page(sel=chosen, r={"run": run, "layout": layout, "cost": cost,
                                    "leads": leads, "n_leads": n_leads, "secs": secs,
                                    "manual": bool(chosen), "upload": src.name, "twin": has_twin, "stages": has_stages,
                                    "filled": filled, "filled_sec": round(filled / 1000.0, 1),
                                    "resid": resid,
                                    "resid_ok": (resid is not None and resid < 0.15),
                                    "resid_txt": (f"{100*resid:.0f}%" if resid is not None
                                                  else "нет данных")})
    except Exception as exc:
        app.logger.error("digitize failed: %s", traceback.format_exc())
        return _page(error=str(exc), sel=chosen)


@app.route("/img/<run>/<name>")
def img(run, name):
    p = (RUNS / run / name).resolve()
    if RUNS.resolve() not in p.parents or not p.exists():
        abort(404)
    return send_file(str(p))


if __name__ == "__main__":
    print("ECG1.2 -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
