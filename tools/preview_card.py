"""Отрисовать карточку измерений в отдельный HTML — чтобы посмотреть глазами.

Служебный инструмент: берёт готовый signal.csv, считает измерения и кладёт
рядом страницу только с карточкой, без остального сайта.

    .venv/bin/python tools/preview_card.py output/web/<run>/signal.csv out.html
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "service"))

from flask import render_template_string          # noqa: E402

import app as site                                # noqa: E402
import ecg_card as ec                             # noqa: E402
import ecg_delineate as dl                        # noqa: E402
from oecg_render import FS, load_csv              # noqa: E402

FRAG = """<!doctype html><html lang=ru><head><meta charset=utf-8>{{style|safe}}
<!-- Предпросмотр рисуется QuickLook в узком окне, поэтому раскладку для
     широкого экрана задаём принудительно: смотреть надо именно её. -->
<style>body{width:1500px;zoom:.62}
.beatbox{grid-template-columns:repeat(6,1fr)!important}
.mrow{grid-template-columns:repeat(4,1fr) 168px!important}
.st{grid-template-columns:repeat(12,1fr)!important}</style>
</head><body><div class=wrap><div class=card>
<h2 style="margin-bottom:16px">Измерения</h2>
{% include "card" %}</div></div></body></html>"""


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "card.html")
    sig, names = load_csv(csv_path)
    c = ec.card(dl.measure(sig, names, FS, simultaneous=False))
    if not c:
        raise SystemExit("измерения не сошлись")

    style = re.search(r"<style>.*?</style>", site.PAGE, re.S).group(0)
    body = re.search(r"\{% if r\.card %\}(.*?)\{% endif %\}\n\n  \{% if r\.filled",
                     site.PAGE, re.S).group(1)
    with site.app.test_request_context():
        html = render_template_string(
            FRAG.replace('{% include "card" %}', body), style=style, r={"card": c})
    out.write_text(html, encoding="utf-8")
    print(f"{out}  ({len(html)} символов)")


if __name__ == "__main__":
    main()
