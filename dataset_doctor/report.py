"""HTML-отчёт: заключение по датасету в духе медицинской справки."""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime

from PIL import Image, ImageDraw, ImageOps

from .checks import CRITICAL, INFO, WARNING, Example, Finding

SEVERITY = {
    CRITICAL: ("Критично", "crit"),
    WARNING: ("Внимание", "warn"),
    INFO: ("К сведению", "info"),
}

CSS = """
:root{--paper:#F7F9FA;--sheet:#FFFFFF;--ink:#1D2733;--muted:#5B6B7A;--line:#DCE3E8;
--crit:#B42318;--warn:#A66A12;--info:#2F6F8F;--ok:#2F7D6A;--head:#1F4E79}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;font-variant-numeric:tabular-nums}
.sheet{max-width:980px;margin:32px auto;background:var(--sheet);border:1px solid var(--line);padding:40px 44px}
header{border-bottom:3px double var(--line);padding-bottom:20px;margin-bottom:24px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px;color:var(--head);font-weight:700;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0;word-break:break-all}
.card{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);margin:22px 0 8px}
.card div{padding:12px 16px;border-right:1px solid var(--line)}
.card div:last-child{border-right:0}
.card b{display:block;font-size:22px;line-height:1.2}
.card span{color:var(--muted);font-size:13px}
.verdict{font-size:18px;margin:18px 0 0;max-width:70ch}
.verdict.bad{color:var(--crit)}.verdict.mid{color:var(--warn)}.verdict.good{color:var(--ok)}
h2{font-size:19px;margin:34px 0 12px;color:var(--head)}
.finding{border:1px solid var(--line);border-left:6px solid var(--muted);margin:0 0 14px;padding:14px 18px}
.finding.crit{border-left-color:var(--crit)}.finding.warn{border-left-color:var(--warn)}.finding.info{border-left-color:var(--info)}
.fhead{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.fhead h3{margin:0;font-size:16.5px}
.tag{font-size:12.5px;font-weight:600;padding:1px 8px;border-radius:3px;color:#fff}
.crit .tag{background:var(--crit)}.warn .tag{background:var(--warn)}.info .tag{background:var(--info)}
.count{color:var(--muted);font-size:14px}
.finding p{margin:8px 0 0;max-width:78ch}
.finding p.fix{color:var(--ink)}
.finding p.fix::before{content:"Что сделать: ";font-weight:600}
details{margin-top:10px}
summary{cursor:pointer;color:var(--head);font-weight:600}
summary:focus-visible{outline:2px solid var(--head);outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-top:12px}
.ex{border:1px solid var(--line);padding:8px;font-size:13px;color:var(--muted);word-break:break-all}
.ex.pair{grid-column:span 2}
.ex .pics{display:flex;gap:6px}
.ex .pics figure{margin:0;flex:1;min-width:0}
.ex img{width:100%;height:auto;display:block;border:1px solid var(--line)}
.ex figcaption{margin-top:4px}
.ex .note{margin-top:6px;color:var(--ink)}
.empty{padding:16px 18px;border:1px dashed var(--line);color:var(--ok)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:middle}
th{color:var(--muted);font-weight:600}
td.num{text-align:right;width:70px}
.bar{height:9px;background:var(--head);opacity:.75;border-radius:2px;min-width:2px}
.wrap{overflow-x:auto}
footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
@media (max-width:700px){.sheet{margin:0;padding:22px 18px;border:0}.card{grid-template-columns:repeat(2,1fr)}
.card div:nth-child(2){border-right:0}.card div:nth-child(-n+2){border-bottom:1px solid var(--line)}.ex.pair{grid-column:auto}}
"""


def _thumb(ex: Example, item, highlight) -> str:
    try:
        with Image.open(item.image) as im:
            im.draft("RGB", (720, 720))
            img = im.convert("RGB")
    except Exception:
        return ""
    img.thumbnail((440, 440))
    W, H = img.size
    d = ImageDraw.Draw(img)
    hl = {id(b) for b in highlight}
    for b in item.boxes:
        x1, y1 = (b.x - b.w / 2) * W, (b.y - b.h / 2) * H
        x2, y2 = (b.x + b.w / 2) * W, (b.y + b.h / 2) * H
        if id(b) in hl:
            d.rectangle([x1, y1, x2, y2], outline=(180, 35, 24), width=3)
        else:
            d.rectangle([x1, y1, x2, y2], outline=(31, 78, 121), width=1)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _example_html(ex: Example, root) -> str:
    def rel(p):
        try:
            return str(p.relative_to(root))
        except ValueError:
            return str(p)

    note = f'<div class="note">{html.escape(ex.note)}</div>' if ex.note else ""
    if not ex.items:
        return f'<div class="ex">{note}</div>'
    figs = []
    for it in ex.items:
        src = _thumb(ex, it, ex.boxes)
        img = f'<img src="{src}" alt="{html.escape(it.image.name)}">' if src else ""
        figs.append(f"<figure>{img}<figcaption>{html.escape(it.split)}: {html.escape(rel(it.image))}</figcaption></figure>")
    cls = "ex pair" if len(ex.items) > 1 else "ex"
    return f'<div class="{cls}"><div class="pics">{"".join(figs)}</div>{note}</div>'


def _verdict(findings: list[Finding]) -> tuple[str, str]:
    crit = sum(1 for f in findings if f.severity == CRITICAL)
    warn = sum(1 for f in findings if f.severity == WARNING)
    if crit:
        return "bad", f"Перед обучением нужно исправить {crit} критичных проблем{'у' if crit == 1 else ''}: иначе модель будет учиться на ошибках, а метрики окажутся неправдивыми."
    if warn:
        return "mid", f"Критичных проблем нет. Есть {warn} замечани{'е' if warn == 1 else 'й' if warn >= 5 else 'я'}, которые стоит разобрать."
    return "good", "Проблем не найдено: датасет можно отдавать в обучение."


def render(findings: list[Finding], stats: dict, root, title: str | None = None) -> str:
    names = stats["names"]
    nm = lambda c: names[c] if c < len(names) else f"класс {c}"
    vcls, vtext = _verdict(findings)
    splits_txt = ", ".join(f"{s}: {n}" for s, n in stats["splits"].items())
    card = (f'<div class="card"><div><b>{stats["images"]}</b><span>фото</span></div>'
            f'<div><b>{stats["boxes"]}</b><span>рамок</span></div>'
            f'<div><b>{len(names) or "—"}</b><span>классов в описании</span></div>'
            f'<div><b>{len(stats["splits"])}</b><span>выборки ({html.escape(splits_txt)})</span></div></div>')

    parts = []
    for f in findings:
        label, cls = SEVERITY[f.severity]
        examples = "".join(_example_html(e, root) for e in f.examples)
        more = f" (показаны первые {len(f.examples)})" if f.count > len(f.examples) else ""
        details = (f'<details><summary>Примеры{more}</summary><div class="grid">{examples}</div></details>'
                   if f.examples else "")
        parts.append(f'<section class="finding {cls}"><div class="fhead"><span class="tag">{label}</span>'
                     f'<h3>{html.escape(f.title)}</h3><span class="count">{f.count}</span></div>'
                     f'<p>{html.escape(f.what)}</p><p class="fix">{html.escape(f.fix)}</p>{details}</section>')
    findings_html = "".join(parts) or '<div class="empty">Проверки пройдены: ошибок разметки, дублей и утечек не найдено.</div>'

    per = stats["per_split_cls"]
    splits = list(per)
    all_cls = sorted({c for d in per.values() for c in d} | set(range(len(names))))
    peak = max([n for d in per.values() for n in d.values()] or [1])
    rows = []
    for c in all_cls:
        cells = "".join(f'<td class="num">{per[s].get(c, 0)}</td>' for s in splits)
        total = sum(per[s].get(c, 0) for s in splits)
        width = 100 * max(per[s].get(c, 0) for s in splits) / peak if splits else 0
        rows.append(f'<tr><td>{html.escape(nm(c))}</td>{cells}<td class="num">{total}</td>'
                    f'<td style="width:30%"><div class="bar" style="width:{width:.0f}%"></div></td></tr>')
    head = "".join(f'<th class="num">{html.escape(s)}</th>' for s in splits)
    cls_table = (f'<div class="wrap"><table><thead><tr><th>Класс</th>{head}<th class="num">всего</th><th></th></tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')

    sz = stats["sizes"]
    tot = sum(sz.values()) or 1
    size_rows = "".join(
        f'<tr><td>{lbl}</td><td class="num">{sz.get(k, 0)}</td><td class="num">{sz.get(k, 0) / tot:.0%}</td></tr>'
        for k, lbl in (("small", "Мелкие, до 32×32 пикселей"), ("medium", "Средние, до 96×96"), ("large", "Крупные")))
    bg = "".join(f'<tr><td>{html.escape(s)}</td><td class="num">{v:.0%}</td></tr>' for s, v in stats["empty_share"].items())
    seg = "<p>Разметка похожа на сегментацию: для проверок использованы описывающие прямоугольники полигонов.</p>" if stats["segmentation"] else ""

    when = datetime.now().strftime("%d.%m.%Y %H:%M")
    name = title or stats["root"]
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Заключение по датасету</title><style>{CSS}</style></head><body><main class="sheet">
<header><h1>Заключение по датасету</h1><p class="sub">{html.escape(str(name))}, проверено {when}</p>
{card}<p class="verdict {vcls}">{html.escape(vtext)}</p></header>
<h2>Найденные проблемы</h2>{findings_html}
<h2>Классы по выборкам</h2>{cls_table}
<h2>Размеры объектов</h2><div class="wrap"><table><tbody>{size_rows}</tbody></table></div>
<h2>Доля фото без объектов</h2><div class="wrap"><table><tbody>{bg}</tbody></table></div>
<p>Пар похожих кадров внутри одной выборки: {stats["near_within"]}. Это не ошибка, но много похожих кадров снижает разнообразие данных.</p>
{seg}
<footer>Отчёт подготовлен инструментом dataset-doctor. Проверки: целостность файлов, формат разметки YOLO, геометрия рамок, дубли и утечки между выборками, EXIF, баланс классов.</footer>
</main></body></html>"""
