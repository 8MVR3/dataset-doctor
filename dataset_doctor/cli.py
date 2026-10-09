"""Командная строка: dataset-doctor путь/к/data.yaml --out отчёт.html"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .checks import CRITICAL, WARNING, inspect_images, run_checks
from .loader import load
from .report import render


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dataset-doctor", description="Проверка датасета YOLO перед обучением")
    ap.add_argument("path", help="data.yaml или папка датасета")
    ap.add_argument("--out", default="dataset_report.html", help="куда сохранить HTML-отчёт")
    ap.add_argument("--json", dest="json_out", help="дополнительно сохранить находки в JSON")
    ap.add_argument("--near", type=int, default=6, help="порог похожести кадров (0–64, меньше — строже), по умолчанию 6")
    ap.add_argument("--fail-on-critical", action="store_true", help="код выхода 1, если есть критичные проблемы (для CI)")
    ap.add_argument("--version", action="version", version=f"dataset-doctor {__version__}")
    a = ap.parse_args(argv)

    ds = load(a.path)
    if not ds.items:
        print(f"В «{a.path}» не нашлось ни одного изображения. Укажите data.yaml или папку с images/ и labels/.", file=sys.stderr)
        return 2
    print(f"Датасет: {ds.root}")
    print(f"Фото: {len(ds.items)}, выборки: {', '.join(sorted({it.split for it in ds.items}))}")

    def progress(done, total):
        print(f"\r  проверено фото: {done} из {total}", end="", flush=True)

    infos = inspect_images(ds, progress)
    print()
    findings, stats = run_checks(ds, infos, near_threshold=a.near)

    Path(a.out).write_text(render(findings, stats, ds.root), encoding="utf-8")
    crit = sum(1 for f in findings if f.severity == CRITICAL)
    warn = sum(1 for f in findings if f.severity == WARNING)
    print(f"Критичных проблем: {crit}, замечаний: {warn}")
    for f in findings:
        print(f"  [{f.severity}] {f.title}: {f.count}")
    print(f"Отчёт: {Path(a.out).resolve()}")

    if a.json_out:
        data = {"stats": stats, "findings": [
            {"code": f.code, "severity": f.severity, "title": f.title, "count": f.count,
             "examples": [{"files": [str(it.image) for it in e.items], "note": e.note} for e in f.examples]}
            for f in findings]}
        Path(a.json_out).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 1 if (a.fail_on_critical and crit) else 0
