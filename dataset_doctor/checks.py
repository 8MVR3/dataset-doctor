"""Проверки датасета. Каждая проверка возвращает находку с объяснением и примерами."""
from __future__ import annotations

import hashlib
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from PIL import Image

from .loader import Box, Dataset, Item

warnings.filterwarnings("ignore", category=UserWarning)

CRITICAL, WARNING, INFO = "critical", "warning", "info"
MAX_EXAMPLES = 8


@dataclass
class Example:
    items: list[Item]
    note: str = ""
    boxes: list[Box] = field(default_factory=list)


@dataclass
class Finding:
    code: str
    severity: str
    title: str
    what: str
    fix: str
    count: int
    examples: list[Example] = field(default_factory=list)


@dataclass
class ImageInfo:
    ok: bool
    error: str = ""
    orientation: int = 1
    md5: str = ""
    dhash: int | None = None


def plural(n: int, one: str, few: str, many: str) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return f"{n} {one}"
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def _dhash(img: Image.Image) -> int:
    small = img.convert("L").resize((9, 8), Image.BILINEAR)
    px = small.tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            left, right = px[row * 9 + col], px[row * 9 + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def inspect_images(ds: Dataset, progress=None) -> dict[int, ImageInfo]:
    infos = {}
    for i, it in enumerate(ds.items):
        try:
            data = it.image.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
            with Image.open(it.image) as img:
                orientation = img.getexif().get(274, 1) or 1
                it.width, it.height = img.size
                img.draft("RGB", (128, 128))
                h = _dhash(img)
            infos[i] = ImageInfo(ok=True, orientation=orientation, md5=md5, dhash=h)
        except Exception as e:  # битый или нечитаемый файл
            infos[i] = ImageInfo(ok=False, error=str(e)[:120])
        if progress and (i + 1) % 200 == 0:
            progress(i + 1, len(ds.items))
    return infos


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.x - a.w / 2, a.y - a.h / 2, a.x + a.w / 2, a.y + a.h / 2
    bx1, by1, bx2, by2 = b.x - b.w / 2, b.y - b.h / 2, b.x + b.w / 2, b.y + b.h / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _source_group(stem: str) -> str:
    stem = stem.split(".rf.")[0]
    stem = re.sub(r"_(jpg|jpeg|png)$", "", stem, flags=re.I)
    return re.sub(r"[\d]+$", "", stem).rstrip("_- ") or stem


def run_checks(ds: Dataset, infos: dict[int, ImageInfo], near_threshold: int = 6) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    items = ds.items
    nc = len(ds.names)

    def add(code, severity, title, what, fix, examples, count=None):
        if examples or count:
            findings.append(Finding(code, severity, title, what, fix, count if count is not None else len(examples),
                                    examples[:MAX_EXAMPLES]))

    # --- фото ---
    broken = [Example([it], infos[i].error) for i, it in enumerate(items) if not infos[i].ok]
    add("broken_images", CRITICAL, "Битые или нечитаемые фото",
        "Эти файлы не открываются как изображения. При обучении они либо вызовут ошибку, либо будут молча пропущены.",
        "Удалите файлы или замените их исправными копиями.", broken)

    exif = [Example([it], f"метка поворота EXIF = {infos[i].orientation}") for i, it in enumerate(items)
            if infos[i].ok and infos[i].orientation not in (0, 1)]
    add("exif_rotation", WARNING, "Поворот записан только в EXIF",
        "Пиксели лежат в одной ориентации, а метка EXIF велит показать фото повёрнутым. Разные программы читают это по-разному, "
        "и рамки могут оказаться не на объектах.",
        "Поверните фото по-настоящему (например, ImageOps.exif_transpose в Pillow), сохраните без метки и проверьте разметку.", exif)

    # --- файлы разметки ---
    missing = [Example([it]) for it in items if it.label is None]
    add("missing_labels", WARNING, "Фото без файла разметки",
        "Для этих фото нет файла .txt. YOLO считает такие фото фоном без объектов, но часто это просто забытая разметка.",
        "Проверьте фото: если объектов нет, создайте пустой .txt; если есть, разметьте.", missing)

    orphans = [Example([], str(p)) for p in ds.orphan_labels]
    add("orphan_labels", WARNING, "Разметка без фото",
        "Файлы .txt, для которых не нашлось изображения с тем же именем. Обычно это остатки переименований или удалённых фото.",
        "Удалите лишние файлы или верните недостающие фото.", orphans)

    syntax = [Example([it], "; ".join(it.label_errors[:3])) for it in items if it.label_errors]
    add("label_syntax", CRITICAL, "Ошибки в строках разметки",
        "Строки, которые нельзя прочитать как рамку YOLO: не числа, не то количество значений, дробный или отрицательный класс.",
        "Исправьте строки вручную или пересоберите разметку из инструмента разметки.", syntax)

    bad_cls, bad_coords, off_edge, tiny, dup_boxes, conflict = [], [], [], [], [], []
    for it in items:
        W, H = it.width or 0, it.height or 0
        cls_hits, coord_hits, edge_hits, tiny_hits = [], [], [], []
        for b in it.boxes:
            if nc and b.cls >= nc:
                cls_hits.append(b)
            if not (0 <= b.x <= 1 and 0 <= b.y <= 1) or b.w <= 0 or b.h <= 0 or b.w > 1.001 or b.h > 1.001:
                coord_hits.append(b)
                continue
            if b.x - b.w / 2 < -0.002 or b.y - b.h / 2 < -0.002 or b.x + b.w / 2 > 1.002 or b.y + b.h / 2 > 1.002:
                edge_hits.append(b)
            if W and H and (b.w * W < 4 or b.h * H < 4):
                tiny_hits.append(b)
        if cls_hits:
            bad_cls.append(Example([it], f"классы {sorted({b.cls for b in cls_hits})} при {nc} классах в описании", cls_hits))
        if coord_hits:
            bad_coords.append(Example([it], f"строки {[b.line for b in coord_hits][:5]}", coord_hits))
        if edge_hits:
            edge_boxes = edge_hits
            off_edge.append(Example([it], f"{plural(len(edge_boxes), 'рамка', 'рамки', 'рамок')} за краем фото", edge_boxes))
        if tiny_hits:
            tiny.append(Example([it], f"{plural(len(tiny_hits), 'рамка', 'рамки', 'рамок')} меньше 4 пикселей по стороне", tiny_hits))
        dups, confl = [], []
        for a, b in combinations(it.boxes, 2):
            iou = _iou(a, b)
            if iou > 0.9:
                (dups if a.cls == b.cls else confl).append((a, b))
        if dups:
            dup_boxes.append(Example([it], f"{plural(len(dups), 'пара', 'пары', 'пар')} почти одинаковых рамок", [x for p in dups for x in p]))
        if confl:
            conflict.append(Example([it], "одна рамка размечена разными классами", [x for p in confl for x in p]))

    add("class_out_of_range", CRITICAL, "Номер класса больше, чем классов в описании",
        "В разметке встречаются классы, которых нет в data.yaml. Обучение упадёт или будет учить «пустой» класс.",
        "Проверьте список классов в data.yaml и порядок классов при экспорте из инструмента разметки.", bad_cls)
    add("bad_coords", CRITICAL, "Некорректные координаты рамок",
        "Центр рамки вне фото, нулевая или отрицательная ширина или высота, размер больше фото. Такие рамки бессмысленны.",
        "Удалите или переразметьте эти рамки.", bad_coords)
    add("box_off_edge", WARNING, "Рамки выходят за край фото",
        "Часть рамки оказывается за границей изображения. Обычно это ошибка экспорта или поворота.",
        "Обрежьте рамки по краю фото или проверьте, не повёрнуто ли фото.", off_edge)
    add("tiny_boxes", WARNING, "Слишком маленькие рамки",
        "Рамки меньше 4 пикселей по одной из сторон. Модель почти не может их выучить, а иногда это случайные клики.",
        "Удалите случайные рамки, а для мелких объектов увеличьте размер входа модели или режьте фото на части.", tiny)
    add("duplicate_boxes", WARNING, "Двойные рамки на одном объекте",
        "На одном объекте стоят две почти совпадающие рамки одного класса (IoU больше 0,9). Модель учится выдавать дубли.",
        "Оставьте одну рамку из каждой пары.", dup_boxes)
    add("class_conflict", WARNING, "Один объект — два класса",
        "Почти совпадающие рамки разных классов. Разметчики разошлись во мнениях, или объект размечен дважды.",
        "Решите, какой класс верный, и уберите лишнюю рамку. Если такое частое — уточните требования к разметке.", conflict)

    # --- дубли и утечки ---
    ok_idx = [i for i in range(len(items)) if infos[i].ok]
    by_md5 = defaultdict(list)
    for i in ok_idx:
        by_md5[infos[i].md5].append(i)
    exact_cross, exact_within = [], []
    exact_pairs = set()
    for group in by_md5.values():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            exact_pairs.add((a, b))
            ex = Example([items[a], items[b]], "файлы совпадают побайтно")
            (exact_cross if items[a].split != items[b].split else exact_within).append(ex)

    buckets = defaultdict(list)
    for i in ok_idx:
        h = infos[i].dhash
        for band in range(8):
            buckets[(band, (h >> (band * 8)) & 0xFF)].append(i)
    near_pairs = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 2000:
            continue
        for a, b in combinations(members, 2):
            if (a, b) in exact_pairs or (a, b) in near_pairs:
                continue
            if bin(infos[a].dhash ^ infos[b].dhash).count("1") <= near_threshold:
                near_pairs.add((a, b))
    near_cross = [Example([items[a], items[b]], "почти одинаковые кадры") for a, b in sorted(near_pairs)
                  if items[a].split != items[b].split]
    near_within_count = sum(1 for a, b in near_pairs if items[a].split == items[b].split)

    add("leak_exact", CRITICAL, "Одни и те же фото в разных выборках",
        "Одинаковые файлы лежат, например, и в train, и в val. Метрики на проверке получаются завышенными: модель «видела» эти фото.",
        "Оставьте каждое фото только в одной выборке.", exact_cross)
    add("leak_near", CRITICAL, "Похожие кадры в разных выборках",
        "Почти одинаковые фото — обычно соседние кадры одного видео — попали в разные выборки. Проверка перестаёт быть честной: "
        "на новых данных качество будет ниже, чем показывает val.",
        "Делите данные на выборки по источникам (видео, дням съёмки, объектам), а не по отдельным кадрам.",
        near_cross, count=len(near_cross))
    add("dup_within", WARNING, "Дубли фото внутри выборки",
        "Одинаковые файлы внутри одной выборки. Они не добавляют информации, но увеличивают вес отдельных примеров.",
        "Удалите копии.", exact_within)

    groups = defaultdict(set)
    for it in items:
        groups[_source_group(it.image.stem)].add(it.split)
    framey = [g for g, s in groups.items() if len(s) > 1 and re.search(r"frame|video|кадр|vid", g, re.I)]
    if framey and not near_cross:
        ex = [Example([], f"группа «{g}*» встречается в выборках: {', '.join(sorted(groups[g]))}") for g in framey]
        add("leak_names", WARNING, "Кадры одного видео могут быть в разных выборках",
            "Судя по именам файлов, кадры одних и тех же видео лежат в разных выборках. Похожих пар мы не нашли, но стоит проверить вручную.",
            "Делите данные по видео, а не по кадрам.", ex)

    # --- статистика и баланс ---
    splits = sorted({it.split for it in items}, key=lambda s: ["train", "val", "test", "all"].index(s) if s in ("train", "val", "test", "all") else 9)
    per_split_cls = {s: Counter() for s in splits}
    sizes = Counter()
    boxes_total = 0
    for it in items:
        for b in it.boxes:
            per_split_cls[it.split][b.cls] += 1
            boxes_total += 1
            if it.width and it.height:
                area = b.w * it.width * b.h * it.height
                sizes["small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"] += 1
    all_cls = sorted({c for cnt in per_split_cls.values() for c in cnt} | set(range(nc)))
    totals = Counter()
    for cnt in per_split_cls.values():
        totals.update(cnt)

    missing_cls = []
    if "train" in per_split_cls and "val" in per_split_cls:
        for c in all_cls:
            tr, va = per_split_cls["train"][c], per_split_cls["val"][c]
            if (tr == 0) != (va == 0):
                name = ds.names[c] if c < nc else str(c)
                where = "train" if tr == 0 else "val"
                missing_cls.append(Example([], f"класс «{name}»: нет ни одного объекта в {where}"))
    add("class_missing_split", WARNING, "Класс есть только в одной выборке",
        "Если класса нет в train, модель его не выучит; если нет в val, его качество никто не проверит.",
        "Добавьте примеры класса в обе выборки.", missing_cls)

    present = [totals[c] for c in all_cls if totals[c] > 0]
    if len(present) >= 2 and max(present) / min(present) > 10:
        hi = max(all_cls, key=lambda c: totals[c])
        rare = sorted((c for c in all_cls if 0 < totals[c] < totals[hi] / 10), key=lambda c: totals[c])
        lo = rare[0]
        nm = lambda c: ds.names[c] if c < nc else str(c)
        add("imbalance", WARNING, "Сильный дисбаланс классов",
            f"Самый частый класс «{nm(hi)}» встречается в {totals[hi] / max(totals[lo], 1):.0f} раз чаще самого редкого «{nm(lo)}». "
            "Редкие классы модель распознаёт заметно хуже.",
            "Доразметьте редкие классы, добавьте синтетику или используйте взвешивание.",
            [Example([], f"«{nm(c)}»: {totals[c]} объектов против {totals[hi]} у «{nm(hi)}»") for c in rare])

    empty_share = {}
    for s in splits:
        s_items = [it for it in items if it.split == s and it.label is not None]
        if s_items:
            empty_share[s] = sum(1 for it in s_items if not it.boxes) / len(s_items)
    heavy_bg = [s for s, v in empty_share.items() if v > 0.3]
    if heavy_bg:
        add("many_backgrounds", INFO, "Много фото без объектов",
            "Фото с пустой разметкой (фон) полезны, но обычно их держат на уровне до 10%. Больше — модель тратит обучение впустую.",
            "Проверьте, не пропущены ли объекты на этих фото, и уберите лишний фон.",
            [Example([], f"{s}: {empty_share[s]:.0%} фото без объектов") for s in heavy_bg])

    stats = {
        "images": len(items), "boxes": boxes_total, "classes": nc, "names": ds.names,
        "splits": {s: sum(1 for it in items if it.split == s) for s in splits},
        "per_split_cls": {s: dict(per_split_cls[s]) for s in splits},
        "sizes": dict(sizes), "empty_share": empty_share, "near_within": near_within_count,
        "segmentation": ds.is_segmentation, "root": str(ds.root),
    }
    order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    findings.sort(key=lambda f: (order[f.severity], -f.count))
    return findings, stats
