"""Загрузка датасета в формате YOLO: выборки, пары «фото — разметка», рамки."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_KEYS = ("train", "val", "valid", "test")


@dataclass
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float
    line: int


@dataclass
class Item:
    split: str
    image: Path
    label: Path | None
    boxes: list[Box] = field(default_factory=list)
    label_errors: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None


@dataclass
class Dataset:
    root: Path
    names: list[str]
    items: list[Item]
    orphan_labels: list[Path]
    is_segmentation: bool = False


def _label_path_for(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    sibling = image.parent.parent / "labels" / (image.stem + ".txt")
    if sibling.parent.is_dir():
        return sibling
    return image.with_suffix(".txt")


def _resolve(value: str, root: Path, yaml_dir: Path) -> Path | None:
    candidates = [root / value, yaml_dir / value, yaml_dir / value.replace("../", "", 1)]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return None


def _names_from_yaml(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    if isinstance(names, list):
        return [str(n) for n in names]
    if "nc" in data:
        return [str(i) for i in range(int(data["nc"]))]
    return []


def _find_splits(path: Path) -> tuple[Path, dict[str, Path], list[str]]:
    """Возвращает корень, словарь «выборка → папка с фото» и имена классов."""
    yaml_file = None
    if path.is_file() and path.suffix in (".yaml", ".yml"):
        yaml_file = path
    elif path.is_dir():
        found = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
        yaml_file = found[0] if found else None

    if yaml_file:
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        yaml_dir = yaml_file.parent
        root = Path(data["path"]) if data.get("path") else yaml_dir
        if not root.is_absolute():
            root = (yaml_dir / root).resolve()
        splits = {}
        for key in SPLIT_KEYS:
            if isinstance(data.get(key), str):
                d = _resolve(data[key], root, yaml_dir)
                if d:
                    splits["val" if key == "valid" else key] = d
        if splits:
            return root, splits, _names_from_yaml(data)
        path = yaml_dir

    root = path.resolve()
    splits = {}
    for key in SPLIT_KEYS:
        for cand in (root / key / "images", root / "images" / key):
            if cand.is_dir():
                splits["val" if key == "valid" else key] = cand
    if not splits and (root / "images").is_dir():
        splits["all"] = root / "images"
    if not splits:
        splits["all"] = root
    names = []
    for cand in (root / "classes.txt", root / "labels" / "classes.txt"):
        if cand.exists():
            names = [l.strip() for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()]
            break
    return root, splits, names


def _parse_label(item: Item) -> bool:
    """Читает разметку. Возвращает True, если строки похожи на полигоны сегментации."""
    seg = False
    text = item.label.read_text(encoding="utf-8", errors="replace")
    for n, raw in enumerate(text.splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            item.label_errors.append(f"строка {n}: не числа — «{raw.strip()[:60]}»")
            continue
        if len(values) == 5:
            c, x, y, w, h = values
        elif len(values) > 5 and (len(values) - 1) % 2 == 0:
            seg = True
            xs, ys = values[1::2], values[2::2]
            c = values[0]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            item.label_errors.append(f"строка {n}: {len(values)} чисел вместо 5")
            continue
        if c != int(c) or c < 0:
            item.label_errors.append(f"строка {n}: номер класса {parts[0]} не целое неотрицательное число")
            continue
        item.boxes.append(Box(int(c), x, y, w, h, n))
    return seg


def load(path: str | Path) -> Dataset:
    root, splits, names = _find_splits(Path(path))
    items, label_files, seg_any = [], set(), False
    for split, img_dir in splits.items():
        for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT):
            lbl = _label_path_for(img)
            item = Item(split=split, image=img, label=lbl if lbl.exists() else None)
            if item.label:
                label_files.add(lbl.resolve())
                seg_any |= _parse_label(item)
            items.append(item)

    orphans = []
    label_dirs = {(_label_path_for(it.image)).parent for it in items}
    for d in label_dirs:
        if d.is_dir():
            for txt in d.glob("*.txt"):
                if txt.name != "classes.txt" and txt.resolve() not in label_files:
                    orphans.append(txt)
    return Dataset(root=root, names=names, items=items, orphan_labels=sorted(orphans), is_segmentation=seg_any)
