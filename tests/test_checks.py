"""Мини-датасет с заранее известными ошибками: каждая должна найтись."""
import random

from PIL import Image, ImageDraw

from dataset_doctor.checks import inspect_images, run_checks
from dataset_doctor.loader import load


def _img(path, seed):
    rnd = random.Random(seed)
    im = Image.new("RGB", (320, 240), (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x, y = rnd.randint(0, 300), rnd.randint(0, 220)
        d.rectangle([x, y, x + rnd.randint(5, 60), y + rnd.randint(5, 60)],
                    fill=(rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))
    im.save(path, quality=95)


def make_dataset(root):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for i in range(6):
        _img(root / "images" / "train" / f"a{i}.jpg", i)
        (root / "labels" / "train" / f"a{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    for i in range(3):
        _img(root / "images" / "val" / f"b{i}.jpg", 100 + i)
        (root / "labels" / "val" / f"b{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    # утечка: точная копия train-фото в val
    (root / "images" / "val" / "leak.jpg").write_bytes((root / "images" / "train" / "a0.jpg").read_bytes())
    (root / "labels" / "val" / "leak.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    # ошибки разметки
    (root / "labels" / "train" / "a1.txt").write_text("0 1.4 0.5 0.2 0.2\n5 0.5 0.5 0.1 0.1\n0 0.5 x 0.1 0.1\n")
    (root / "labels" / "train" / "a2.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
    (root / "labels" / "train" / "a3.txt").unlink()
    (root / "labels" / "train" / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (root / "images" / "train" / "broken.jpg").write_bytes(b"not an image")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: a\n  1: b\n")


def test_all_injected_problems_are_found(tmp_path):
    make_dataset(tmp_path)
    ds = load(tmp_path / "data.yaml")
    findings, stats = run_checks(ds, inspect_images(ds))
    codes = {f.code for f in findings}
    expected = {"leak_exact", "bad_coords", "class_out_of_range", "label_syntax", "duplicate_boxes",
                "missing_labels", "orphan_labels", "broken_images", "class_missing_split"}
    assert expected <= codes, expected - codes
    assert stats["splits"] == {"train": 7, "val": 4}


def test_clean_dataset_has_no_critical(tmp_path):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        for i in range(3):
            _img(tmp_path / "images" / split / f"{split}{i}.jpg", i + (0 if split == "train" else 50))
            (tmp_path / "labels" / split / f"{split}{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    (tmp_path / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [obj]\n")
    ds = load(tmp_path / "data.yaml")
    findings, _ = run_checks(ds, inspect_images(ds))
    assert not [f for f in findings if f.severity == "critical"]
