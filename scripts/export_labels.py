#!/usr/bin/env python3
"""Export canonical 0–1000 annotations to YOLO / Labelme / VOC / COCO / CSV / raw."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def image_size(path: Path) -> tuple[int, int]:
    if Image is not None:
        with Image.open(path) as im:
            return im.size  # W, H
    # minimal PNG/JPEG fallback without PIL: require PIL for accurate size
    raise RuntimeError("Pillow (PIL) is required for image size. pip install Pillow")


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def to_px(v: float, dim: int) -> int:
    return clamp(int(round(float(v) / 1000.0 * dim)), 0, max(0, dim - 1))


def normalize_instance(obj: dict[str, Any], mode: str) -> dict[str, Any] | None:
    label = str(obj.get("label") or obj.get("name") or obj.get("class") or "").strip()
    if not label:
        return None
    if mode == "bbox":
        box = obj.get("box") or obj.get("bbox") or obj.get("bndbox")
        if not box or len(box) != 4:
            # rectangle points?
            pts = obj.get("points")
            if pts and len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                box = [min(xs), min(ys), max(xs), max(ys)]
            else:
                return None
        x1, y1, x2, y2 = [float(x) for x in box]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        # clamp 0-1000
        x1, y1, x2, y2 = [max(0, min(1000, v)) for v in (x1, y1, x2, y2)]
        if x2 <= x1 or y2 <= y1:
            return None
        return {"label": label, "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]}
    # polygon
    pts = obj.get("points") or obj.get("polygon") or obj.get("segmentation")
    if not pts and obj.get("box"):
        x1, y1, x2, y2 = obj["box"]
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if not pts or len(pts) < 3:
        return None
    clean = []
    for p in pts:
        if len(p) < 2:
            continue
        x = max(0, min(1000, float(p[0])))
        y = max(0, min(1000, float(p[1])))
        clean.append([int(round(x)), int(round(y))])
    if len(clean) < 3:
        return None
    return {"label": label, "points": clean}


def ensure_class_index(classes: list[str], label: str) -> int:
    if label not in classes:
        classes.append(label)
    return classes.index(label)


def write_classes(labels_dir: Path, classes: list[str]) -> None:
    (labels_dir / "classes.txt").write_text("\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")
    data = {
        "path": str(labels_dir.parent.resolve()),
        "train": ".",
        "val": ".",
        "names": {i: n for i, n in enumerate(classes)},
        "nc": len(classes),
    }
    # YOLO-friendly yaml without pyyaml
    lines = [
        f"path: {data['path']}",
        "train: .",
        "val: .",
        f"nc: {data['nc']}",
        "names:",
    ]
    for i, n in enumerate(classes):
        lines.append(f"  {i}: {n}")
    (labels_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_yolo_det(stem: str, instances: list[dict], labels_dir: Path, classes: list[str]) -> None:
    lines = []
    for inst in instances:
        cid = ensure_class_index(classes, inst["label"])
        x1, y1, x2, y2 = inst["box"]
        xc = ((x1 + x2) / 2) / 1000.0
        yc = ((y1 + y2) / 2) / 1000.0
        w = (x2 - x1) / 1000.0
        h = (y2 - y1) / 1000.0
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_yolo_seg(stem: str, instances: list[dict], labels_dir: Path, classes: list[str]) -> None:
    lines = []
    for inst in instances:
        cid = ensure_class_index(classes, inst["label"])
        pts = inst.get("points")
        if not pts and inst.get("box"):
            x1, y1, x2, y2 = inst["box"]
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if not pts:
            continue
        coords = " ".join(f"{p[0]/1000.0:.6f} {p[1]/1000.0:.6f}" for p in pts)
        lines.append(f"{cid} {coords}")
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_labelme(
    stem: str,
    image_path: Path,
    instances: list[dict],
    labels_dir: Path,
    mode: str,
    w: int,
    h: int,
) -> None:
    shapes = []
    for inst in instances:
        if mode == "bbox" or (inst.get("box") and not inst.get("points")):
            x1, y1, x2, y2 = inst["box"]
            pts = [
                [to_px(x1, w), to_px(y1, h)],
                [to_px(x2, w), to_px(y2, h)],
            ]
            shape_type = "rectangle"
        else:
            pts = [[to_px(p[0], w), to_px(p[1], h)] for p in inst["points"]]
            shape_type = "polygon"
        shapes.append(
            {
                "label": inst["label"],
                "points": pts,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
            }
        )
    # relative path from Labels/ to image
    try:
        rel = os_path_rel(image_path, labels_dir)
    except Exception:
        rel = image_path.name
    doc = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": rel,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    (labels_dir / f"{stem}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def os_path_rel(image_path: Path, labels_dir: Path) -> str:
    import os

    return os.path.relpath(str(image_path.resolve()), str(labels_dir.resolve())).replace("\\", "/")


def export_voc(stem: str, image_path: Path, instances: list[dict], labels_dir: Path, w: int, h: int) -> None:
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = image_path.parent.name
    ET.SubElement(ann, "filename").text = image_path.name
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"
    for inst in instances:
        if "box" not in inst and inst.get("points"):
            xs = [p[0] for p in inst["points"]]
            ys = [p[1] for p in inst["points"]]
            box = [min(xs), min(ys), max(xs), max(ys)]
        else:
            box = inst["box"]
        x1, y1, x2, y2 = box
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = inst["label"]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(to_px(x1, w))
        ET.SubElement(bb, "ymin").text = str(to_px(y1, h))
        ET.SubElement(bb, "xmax").text = str(to_px(x2, w))
        ET.SubElement(bb, "ymax").text = str(to_px(y2, h))
    rough = ET.tostring(ann, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    # minidom adds xml declaration
    (labels_dir / f"{stem}.xml").write_bytes(pretty)


def export_raw(stem: str, instances: list[dict], labels_dir: Path) -> None:
    (labels_dir / f"{stem}.raw.json").write_text(
        json.dumps(instances, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_csv(rows_path: Path, image_name: str, instances: list[dict], w: int, h: int, mode: str) -> None:
    new_file = not rows_path.exists()
    with rows_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["filename", "label", "x_min", "y_min", "x_max", "y_max", "points_json"])
        for inst in instances:
            if inst.get("box"):
                x1, y1, x2, y2 = inst["box"]
            elif inst.get("points"):
                xs = [p[0] for p in inst["points"]]
                ys = [p[1] for p in inst["points"]]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                continue
            pts = json.dumps(inst.get("points") or [], ensure_ascii=False)
            writer.writerow(
                [
                    image_name,
                    inst["label"],
                    to_px(x1, w),
                    to_px(y1, h),
                    to_px(x2, w),
                    to_px(y2, h),
                    pts,
                ]
            )


def export_one(
    image_path: Path,
    raw_instances: list[dict],
    labels_dir: Path,
    fmt: str,
    mode: str,
    classes: list[str],
    also_raw: bool = True,
) -> list[dict]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    w, h = image_size(image_path)
    instances = []
    for obj in raw_instances:
        if not isinstance(obj, dict):
            continue
        norm = normalize_instance(obj, mode if mode != "auto" else ("polygon" if "points" in obj else "bbox"))
        if norm:
            instances.append(norm)

    stem = image_path.stem
    fmt = fmt.lower().strip()

    if also_raw:
        export_raw(stem, instances, labels_dir)

    if fmt in ("yolo", "yolo_det", "yolov5", "yolov8"):
        export_yolo_det(stem, instances, labels_dir, classes)
    elif fmt in ("yolo_seg", "yolo-seg", "yoloseg"):
        export_yolo_seg(stem, instances, labels_dir, classes)
    elif fmt in ("labelme", "labelme_json"):
        export_labelme(stem, image_path, instances, labels_dir, mode, w, h)
    elif fmt in ("voc", "pascal", "pascal_voc", "xml"):
        export_voc(stem, image_path, instances, labels_dir, w, h)
    elif fmt in ("csv",):
        append_csv(labels_dir / "labels.csv", image_path.name, instances, w, h, mode)
    elif fmt in ("raw", "json"):
        pass  # raw already written
    else:
        raise ValueError(f"Unknown export format: {fmt}")

    return instances


def build_coco(dataset: list[dict[str, Any]], labels_dir: Path, classes: list[str]) -> None:
    """dataset items: {file_name, width, height, instances}"""
    images = []
    annotations = []
    categories = [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(classes)]
    name_to_id = {n: i + 1 for i, n in enumerate(classes)}
    ann_id = 1
    for img_id, item in enumerate(dataset, start=1):
        images.append(
            {
                "id": img_id,
                "file_name": item["file_name"],
                "width": item["width"],
                "height": item["height"],
            }
        )
        w, h = item["width"], item["height"]
        for inst in item["instances"]:
            label = inst["label"]
            if label not in name_to_id:
                classes.append(label)
                name_to_id[label] = len(classes)
                categories.append({"id": name_to_id[label], "name": label, "supercategory": "object"})
            if inst.get("box"):
                x1, y1, x2, y2 = inst["box"]
                bx = to_px(x1, w)
                by = to_px(y1, h)
                bw = max(1, to_px(x2, w) - bx)
                bh = max(1, to_px(y2, h) - by)
                ann: dict[str, Any] = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": name_to_id[label],
                    "bbox": [bx, by, bw, bh],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            else:
                pts = inst["points"]
                flat = []
                xs, ys = [], []
                for p in pts:
                    px, py = to_px(p[0], w), to_px(p[1], h)
                    flat.extend([px, py])
                    xs.append(px)
                    ys.append(py)
                bx, by = min(xs), min(ys)
                bw, bh = max(xs) - bx, max(ys) - by
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": name_to_id[label],
                    "bbox": [bx, by, max(1, bw), max(1, bh)],
                    "segmentation": [flat],
                    "area": float(max(1, bw) * max(1, bh)),
                    "iscrowd": 0,
                }
            annotations.append(ann)
            ann_id += 1
    doc = {"images": images, "annotations": annotations, "categories": categories}
    (labels_dir / "annotations_coco.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export annotations to disk formats")
    ap.add_argument("--image", required=True, help="Source image path")
    ap.add_argument("--ann", required=True, help="Path to canonical JSON array file, or inline JSON string")
    ap.add_argument("--format", required=True, help="yolo|yolo_seg|labelme|voc|csv|raw|coco")
    ap.add_argument("--mode", default="bbox", choices=["bbox", "polygon"])
    ap.add_argument("--labels-dir", default="", help="Output Labels dir (default: <image_dir>/Labels)")
    ap.add_argument("--classes", default="", help="Comma-separated class order")
    args = ap.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    labels_dir = Path(args.labels_dir).expanduser() if args.labels_dir else image_path.parent / "Labels"
    labels_dir = labels_dir.resolve()
    labels_dir.mkdir(parents=True, exist_ok=True)

    ann_arg = args.ann
    if Path(ann_arg).expanduser().exists():
        raw = json.loads(Path(ann_arg).expanduser().read_text(encoding="utf-8"))
    else:
        raw = json.loads(ann_arg)
    if not isinstance(raw, list):
        raise SystemExit("Annotation must be a JSON array")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    classes_path = labels_dir / "classes.txt"
    if classes_path.exists() and not classes:
        classes = [ln.strip() for ln in classes_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    if args.format.lower() == "coco":
        # single-image coco append-ish: rebuild from this one (caller should use batch for full set)
        w, h = image_size(image_path)
        instances = []
        for obj in raw:
            if isinstance(obj, dict):
                n = normalize_instance(obj, args.mode)
                if n:
                    instances.append(n)
                    ensure_class_index(classes, n["label"])
        export_raw(image_path.stem, instances, labels_dir)
        build_coco(
            [{"file_name": image_path.name, "width": w, "height": h, "instances": instances}],
            labels_dir,
            classes,
        )
        write_classes(labels_dir, classes)
        print(f"Wrote COCO + raw for {image_path.name} → {labels_dir}")
        return

    instances = export_one(image_path, raw, labels_dir, args.format, args.mode, classes)
    if args.format.lower() in ("yolo", "yolo_det", "yolov5", "yolov8", "yolo_seg", "yolo-seg", "yoloseg"):
        write_classes(labels_dir, classes)
    print(f"Exported {len(instances)} instances ({args.format}) → {labels_dir}")


if __name__ == "__main__":
    main()
