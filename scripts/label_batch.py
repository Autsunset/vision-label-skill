#!/usr/bin/env python3
"""Batch label images with a vision model and export YOLO/Labelme/etc.

Supports rich class specs (name + description + example paths/URLs):

  python label_batch.py \\
    --images-dir ./images \\
    --format yolo \\
    --mode bbox \\
    --class-spec ./images/Labels/class_spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from vision_client import (  # noqa: E402
    analyze_image,
    extract_json_array,
    load_config,
    load_image,
    call_vision,
)
from export_labels import (  # noqa: E402
    build_coco,
    export_one,
    image_size,
    write_classes,
    ensure_class_index,
)
from class_spec import (  # noqa: E402
    build_class_definitions_block,
    build_example_caption_block,
    format_specs_table,
    load_class_spec,
    merge_names_and_spec,
    resolve_examples,
    save_class_spec,
    specs_from_names,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


def list_images(folder: Path) -> list[Path]:
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def build_prompt(
    mode: str,
    specs: list[dict[str, Any]],
    density_note: str,
    few_shot_note: str,
    example_captions: list[str],
) -> str:
    classes = [s["name"] for s in specs]
    class_list = "\n".join(f"- {c}" for c in classes) if classes else (
        "- (infer short labels — prefer concrete nouns)"
    )
    allowed = (
        "Allowed labels (use EXACTLY these strings):\n" + class_list
        if classes
        else "Labels: use short English nouns.\n"
    )
    shared = """Return ONLY a JSON array. No markdown, no prose, no code fences.

Coordinate system:
- All numbers are integers 0–1000.
- X first, then Y. NEVER y-first.
- Tight fit per instance. One object = one entry.
- No duplicate nested boxes for the same thing.
- Skip watermarks, pure shadows, and tiny unidentifiable blobs.
"""
    if mode == "polygon":
        schema = (
            'Schema:\n{"label":"<class>","points":[[x,y],[x,y],...]}\n'
            "- Closed outline implied (do not repeat first point).\n"
            "- 6–24 points typical; more only for irregular shapes.\n"
        )
    else:
        schema = (
            'Schema:\n{"label":"<class>","box":[x_min,y_min,x_max,y_max]}\n'
            "- x_min<x_max, y_min<y_max.\n"
        )
    density = density_note.strip() or (
        "Label every clearly visible instance of the target classes. "
        "Do not invent classes outside the allowed list. "
        "If nothing matches, return []."
    )

    parts = [
        "Object labeling for computer vision datasets.",
        shared,
        schema,
        allowed,
    ]

    defs = build_class_definitions_block(specs)
    if defs:
        parts.append(defs)

    ex_map = build_example_caption_block(example_captions)
    if ex_map:
        parts.append(ex_map)

    parts.append("DENSITY / SCOPE:\n" + density)

    few = few_shot_note.strip()
    if few:
        parts.append(few)

    parts.append(
        "SELF-CHECK:\n"
        "1) Valid JSON array only.\n"
        "2) Coordinates integers 0–1000, X then Y.\n"
        "3) Every label is an allowed class name (exact string).\n"
        "4) Prefer matches consistent with CLASS DEFINITIONS and example images.\n"
    )
    return "\n\n".join(parts)


def repair_prompt(mode: str) -> str:
    if mode == "polygon":
        schema = '[{"label":"name","points":[[x,y],[x,y],[x,y]]}]'
    else:
        schema = '[{"label":"name","box":[x_min,y_min,x_max,y_max]}]'
    return (
        "Your previous answer was not valid JSON matching the schema. "
        f"Reply again with ONLY a JSON array like {schema}. "
        "Coordinates integers 0–1000, X then Y. No markdown."
    )


def label_one(
    image_path: Path,
    prompt: str,
    mode: str,
    example_images: list[Path],
    retries: int = 1,
) -> list[dict[str, Any]]:
    text = prompt
    if example_images:
        text = (
            f"The first {len(example_images)} image(s) are CLASS EXAMPLES only (do not annotate them). "
            "Annotate ONLY the last image (the target).\n\n" + prompt
        )
        paths = [str(p) for p in example_images] + [str(image_path)]
        cfg = load_config()
        images = [load_image(p) for p in paths]
        raw_text = call_vision(cfg, images, text)
    else:
        raw_text = analyze_image(str(image_path), text)

    for attempt in range(retries + 1):
        try:
            return extract_json_array(raw_text)
        except ValueError:
            if attempt >= retries:
                print(
                    f"WARN: parse failed for {image_path.name}, saving empty. Raw head:\n{raw_text[:300]}",
                    file=sys.stderr,
                )
                return []
            # repair without re-sending all examples to save tokens
            raw_text = analyze_image(str(image_path), repair_prompt(mode))
    return []


def parse_extra_examples(arg: str) -> list[str]:
    """Comma-separated paths/URLs not tied to a class (legacy)."""
    return [p.strip() for p in arg.split(",") if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch vision labeling → Labels/")
    ap.add_argument("--images-dir", required=True, help="Folder of images to label")
    ap.add_argument("--format", required=True, help="yolo|yolo_seg|labelme|voc|csv|raw|coco")
    ap.add_argument("--mode", default="bbox", choices=["bbox", "polygon"])
    ap.add_argument("--classes", default="", help="Comma-separated class names (order = YOLO ids)")
    ap.add_argument(
        "--class-spec",
        default="",
        help="Path to class_spec.json with name/description/examples per class",
    )
    ap.add_argument(
        "--example-images",
        default="",
        help="Legacy: extra example paths/URLs (not class-tagged). Prefer --class-spec",
    )
    ap.add_argument("--density-note", default="", help="Extra density / must-label instructions")
    ap.add_argument("--few-shot-note", default="", help="Extra free-text visual cues")
    ap.add_argument("--prompt-file", default="", help="Full prompt file (overrides built prompt)")
    ap.add_argument("--labels-dir", default="", help="Default: <images-dir>/Labels")
    ap.add_argument("--limit", type=int, default=0, help="Only first N images (0=all)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if Labels/<stem>.raw.json exists")
    ap.add_argument("--dry-run", action="store_true", help="List images and print prompt only")
    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        print(f"ERROR: not a directory: {images_dir}", file=sys.stderr)
        sys.exit(1)

    labels_dir = Path(args.labels_dir).expanduser().resolve() if args.labels_dir else images_dir / "Labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    specs: list[dict[str, Any]] = []
    if args.class_spec:
        specs = load_class_spec(args.class_spec)
    elif (labels_dir / "class_spec.json").exists() and not names:
        specs = load_class_spec(labels_dir / "class_spec.json")

    if names and specs:
        specs = merge_names_and_spec(names, specs)
    elif names and not specs:
        specs = specs_from_names(names)
    elif not specs and not names:
        # try classes.txt
        cp = labels_dir / "classes.txt"
        if cp.exists():
            names = [ln.strip() for ln in cp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            specs = specs_from_names(names)

    classes = [s["name"] for s in specs]

    # resolve examples (paths + URLs)
    cache_dir = labels_dir / "_examples"
    example_paths, example_captions = resolve_examples(specs, cache_dir)

    # legacy untagged examples
    for ex in parse_extra_examples(args.example_images):
        if ex.startswith("http://") or ex.startswith("https://"):
            from class_spec import download_example

            p = download_example(ex, cache_dir)
            example_paths.append(p)
            example_captions.append("(unspecified class)")
        else:
            p = Path(ex).expanduser()
            if not p.exists():
                print(f"ERROR: example image missing: {p}", file=sys.stderr)
                sys.exit(1)
            example_paths.append(p)
            example_captions.append("(unspecified class)")

    # persist merged class_spec for reproducibility
    if specs:
        save_class_spec(specs, labels_dir / "class_spec.json")

    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    else:
        prompt = build_prompt(
            args.mode, specs, args.density_note, args.few_shot_note, example_captions
        )

    images = list_images(images_dir)
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    if not images:
        print(f"ERROR: no images in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Images: {len(images)} | format={args.format} mode={args.mode} → {labels_dir}")
    print("--- CLASS SPEC ---")
    print(format_specs_table(specs) if specs else "(no classes)")
    print(f"Example images resolved: {len(example_paths)}")
    if args.dry_run:
        print("--- PROMPT ---")
        print(prompt)
        print("--- FILES ---")
        for p in images:
            print(p.name)
        return

    load_config()

    coco_items: list[dict[str, Any]] = []
    failures: list[str] = []
    t0 = time.time()

    for i, img in enumerate(images, 1):
        raw_path = labels_dir / f"{img.stem}.raw.json"
        if args.skip_existing and raw_path.exists():
            print(f"[{i}/{len(images)}] skip existing {img.name}")
            try:
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
            except Exception:
                raw = []
        else:
            print(f"[{i}/{len(images)}] labeling {img.name} …")
            try:
                raw = label_one(img, prompt, args.mode, example_paths)
            except SystemExit:
                raise
            except Exception as e:
                print(f"ERROR on {img.name}: {e}", file=sys.stderr)
                failures.append(img.name)
                raw = []
            raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            if args.format.lower() == "coco":
                w, h = image_size(img)
                from export_labels import normalize_instance

                instances = []
                for obj in raw:
                    if isinstance(obj, dict):
                        n = normalize_instance(obj, args.mode)
                        if n:
                            instances.append(n)
                            ensure_class_index(classes, n["label"])
                coco_items.append(
                    {"file_name": img.name, "width": w, "height": h, "instances": instances}
                )
            else:
                instances = export_one(img, raw, labels_dir, args.format, args.mode, classes, also_raw=True)
                for inst in instances:
                    ensure_class_index(classes, inst["label"])
        except Exception as e:
            print(f"ERROR export {img.name}: {e}", file=sys.stderr)
            failures.append(img.name)

    if args.format.lower() in ("yolo", "yolo_det", "yolov5", "yolov8", "yolo_seg", "yolo-seg", "yoloseg"):
        write_classes(labels_dir, classes)
    if args.format.lower() == "coco":
        build_coco(coco_items, labels_dir, classes)
        write_classes(labels_dir, classes)

    meta = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "format": args.format,
        "mode": args.mode,
        "classes": classes,
        "class_spec": specs,
        "example_count": len(example_paths),
        "count": len(images),
        "failures": failures,
        "seconds": round(time.time() - t0, 2),
    }
    (labels_dir / "session_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "class_spec"}, ensure_ascii=False, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
