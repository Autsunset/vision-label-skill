#!/usr/bin/env python3
"""Rich class specifications: name + description + example images (path or URL)."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def normalize_spec(data: Any) -> list[dict[str, Any]]:
    """Return list of {name, description, examples: [str]}."""
    if data is None:
        return []
    if isinstance(data, dict) and "classes" in data:
        items = data["classes"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # map name -> description or object
        items = []
        for k, v in data.items():
            if k == "classes":
                continue
            if isinstance(v, str):
                items.append({"name": k, "description": v, "examples": []})
            elif isinstance(v, dict):
                items.append({"name": k, **v})
            else:
                items.append({"name": k, "description": str(v), "examples": []})
    else:
        raise ValueError("class_spec must be a list or {classes:[...]}")

    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append({"name": item.strip(), "description": "", "examples": []})
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("label") or item.get("class") or "").strip()
        if not name:
            continue
        desc = str(item.get("description") or item.get("desc") or item.get("definition") or "").strip()
        ex = item.get("examples") or item.get("example") or item.get("images") or []
        if isinstance(ex, str):
            ex = [ex]
        examples = [str(x).strip() for x in ex if str(x).strip()]
        out.append({"name": name, "description": desc, "examples": examples})
    return out


def load_class_spec(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"class_spec not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return normalize_spec(data)


def specs_from_names(names: list[str]) -> list[dict[str, Any]]:
    return [{"name": n, "description": "", "examples": []} for n in names if n.strip()]


def merge_names_and_spec(names: list[str], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer order of names if given; overlay descriptions/examples from specs."""
    by_name = {s["name"]: s for s in specs}
    if names:
        merged = []
        for n in names:
            if n in by_name:
                merged.append(by_name[n])
            else:
                merged.append({"name": n, "description": "", "examples": []})
        # append specs not in names
        for s in specs:
            if s["name"] not in names:
                merged.append(s)
        return merged
    return specs or []


def save_class_spec(specs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"classes": specs}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def download_example(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    # guess extension
    path_part = urlparse(url).path
    ext = Path(path_part).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        ext = ".jpg"
    dest = cache_dir / f"url_{h}{ext}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "vision-label-skill/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "png" in ctype:
                dest = cache_dir / f"url_{h}.png"
            elif "webp" in ctype:
                dest = cache_dir / f"url_{h}.webp"
            elif "gif" in ctype:
                dest = cache_dir / f"url_{h}.gif"
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download example {url}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download example {url}: {e.reason}") from e
    dest.write_bytes(data)
    return dest


def resolve_examples(
    specs: list[dict[str, Any]],
    cache_dir: Path,
) -> tuple[list[Path], list[str]]:
    """
    Resolve all example paths/URLs.
    Returns (paths_in_order, captions_aligned_with_paths) where caption is class name.
    """
    paths: list[Path] = []
    captions: list[str] = []
    for spec in specs:
        name = spec["name"]
        for ex in spec.get("examples") or []:
            if is_url(ex):
                p = download_example(ex, cache_dir)
            else:
                p = Path(ex).expanduser()
                if not p.exists():
                    print(f"WARN: example missing for class '{name}': {ex}", file=sys.stderr)
                    continue
            paths.append(p)
            captions.append(name)
    return paths, captions


def build_class_definitions_block(specs: list[dict[str, Any]]) -> str:
    if not specs:
        return ""
    lines = ["CLASS DEFINITIONS (use EXACT name strings as labels):"]
    for i, s in enumerate(specs):
        lines.append(f"\n[{i}] name: {s['name']}")
        if s.get("description"):
            lines.append(f"    meaning: {s['description']}")
        else:
            lines.append("    meaning: (no extra description; use common sense for this name)")
        n_ex = len(s.get("examples") or [])
        if n_ex:
            lines.append(f"    reference images: {n_ex} provided (see example images in this request)")
    lines.append(
        "\nMatch targets to these definitions. Prefer the listed names only. "
        "If uncertain between two classes, pick the closer definition."
    )
    return "\n".join(lines)


def build_example_caption_block(captions: list[str]) -> str:
    if not captions:
        return ""
    lines = ["EXAMPLE IMAGE MAP (do NOT annotate these; they are references only):"]
    for i, c in enumerate(captions, 1):
        lines.append(f"  example image #{i} → class `{c}`")
    lines.append(
        "The LAST image in this request is the TARGET to annotate. "
        "Find objects that look like the referenced classes."
    )
    return "\n".join(lines)


def format_specs_table(specs: list[dict[str, Any]]) -> str:
    rows = ["| id | name | description | examples |", "|---:|---|---|---|"]
    for i, s in enumerate(specs):
        desc = (s.get("description") or "").replace("\n", " ")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        n = len(s.get("examples") or [])
        rows.append(f"| {i} | {s['name']} | {desc or '—'} | {n} |")
    return "\n".join(rows)


def main() -> None:
    """CLI: validate / pretty-print a class_spec.json"""
    if len(sys.argv) < 2:
        print("Usage: class_spec.py <class_spec.json>")
        sys.exit(0)
    specs = load_class_spec(sys.argv[1])
    print(format_specs_table(specs))
    print(json.dumps({"classes": specs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
