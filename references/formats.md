# Annotation export formats

Canonical model output is always a JSON array of instances. Exporters convert that into on-disk formats under `<image_folder>/Labels/`.

## Canonical detection schema (bbox)

```json
[
  {"label": "person", "box": [120, 80, 640, 900]}
]
```

- `box`: integers **0–1000**, order **`[x_min, y_min, x_max, y_max]`** (never y-first).
- Tight box per instance; `x_min < x_max`, `y_min < y_max`.

## Canonical segmentation schema (polygon / SAM multi-point)

```json
[
  {"label": "person", "points": [[120, 80], [640, 90], [620, 900], [100, 880]]}
]
```

- Each point: integers **0–1000**, **`[x, y]`**.
- Closed polygon implied (do not require repeating first point).
- Prefer 6–24 points outlining the instance; not dense pixel masks.

---

## YOLO (detection)

**Files:** `Labels/<stem>.txt` + `Labels/classes.txt`

Each line:

```
class_id x_center y_center width height
```

All values normalized **0–1** relative to image size. Convert from 0–1000:

```
x_c = ((x_min + x_max) / 2) / 1000
y_c = ((y_min + y_max) / 2) / 1000
w   = (x_max - x_min) / 1000
h   = (y_max - y_min) / 1000
```

`classes.txt`: one class name per line, index = class_id.

Optional YAML (YOLOv5/v8 style) can be written as `Labels/data.yaml` with `names` and `nc`.

## YOLO-seg (instance polygons)

**Files:** `Labels/<stem>.txt` + `Labels/classes.txt`

```
class_id x1 y1 x2 y2 ... xn yn
```

Coordinates normalized 0–1 (`/1000`).

## Labelme (JSON)

**File:** `Labels/<stem>.json`

Shape:

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "person",
      "points": [[x_px, y_px], [x_px, y_px]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    }
  ],
  "imagePath": "../image.jpg",
  "imageData": null,
  "imageHeight": H,
  "imageWidth": W
}
```

- Bbox → `shape_type: "rectangle"`, two corner points in **pixel** coords.
- Polygon → `shape_type: "polygon"`, all vertices in **pixel** coords.
- Pixel convert: `px = round(v / 1000 * dim)`, clamp to `[0, dim-1]`.

## COCO (dataset JSON)

**File:** `Labels/annotations_coco.json` (dataset-level, not per image)

Minimal fields: `images`, `annotations`, `categories`. Boxes as COCO `[x, y, w, h]` in pixels; segmentation as polygon lists if mode is seg.

## VOC XML (Pascal VOC)

**File:** `Labels/<stem>.xml`

`<object><name>…</name><bndbox><xmin>…</xmin>…` in pixels. Detection only.

## CSV (simple)

**File:** `Labels/labels.csv`

```
filename,label,x_min,y_min,x_max,y_max
```

Pixel coords. One row per instance.

## Internal raw (debug)

**File:** `Labels/<stem>.raw.json`

Exact model JSON array (0–1000 space) for re-export without re-calling the API.
