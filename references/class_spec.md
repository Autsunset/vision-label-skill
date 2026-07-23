# Class specifications (rich class definitions)

A **class name alone is often ambiguous** (e.g. `Label` could mean price tag, shipping label, UI label). This skill supports **rich class specs**:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Exact string written into annotations / YOLO id order |
| `description` | no | What the object is, where it appears, color/shape/text cues, what NOT to confuse it with |
| `examples` | no | Local paths and/or http(s) image URLs showing typical instances |

## Interview flow (mandatory when collecting classes)

After the user lists short names (`A, B, C` or `Label, Logo`):

1. **Echo the list** with YOLO ids: `0=A, 1=B, …`
2. **Invite enrichment for each class** (do not force if user says skip):
   - "A 具体是什么？颜色/形状/位置/别和什么搞混？"
   - "有没有 A 的示例图？本地路径或 URL 都可以"
3. Accept answers in any free form, e.g.:
   - `A=纸箱上的白色物流面单，长方形条码标签`
   - `A 示例: https://example.com/a.jpg` or `./refs/a1.png`
   - One message covering all classes is fine
4. **Summarize** a class table before labeling and get a quick OK:
   ```
   | id | name  | description (short)     | examples |
   | 0  | Label | 纸箱白色物流面单…        | 1 local  |
   ```
5. Only then run dry-run / batch.

If the user only gives names and refuses details, proceed with names only — but still ask once.

## On-disk format: `class_spec.json`

Write under `Labels/class_spec.json` (or pass path via `--class-spec`):

```json
{
  "classes": [
    {
      "name": "Label",
      "description": "White rectangular logistics shipping label on cardboard boxes, usually with barcode and printed address. NOT product brand logos.",
      "examples": [
        "./refs/label_example1.jpg",
        "https://example.com/label2.jpg"
      ]
    },
    {
      "name": "Logo",
      "description": "Printed brand trademark mark on packaging.",
      "examples": []
    }
  ]
}
```

Rules:

- `name` order = YOLO class id (0, 1, 2, …)
- `examples` may mix file paths and URLs
- Empty `description` / `examples` allowed

## How scripts use specs

1. Build prompt section **CLASS DEFINITIONS** with name + description for every class.
2. Download URL examples to a temp/cache dir under `Labels/_examples/`.
3. Attach example images to the vision request (examples first, target last), with text:
   - "Images 1..K are class examples (do not annotate them). Image K+1 is the target."
4. Optionally prefix example images with which class they belong to in the prompt.

## Free-text parsing hints for Claude

When the user writes casually, map into specs:

- `A,B,C` → three names
- `A是红色螺丝` → description for A
- `A: https://...` or `A示例图 xxx.jpg` → examples for A
- Chinese class names OK if user wants them as the label string

Always keep **`name` exactly** as the annotation label string the user confirmed.
