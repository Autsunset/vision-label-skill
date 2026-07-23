---
name: vision-label-skill
description: >
  Batch object-detection / instance-segmentation labeling via an external vision API,
  exporting YOLO, YOLO-seg, Labelme, VOC XML, COCO, or CSV under <image_folder>/Labels/.
  Use this skill whenever the user wants to auto-label images, generate YOLO labels,
  Labelme JSON, VOC XML, COCO annotations, SAM-style polygon masks, detection boxes,
  or build a CV training dataset from a folder of photos — even if they say "标注",
  "打框", "分割标注", "导出标签", "label dataset", or only mention a labels format.
  Always collect format + mode + classes + image folder from the user first (do not
  assume portrait/person-parts classes). Works with openai_chat (/chat/completions),
  openai_responses (/responses), and anthropic (/messages) APIs configured in the
  skill's env/.env.
---

# Vision Label Skill — Dataset Auto-Labeling

Turn a folder of images into training labels using a multimodal vision model. Canonical model output is a **JSON array in 0–1000 coordinate space**; scripts convert it to YOLO / Labelme / VOC / COCO / CSV under **`<images_dir>/Labels/`**.

**Classes are whatever the user specifies** — industrial parts (screw, nut, paper roll), retail (label, logo, trademark), household items, animals, etc. Do **not** default to the person/face/parts checklist unless the user asks for that grain.

## When to use

- Auto-annotate a folder for detection or segmentation
- Export **yolo / yolo_seg / labelme / voc / coco / csv**
- User mentions 标注 / 打标签 / 数据集 / bounding box / polygon / SAM multi-point

## Prerequisites

1. **API config** in this skill's `env/` folder (preferred):

```bash
cp env/.env.example env/.env
# edit env/.env
```

| Variable | Meaning |
|---|---|
| `VISION_API_KEY` | Required |
| `VISION_API_BASE` | Base URL without path suffix (or with `/v1`) |
| `VISION_MODEL` | Model id |
| `VISION_API_FORMAT` | `openai_chat` → `/chat/completions` · `openai_responses` → `/responses` · `anthropic` → `/messages` |
| `VISION_MAX_TOKENS` | Optional, default 8192 |
| `VISION_API_URL` | Optional full endpoint override |

Legacy aliases: `openai`→`openai_chat`, `responses`→`openai_responses`, `messages`→`anthropic`.

Fallback: process env `VISION_*` or `~/.claude/vision-config.json` (same keys as vision-skill).

2. **Python 3** + **Pillow**:

```bash
pip install Pillow
```

Scripts use only stdlib + Pillow (no `openai`/`anthropic` SDKs).

## Mandatory interview (before labeling)

**Ask first — do not invent format or classes. Do not dump a giant markdown questionnaire.**

### How to ask (critical UX)

1. **Use `AskUserQuestion`** with selectable options. Prefer **one tool call with 2–4 questions** so the user answers in one pass, then you process once.
2. **Never** interleave long analysis between single answers (“选一次思考一次”). After a partial answer, either:
   - immediately ask the remaining questions in the **next** `AskUserQuestion` with **no** long preamble, or
   - if everything needed is already known, jump to confirm + run.
3. **Do not** restate a full multi-step table in chat when the chooser UI can carry the steps.
4. Fixed topic order (can pack into 1–2 AskUserQuestion rounds):

| Topic | Options / input |
|---|---|
| Export format | YOLO / YOLO-seg / Labelme / VOC·COCO·CSV |
| Geometry mode | bbox / polygon |
| Class names | short names (Other for custom list) |
| Class enrichment | later: description + optional example path/URL; allow「跳过」 |
| Images folder | see default options below — **not** `datasets` |
| Scope (optional) | full vs first N — only if useful |

**Images folder — default chooser options (use these labels, not `datasets`):**

1. **`./images`（当前终端工作目录下的 `images/` 子文件夹）** — primary default  
2. **skill 自带 `images/`** — only if that folder exists and differs from cwd  
3. **自定义路径（Other）** — user pastes absolute/relative path  

Do **not** offer `./datasets` as a preset. Prefer cwd-relative `./images`.

**Recommended packing:**

- **Round A (one AskUserQuestion):** format + mode + class names + image folder path  
- **Round B (only if classes need clarity):** enrichment — what each class is + example images; allow skip  
- **Then:** short confirm summary → write `class_spec.json` → dry-run/run  

If the user already stated some fields in free text, skip those topics.

### Class enrichment (after names exist)

Echo ids `0=A, 1=B…`, then one short ask (tool or one chat line): description + optional path/URL;「跳过」OK.  
Read `references/class_spec.md` for schema. Bare names alone are weak — offer enrichment once, never force.

## Workflow

### 1. Confirm config

```bash
python scripts/vision_client.py --config-check
```

(from skill root, or absolute path to the script)

### 2. Build job parameters + `class_spec.json`

From the interview write `Labels/class_spec.json` (or any path):

```json
{
  "classes": [
    {
      "name": "Label",
      "description": "White logistics shipping label on cartons, with barcode",
      "examples": ["./refs/label1.jpg", "https://example.com/label2.png"]
    }
  ]
}
```

Then run with:

```bash
python scripts/label_batch.py \
  --images-dir "/path/to/images" \
  --format yolo \
  --mode bbox \
  --class-spec "/path/to/images/Labels/class_spec.json" \
  --density-note "..."
```

Or still use `--classes "A,B,C"` for names-only; prefer `--class-spec` when descriptions/examples exist.

Scripts will:

- inject **CLASS DEFINITIONS** (name + description) into the vision prompt
- download URL examples into `Labels/_examples/`
- send example images first (tagged by class) + target image last

Read `references/prompts.md` / `references/class_spec.md` for prompt details.

### 3. Dry-run (recommended)

```bash
python scripts/label_batch.py \
  --images-dir "/path/to/images" \
  --format yolo \
  --mode bbox \
  --classes "screw,nut,washer" \
  --dry-run
```

Show the user image count + prompt summary; proceed if OK.

### 4. Run batch

```bash
python scripts/label_batch.py \
  --images-dir "/path/to/images" \
  --format yolo \
  --mode bbox \
  --class-spec "/path/to/images/Labels/class_spec.json" \
  --density-note "Label every clear instance of the defined classes." \
  --skip-existing
```

Names-only fallback:

```bash
python scripts/label_batch.py \
  --images-dir "/path/to/images" \
  --format yolo \
  --mode bbox \
  --classes "screw,nut,washer" \
  --skip-existing
```

Outputs always go to **`<images_dir>/Labels/`** (or `--labels-dir`):

| Format | Files |
|---|---|
| `yolo` | `<stem>.txt` + `classes.txt` + `data.yaml` |
| `yolo_seg` | polygon YOLO lines + `classes.txt` |
| `labelme` | `<stem>.json` (rectangle or polygon shapes) |
| `voc` | `<stem>.xml` |
| `coco` | `annotations_coco.json` |
| `csv` | `labels.csv` |
| always | `<stem>.raw.json` (canonical 0–1000 JSON) + `session_meta.json` |

### 5. Multiple formats

Run export again from raw without re-calling the API:

```bash
python scripts/export_labels.py \
  --image "/path/to/images/img001.jpg" \
  --ann "/path/to/images/Labels/img001.raw.json" \
  --format labelme \
  --mode bbox \
  --classes "screw,nut,washer"
```

Or re-run `label_batch.py` with `--skip-existing` and a different `--format` (re-exports from existing `.raw.json` when skip loads raw — actually skip only skips API; for re-export prefer `export_labels.py` per file, or a small loop).

Re-export all raws:

```bash
# bash example
for f in "/path/to/images/Labels"/*.raw.json; do
  stem=$(basename "$f" .raw.json)
  # find matching image
  img=$(ls "/path/to/images/$stem".* 2>/dev/null | head -1)
  python scripts/export_labels.py --image "$img" --ann "$f" --format labelme --mode bbox --classes "a,b,c"
done
```

### 6. Report to user

Summarize: image count, success/fail list, `Labels/` path, class list + YOLO ids, sample of 1–2 annotations. Mention they should spot-check labels (vision models can miss or hallucinate boxes).

### 7. After run — ask about `.raw.json` (mandatory)

`.raw.json` is the canonical model dump (useful for re-export / debug). Many users only need YOLO/Labelme/etc. and want a cleaner `Labels/` folder.

**After every successful batch (or after reporting skip-heavy runs), use `AskUserQuestion`:**

| Question | Options |
|---|---|
| 是否删除 `Labels/*.raw.json`？ | **保留**（可日后换格式重导出） / **删除全部 raw** / **只删本次新生成的 raw** |

Rules:

- Default recommendation in the chooser: **保留** if the user may re-export; still **ask**, do not delete silently.
- If user chooses delete: remove only under the job’s `labels_dir`, only `*.raw.json` (never delete `.txt` / Labelme / `classes.txt` / `data.yaml` / `class_spec.json` / `session_meta.json` unless they ask).
- Confirm count deleted in the next short message.

**Open-source / privacy:** never embed personal absolute paths, usernames, private photo sets, or API keys in prompts, docs, commits, or sample data.

## Coordinate contract (model)

Detection:

```json
[{"label":"screw","box":[120,80,200,160]}]
```

Segmentation:

```json
[{"label":"screw","points":[[120,80],[200,90],[190,160],[110,150]]}]
```

- Integers **0–1000**, **X then Y**
- Exporters convert to pixels / YOLO 0–1 (see `references/formats.md`)

## Script map

| Script | Role |
|---|---|
| `scripts/vision_client.py` | API client: chat / responses / messages + JSON extract |
| `scripts/label_batch.py` | Batch label folder → Labels/ (supports `--class-spec`) |
| `scripts/class_spec.py` | Load/save/resolve class name+description+examples (URL download) |
| `scripts/export_labels.py` | Single-image export / re-export from raw JSON |

## Error handling

- Missing API key → tell user to copy `env/.env.example` → `env/.env`
- HTTP errors → show body; check `VISION_API_FORMAT` matches the gateway (`/chat/completions` vs `/responses` vs `/messages`)
- Parse failures → script retries once with a repair prompt; empty array + warning if still bad; `.raw.json` may be `[]`
- No Pillow → `pip install Pillow`

## Tips

- Prefer **English short class names** for YOLO interoperability; store a Chinese mapping in chat or a `classes_zh.txt` if needed.
- For logos/trademarks, provide **example crops** via `--example-images`.
- Start with `--limit 3` on a new job to validate quality before full folder.
- `VISION_API_FORMAT=anthropic` works with official Anthropic and many Claude-compatible gateways; if the gateway wants `Authorization: Bearer` instead of `x-api-key`, set `VISION_ANTHROPIC_USE_BEARER=true`.
- Do not put secrets in SKILL.md; only in `env/.env` (gitignored).
- Keep docs, prompts, evals, and examples free of personal usernames, machine absolute paths, private photo sets, and real API keys (use placeholders like `./images/...`, `sk-your-key-here`).
- After labeling, always ask whether to keep or delete `Labels/*.raw.json`.

## What NOT to do

- Do **not** silently use the portrait person-parts checklist when the user needs screws, labels, paper rolls, etc.
- Do **not** start labeling without confirmed **format + mode + classes + images path**.
- Do **not** write labels outside `<images_dir>/Labels/` unless the user overrides `--labels-dir`.
- Do **not** offer `./datasets` as a default images-folder preset (use `./images` under the current working directory).
- Do **not** delete `.raw.json` without asking.
- Do **not** commit `env/.env`, personal `images/`, or run artifacts under `Labels/` when publishing the skill.
