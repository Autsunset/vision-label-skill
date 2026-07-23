# vision-label-skill

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Claude Code Skill](https://img.shields.io/badge/Claude%20Code-Skill-7c3aed)](https://github.com/Autsunset/vision-label-skill)
[![Python 3](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Formats](https://img.shields.io/badge/export-YOLO%20%7C%20Labelme%20%7C%20VOC%20%7C%20COCO%20%7C%20CSV-0ea5e9)](references/formats.md)

**Claude Code skill** for batch **object detection / instance segmentation** labeling via any multimodal vision API. Export **YOLO · YOLO-seg · Labelme · VOC · COCO · CSV** under `<images_dir>/Labels/`.

> 用外部多模态 API 给图片文件夹批量自动标注，导出 YOLO / Labelme / VOC / COCO / CSV。类别完全自定义（螺丝、标签、商标、纸卷、缺陷……）。

**Repo:** [github.com/Autsunset/vision-label-skill](https://github.com/Autsunset/vision-label-skill)

---

## Features

- **Any classes** you define (name + optional description + example images / URLs)
- **bbox** or **polygon** (SAM-style multipoint outlines)
- **API styles:** OpenAI Chat Completions · OpenAI Responses · Anthropic Messages (and compatible gateways)
- Canonical model output: JSON array in **0–1000** coordinate space → exporters convert to training formats
- Re-export from `.raw.json` without calling the API again (if you keep raw files)
- Stdlib + **Pillow** only (no official OpenAI/Anthropic SDKs required)

## Install (Claude Code)

```bash
# User skills directory (example)
git clone https://github.com/Autsunset/vision-label-skill.git \
  ~/.claude/skills/vision-label-skill

cd ~/.claude/skills/vision-label-skill
pip install Pillow
cp env/.env.example env/.env
# edit env/.env with your API key / base / model / format
```

Windows (PowerShell example):

```powershell
git clone https://github.com/Autsunset/vision-label-skill.git "$env:USERPROFILE\.claude\skills\vision-label-skill"
cd "$env:USERPROFILE\.claude\skills\vision-label-skill"
pip install Pillow
copy env\.env.example env\.env
```

Layout:

```text
vision-label-skill/
├── SKILL.md                 # Claude Code skill instructions
├── LICENSE
├── env/
│   └── .env.example
├── scripts/
│   ├── vision_client.py
│   ├── label_batch.py
│   ├── class_spec.py
│   └── export_labels.py
├── references/
│   ├── formats.md
│   ├── prompts.md
│   └── class_spec.md
└── evals/
    └── evals.json
```

## Configure API

Edit `env/.env` (**never commit** this file):

```env
VISION_API_KEY=sk-...
VISION_API_BASE=https://api.openai.com/v1
VISION_MODEL=gpt-4o
# openai_chat | openai_responses | anthropic
VISION_API_FORMAT=openai_chat
VISION_MAX_TOKENS=8192
```

| `VISION_API_FORMAT` | Request path |
|---|---|
| `openai_chat` | `{base}/chat/completions` |
| `openai_responses` | `{base}/responses` |
| `anthropic` | `{base}/v1/messages` or `{base}/messages` |

Also supported: process env `VISION_*`, or `~/.claude/vision-config.json`.

```bash
python scripts/vision_client.py --config-check
```

## Use with Claude Code

Say for example:

> Label `./images` as YOLO boxes; classes screw, nut, washer

The skill will ask (if needed) for format, bbox vs polygon, class names / descriptions, and the image folder. Default folder preset is **`./images`** under the current working directory (not `datasets`).

After a batch finishes, it asks whether to **keep or delete** `Labels/*.raw.json`.

## CLI

```bash
# Recommended: class_spec with descriptions / examples
python scripts/label_batch.py \
  --images-dir "./images" \
  --format yolo \
  --mode bbox \
  --class-spec "./images/Labels/class_spec.json" \
  --skip-existing

# Names only
python scripts/label_batch.py \
  --images-dir "./images" \
  --format yolo \
  --mode bbox \
  --classes "screw,nut,washer" \
  --skip-existing

# Dry-run (count + prompt only)
python scripts/label_batch.py \
  --images-dir "./images" \
  --format yolo \
  --mode bbox \
  --classes "screw,nut,washer" \
  --dry-run
```

Re-export from existing `.raw.json` without another API call:

```bash
python scripts/export_labels.py \
  --image "./images/001.jpg" \
  --ann "./images/Labels/001.raw.json" \
  --format labelme \
  --mode bbox \
  --classes "screw,nut,washer"
```

## Output

| Path | Content |
|---|---|
| `Labels/<stem>.raw.json` | Canonical 0–1000 JSON (optional keep) |
| `Labels/<stem>.txt` | YOLO / YOLO-seg lines |
| `Labels/classes.txt` + `data.yaml` | Class id map |
| `Labels/<stem>.json` | Labelme (if format=labelme) |
| `Labels/session_meta.json` | Run summary |

Details: [`references/formats.md`](references/formats.md).

## Example `class_spec.json`

```json
{
  "classes": [
    {
      "name": "Label",
      "description": "White logistics shipping label on cartons, with barcode",
      "examples": ["./refs/label1.jpg", "https://example.com/label2.png"]
    },
    {
      "name": "Logo",
      "description": "Printed brand trademark on packaging",
      "examples": []
    }
  ]
}
```

## Security & open-source notes

- Secrets only in `env/.env` (gitignored). Never put keys in `SKILL.md` or commits.
- Docs/examples use relative paths like `./images/...`.
- Local `images/` and `Labels/` run outputs are gitignored — do not publish private photos or credentials.
- Spot-check labels; vision models can miss or invent boxes.

## License

[MIT](LICENSE) — use freely with your own datasets and API keys.
