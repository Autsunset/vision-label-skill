# Vision labeling prompts

Build the final vision prompt from: task mode (bbox | polygon), class list, optional few-shot notes, and density policy.

Use only **generic** class examples in docs and default templates (e.g. Label, Logo, screw). Do not inject personal paths, usernames, private filenames, or real API credentials into prompts.

## Shared rules (always inject)

```
Return ONLY a JSON array. No markdown, no prose, no code fences.

Coordinate system:
- All numbers are integers 0–1000.
- X first, then Y. NEVER [y,x,...].
- Boxes: [x_min, y_min, x_max, y_max] with x_min<x_max, y_min<y_max.
- Polygons: "points": [[x,y], ...] clockwise or counter-clockwise outline, 6–24 points typical.
- Tight fit per instance. One object = one entry. No duplicate nested boxes for the same thing.
```

## Detection (bbox) template

```
Object detection labeling. Return ONLY a JSON array.

Schema:
{"label":"<class from allowed list>","box":[x_min,y_min,x_max,y_max]}

Allowed labels (use EXACTLY these strings):
{CLASS_LIST}

{DENSITY_AND_CHECKLIST}

{FEW_SHOT_NOTE}

SELF-CHECK before answer:
1) Every label is in the allowed list.
2) Every box is [x_min,y_min,x_max,y_max] integers 0–1000.
3) No y-first coordinates.
4) No empty array unless the image truly has zero target objects.
```

## Segmentation (polygon / multi-point) template

```
Instance segmentation labeling (polygon outline). Return ONLY a JSON array.

Schema:
{"label":"<class from allowed list>","points":[[x,y],[x,y],...]}

Allowed labels (use EXACTLY these strings):
{CLASS_LIST}

Rules:
- Outline each visible instance tightly with a closed polygon (do not repeat the first point).
- Prefer outer silhouette; skip internal holes unless critical.
- 6–24 points per instance for most objects; more only for highly irregular shapes.
- Same density spirit as detection: label every clearly visible target instance of allowed classes.
- Do NOT output pixel masks or RLE.

{FEW_SHOT_NOTE}

SELF-CHECK:
1) Allowed labels only.
2) Points are [x,y] integers 0–1000.
3) At least 3 points per instance.
```

## Default density checklist (when user wants fine-grained person/parts labeling)

Use the user's class list. If they adopt a portrait/parts policy similar to the source prompt, inject:

```
DENSITY TARGET
- Portrait / selfie / single person: about 8–12 boxes when parts classes are enabled.
- Multi-person: about 6–10 per clearly visible person, cap ~20 total unless user overrides.
- Landscape / no person: 4–10 large salient objects of allowed classes only.

MUST-LABEL when visible AND in allowed list:
person, hat/cap, face, eye (L/R), nose, mouth, main upper garment, hand(s), large background furniture/art if listed.

DO NOT LABEL unless user explicitly added those classes:
brows, lashes, pupils, teeth separate from mouth, jewelry, logos, fingers separate from hand, shadows, watermarks.
```

When the user supplies a simple closed set (e.g. `cat, dog, car`), **do not** force the person-parts checklist — only label allowed classes with sensible instance density (all clear instances, no micro-parts).

## Class definitions block (preferred over bare names)

Always inject when available:

```
CLASS DEFINITIONS (use EXACT name strings as labels):

[0] name: Label
    meaning: White logistics shipping label on cartons, barcode present. NOT brand logos.
    reference images: 2 provided

[1] name: Logo
    meaning: Printed brand trademark on packaging.
```

## Example image map

When example images are attached (before the target):

```
EXAMPLE IMAGE MAP (do NOT annotate these; they are references only):
  example image #1 → class `Label`
  example image #2 → class `Logo`
The LAST image is the TARGET to annotate.
```

## Few-shot note (optional free text)

```
CLASS VISUAL CUES (extra):
- Label: usually bottom-right of box, matte paper
- Logo: glossy print, brand colors
```

## Repair prompt (if parse fails)

```
Your previous answer was not valid JSON matching the schema. Reply again with ONLY the JSON array.
No markdown. Schema: {SCHEMA_ONE_LINE}
```
