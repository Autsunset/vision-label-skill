#!/usr/bin/env python3
"""Vision API client for vision-label-skill.

Supports:
  - openai_chat      → POST {base}/chat/completions
  - openai_responses → POST {base}/responses
  - anthropic        → POST {base}/v1/messages or {base}/messages

Config: skill env/.env (preferred) or ~/.claude/vision-config.json / VISION_* env vars.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

# skill root = parent of scripts/
SKILL_ROOT = Path(__file__).resolve().parent.parent
ENV_CANDIDATES = [
    SKILL_ROOT / "env" / ".env",
    SKILL_ROOT / "env" / "vision.env",
    SKILL_ROOT / ".env",
    Path.home() / ".claude" / "vision-config.json",
]

OPENAI_DEFAULTS = {
    "api_base": "https://api.openai.com/v1",
    "model": "gpt-4o",
}
ANTHROPIC_DEFAULTS = {
    "api_base": "https://api.anthropic.com",
    "model": "claude-sonnet-4-6",
}

FORMAT_ALIASES = {
    "openai": "openai_chat",
    "chat": "openai_chat",
    "chat_completions": "openai_chat",
    "openai_chat": "openai_chat",
    "/chat/completions": "openai_chat",
    "responses": "openai_responses",
    "openai_responses": "openai_responses",
    "/responses": "openai_responses",
    "anthropic": "anthropic",
    "messages": "anthropic",
    "/messages": "anthropic",
    "claude": "anthropic",
}


def _parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        out[key] = val
    return out


def load_config() -> dict[str, Any]:
    """Load API config. Priority: process env > skill env/.env > vision-config.json."""
    file_cfg: dict[str, Any] = {}

    for p in ENV_CANDIDATES:
        if not p.exists():
            continue
        if p.suffix.lower() == ".json":
            with p.open("r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            break
        # dotenv
        parsed = _parse_dotenv(p)
        # map common keys
        file_cfg = {
            "api_key": parsed.get("VISION_API_KEY") or parsed.get("API_KEY") or parsed.get("api_key", ""),
            "api_base": parsed.get("VISION_API_BASE") or parsed.get("API_BASE") or parsed.get("api_base", ""),
            "model": parsed.get("VISION_MODEL") or parsed.get("MODEL") or parsed.get("model", ""),
            "api_format": parsed.get("VISION_API_FORMAT") or parsed.get("API_FORMAT") or parsed.get("api_format", ""),
            "max_tokens": parsed.get("VISION_MAX_TOKENS") or parsed.get("max_tokens", 8192),
            "api_url": parsed.get("VISION_API_URL") or parsed.get("api_url", ""),
        }
        break

    api_key = os.environ.get("VISION_API_KEY") or file_cfg.get("api_key") or ""
    api_base = os.environ.get("VISION_API_BASE") or file_cfg.get("api_base") or ""
    model = os.environ.get("VISION_MODEL") or file_cfg.get("model") or ""
    raw_fmt = (
        os.environ.get("VISION_API_FORMAT")
        or file_cfg.get("api_format")
        or "openai_chat"
    )
    api_format = FORMAT_ALIASES.get(str(raw_fmt).lower().strip(), str(raw_fmt).lower().strip())
    max_tokens = os.environ.get("VISION_MAX_TOKENS") or file_cfg.get("max_tokens") or 8192
    api_url = os.environ.get("VISION_API_URL") or file_cfg.get("api_url") or ""
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 8192

    if not api_key:
        env_hint = SKILL_ROOT / "env" / ".env"
        print(
            "ERROR: No API key configured.\n"
            f"Create {env_hint} from env/.env.example, or set VISION_API_KEY.\n"
            "Required: VISION_API_KEY, VISION_API_BASE, VISION_MODEL, VISION_API_FORMAT\n"
            "Formats: openai_chat | openai_responses | anthropic",
            file=sys.stderr,
        )
        sys.exit(2)

    if api_format not in ("openai_chat", "openai_responses", "anthropic"):
        print(
            f"ERROR: Unknown VISION_API_FORMAT '{raw_fmt}'.\n"
            "Use: openai_chat (/chat/completions), openai_responses (/responses), anthropic (/messages)",
            file=sys.stderr,
        )
        sys.exit(2)

    if api_format in ("openai_chat", "openai_responses"):
        api_base = api_base or OPENAI_DEFAULTS["api_base"]
        model = model or OPENAI_DEFAULTS["model"]
    else:
        api_base = api_base or ANTHROPIC_DEFAULTS["api_base"]
        model = model or ANTHROPIC_DEFAULTS["model"]

    return {
        "api_key": api_key,
        "api_base": str(api_base).rstrip("/"),
        "model": model,
        "api_format": api_format,
        "max_tokens": max_tokens,
        "api_url": str(api_url).strip() if api_url else "",
    }


def load_image(path: str | Path) -> tuple[str, str]:
    path_s = os.path.expanduser(str(path))
    if path_s.startswith("data:image/"):
        header, _, b64data = path_s.partition(",")
        media_type = header.split(":")[1].split(";")[0]
        return b64data, media_type

    p = Path(path_s)
    if not p.exists():
        print(f"ERROR: Image file not found: {p}", file=sys.stderr)
        sys.exit(1)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        print(f"ERROR: Unsupported image format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTS))}", file=sys.stderr)
        sys.exit(1)
    raw = p.read_bytes()
    b64data = base64.b64encode(raw).decode("ascii")
    media_type = mimetypes.guess_type(str(p))[0] or "image/png"
    return b64data, media_type


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: API request failed (HTTP {e.code}) {url}:\n{body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: Could not reach API at {url}:\n{e.reason}", file=sys.stderr)
        sys.exit(1)


def _extract_text_openai_chat(result: dict[str, Any]) -> str:
    content = result["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


def _extract_text_openai_responses(result: dict[str, Any]) -> str:
    # Prefer convenience field
    if isinstance(result.get("output_text"), str) and result["output_text"].strip():
        return result["output_text"]
    chunks: list[str] = []
    for item in result.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text") and part.get("text"):
                    chunks.append(str(part["text"]))
        elif item.get("type") in ("output_text", "text") and item.get("text"):
            chunks.append(str(item["text"]))
    if chunks:
        return "\n".join(chunks)
    # fallback
    return json.dumps(result, ensure_ascii=False)


def _extract_text_anthropic(result: dict[str, Any]) -> str:
    return "\n".join(
        block.get("text", "")
        for block in result.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def call_vision(
    cfg: dict[str, Any],
    images: list[tuple[str, str]],
    prompt: str,
) -> str:
    """Call vision model with one or more (b64, media_type) images + text prompt."""
    fmt = cfg["api_format"]
    if fmt == "openai_chat":
        return _call_openai_chat(cfg, images, prompt)
    if fmt == "openai_responses":
        return _call_openai_responses(cfg, images, prompt)
    return _call_anthropic(cfg, images, prompt)


def _call_openai_chat(cfg: dict[str, Any], images: list[tuple[str, str]], prompt: str) -> str:
    url = cfg["api_url"] or (cfg["api_base"] + "/chat/completions")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64, media_type in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            }
        )
    payload = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "messages": [{"role": "user", "content": content}],
    }
    result = post_json(url, headers, payload)
    return _extract_text_openai_chat(result)


def _call_openai_responses(cfg: dict[str, Any], images: list[tuple[str, str]], prompt: str) -> str:
    url = cfg["api_url"] or (cfg["api_base"] + "/responses")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for b64, media_type in images:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{b64}",
            }
        )
    payload = {
        "model": cfg["model"],
        "max_output_tokens": cfg["max_tokens"],
        "input": [{"role": "user", "content": content}],
    }
    result = post_json(url, headers, payload)
    return _extract_text_openai_responses(result)


def _call_anthropic(cfg: dict[str, Any], images: list[tuple[str, str]], prompt: str) -> str:
    base = cfg["api_base"]
    if cfg["api_url"]:
        url = cfg["api_url"]
    elif base.endswith("/v1"):
        url = base + "/messages"
    elif base.rstrip("/").endswith("messages"):
        url = base
    else:
        # default Anthropic path
        url = base + "/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": cfg["api_key"],
        "anthropic-version": "2023-06-01",
    }
    # Some OpenAI-compatible gateways expect Bearer for Anthropic-format too
    if os.environ.get("VISION_ANTHROPIC_USE_BEARER", "").lower() in ("1", "true", "yes"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    content: list[dict[str, Any]] = []
    for b64, media_type in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "messages": [{"role": "user", "content": content}],
    }
    result = post_json(url, headers, payload)
    return _extract_text_anthropic(result)


def analyze_image(image_path: str | Path, prompt: str, extra_image_paths: list[str | Path] | None = None) -> str:
    cfg = load_config()
    images = [load_image(image_path)]
    for p in extra_image_paths or []:
        images.append(load_image(p))
    return call_vision(cfg, images, prompt)


def extract_json_array(text: str) -> list[Any]:
    """Best-effort extract a JSON array from model output."""
    text = text.strip()
    # strip fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("objects", "annotations", "labels", "shapes", "instances"):
                if isinstance(data.get(k), list):
                    return data[k]
    except json.JSONDecodeError:
        pass
    # find first [...]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON array from model output:\n{text[:500]}")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  vision_client.py <image_path> [prompt]\n"
            "  vision_client.py --config-check\n\n"
            "Config: env/.env with VISION_API_KEY, VISION_API_BASE, VISION_MODEL,\n"
            "        VISION_API_FORMAT=openai_chat|openai_responses|anthropic"
        )
        sys.exit(0)
    if args[0] == "--config-check":
        cfg = load_config()
        print(json.dumps({k: (v if k != "api_key" else (v[:6] + "…")) for k, v in cfg.items()}, ensure_ascii=False, indent=2))
        return
    image_path = args[0]
    prompt = " ".join(args[1:]).strip() or "Describe this image in detail."
    print(analyze_image(image_path, prompt))


if __name__ == "__main__":
    main()
