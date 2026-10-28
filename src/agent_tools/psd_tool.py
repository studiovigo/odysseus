"""Read-only Photoshop PSD interop for the Odysseus agent.

Phase 3 creative-work boundary: the agent can inspect the layer tree of a
workspace PSD and export the composite or a single layer as a PNG inside the
workspace. The original PSD is never mutated; unsupported PSD features are
reported instead of silently flattened, matching the layered-interchange
policy in plans/photo-editor-professional-roadmap.md.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from .media_tools import _resolve_workspace_path

_PSD_SUFFIXES = {".psd", ".psb"}
_PSD_TOOL_ARGUMENTS = {"path", "action", "layer", "output_path"}
_MAX_LISTED_LAYERS = 200
_PREVIEW_LIMIT = 512


def _is_group(layer) -> bool:
    """psd-tools exposes is_group as a method in some versions, a property in others."""
    marker = getattr(layer, "is_group", None)
    if callable(marker):
        try:
            return bool(marker())
        except Exception:
            return False
    return bool(marker)


def _layer_rows(psd) -> tuple[list[dict], bool]:
    """Flatten the layer tree bottom-up, bounded, with group depth."""
    rows: list[dict] = []

    def walk(layers, depth):
        for layer in layers:
            if len(rows) >= _MAX_LISTED_LAYERS:
                return
            bbox = getattr(layer, "bbox", None)
            try:
                bbox_values = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox else None
            except (TypeError, ValueError):
                bbox_values = None
            rows.append({
                "name": getattr(layer, "name", "") or "",
                "kind": getattr(layer, "kind", "") or "",
                "visible": bool(getattr(layer, "visible", True)),
                "opacity": int(getattr(layer, "opacity", 255) or 0),
                "depth": depth,
                "bbox": bbox_values,
            })
            if _is_group(layer):
                walk(layer, depth + 1)

    walk(psd, 0)
    truncated = False
    total = 0

    def count(layers):
        nonlocal total
        for layer in layers:
            total += 1
            if _is_group(layer):
                count(layer)

    count(psd)
    truncated = total > len(rows)
    return rows, truncated


def _flattened_layers(psd) -> list:
    flattened: list = []

    def walk(layers):
        for layer in layers:
            flattened.append(layer)
            if _is_group(layer):
                walk(layer)

    walk(psd)
    return flattened


def _find_layer(psd, selector: str):
    """Match a layer by exact case-insensitive name or 1-based listed index."""
    raw = str(selector or "").strip()
    if not raw:
        return None
    flattened = _flattened_layers(psd)
    lowered = raw.casefold()
    for layer in flattened:
        if str(getattr(layer, "name", "") or "").casefold() == lowered:
            return layer
    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(flattened):
            return flattened[index - 1]
    return None


class PsdTool:
    """Inspect PSD layer structure; export composite or layer pixels as PNG."""

    async def execute(self, content: str, _ctx: dict) -> dict:
        try:
            args = json.loads(content or "{}")
        except json.JSONDecodeError as exc:
            return {"error": f"psd_tool arguments must be JSON: {exc}", "exit_code": 1}
        if not isinstance(args, dict):
            return {"error": "psd_tool arguments must be a JSON object", "exit_code": 1}
        unknown = sorted(set(args) - _PSD_TOOL_ARGUMENTS)
        if unknown:
            return {"error": "psd_tool unknown argument(s): " + ", ".join(unknown), "exit_code": 1}
        action = str(args.get("action") or "info").strip().casefold()
        if action not in {"info", "export"}:
            return {"error": "psd_tool action must be info or export", "exit_code": 1}
        if action == "info" and args.get("output_path") not in (None, ""):
            return {"error": "psd_tool info is read-only; output_path is only valid with action=export", "exit_code": 1}
        if action == "export" and args.get("output_path") in (None, ""):
            return {"error": "psd_tool export requires output_path", "exit_code": 1}
        try:
            path = _resolve_workspace_path(
                str(args.get("path") or ""),
                must_exist=True,
                tool_name="psd_tool",
            )
        except (ValueError, FileNotFoundError) as exc:
            return {"error": str(exc), "exit_code": 1}
        if path.suffix.lower() not in _PSD_SUFFIXES:
            return {"error": "psd_tool supports .psd and .psb files, not " + (path.suffix or "this file type"), "exit_code": 1}

        try:
            from psd_tools import PSDImage
        except ImportError:
            return {
                "error": "psd_tool requires the optional psd-tools package; install it with: pip install psd-tools",
                "exit_code": 1,
            }

        try:
            psd = await _open_psd(PSDImage, path)
        except Exception as exc:
            return {"error": f"psd_tool could not open the PSD: {exc}", "exit_code": 1}

        if action == "info":
            rows, truncated = _layer_rows(psd)
            mode = getattr(psd.color_mode, "name", None) or str(getattr(psd, "color_mode", ""))
            width, height = psd.size
            summary = (
                f"PSD {args.get('path')}: {width}x{height}, color mode {mode}, "
                f"{len(rows)} layer{'s' if len(rows) != 1 else ''} (bottom to top)."
            )
            if truncated:
                summary += f" Layer list truncated at {_MAX_LISTED_LAYERS}; refine per group if needed."
            listing = "\n".join(
                f"{'  ' * row['depth']}- {row['name']} [{row['kind']}]"
                + ("" if row["visible"] else " (hidden)")
                + (f" bbox={row['bbox']}" if row.get("bbox") else "")
                for row in rows
            )
            return {
                "output": summary + ("\n" + listing if listing else "\n(no layers)"),
                "width": width,
                "height": height,
                "color_mode": mode,
                "layer_count": len(rows),
                "layers": rows,
                "truncated": truncated,
                "exit_code": 0,
            }

        # action == "export": composite or one layer, original PSD untouched.
        try:
            output = _resolve_workspace_path(
                str(args.get("output_path") or ""),
                must_exist=False,
                field_name="output_path",
                tool_name="psd_tool",
            )
        except ValueError as exc:
            return {"error": str(exc), "exit_code": 1}
        if output.resolve() == path.resolve():
            return {"error": "psd_tool output_path must differ from the source PSD; the original file is preserved", "exit_code": 1}
        if output.suffix.lower() != ".png":
            return {"error": "psd_tool export output_path must be a PNG file", "exit_code": 1}
        selector = str(args.get("layer") or "").strip()
        try:
            if selector:
                layer = _find_layer(psd, selector)
                if layer is None:
                    return {
                        "error": f"psd_tool found no layer named or indexed '{selector}'; use action=info to list layers",
                        "exit_code": 1,
                    }
                image = layer.composite()
                target_label = f"layer '{layer.name}'"
            else:
                image = psd.composite()
                target_label = "visible-layer composite"
            if image is None:
                return {"error": f"psd_tool could not render {target_label}; the layer may be empty or unsupported", "exit_code": 1}
            image = image.convert("RGB")
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output, "PNG")
        except Exception as exc:
            return {"error": f"psd_tool export failed: {exc}", "exit_code": 1}

        preview = image.copy()
        preview.thumbnail((_PREVIEW_LIMIT, _PREVIEW_LIMIT))
        buffer = io.BytesIO()
        preview.save(buffer, "JPEG", quality=90)
        return {
            "output": (
                f"Exported {target_label} from {args.get('path')} to {args.get('output_path')} "
                f"({image.size[0]}x{image.size[1]}). The original PSD was not modified. One preview follows."
            ),
            "images": [{
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "mimeType": "image/jpeg",
            }],
            "output_path": str(args.get("output_path") or ""),
            "exported_dimensions": list(image.size),
            "exit_code": 0,
        }


async def _open_psd(PSDImage, path: Path):
    """Open off the event loop; PSD parsing is synchronous file work."""
    import asyncio

    return await asyncio.to_thread(PSDImage.open, str(path))
