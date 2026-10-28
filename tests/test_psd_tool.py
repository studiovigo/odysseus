"""Phase 3: read-only PSD interop (layer inspection + PNG export).

The psd_tool never mutates the source PSD, confines reads and writes to the
active workspace, and reports unsupported selections instead of flattening
silently. Real-parse tests require the optional psd-tools package.
"""
import hashlib
import json
from pathlib import Path

import pytest

from src.agent_tools.psd_tool import PsdTool

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "psd" / "sample_layers.psd"


@pytest.fixture(autouse=True)
def active_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr("src.tool_execution.get_active_workspace", lambda: str(tmp_path))
    return tmp_path


async def call(content: dict) -> dict:
    return await PsdTool().execute(json.dumps(content), {})


def _fixture_in_workspace(tmp_path) -> Path:
    target = tmp_path / "design.psd"
    target.write_bytes(FIXTURE.read_bytes())
    return target


# ── Contract tests: run without psd-tools installed ─────────────────────

def test_psd_tool_schema_advertised():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    names = {
        (schema.get("function") or {}).get("name") for schema in FUNCTION_TOOL_SCHEMAS
    }
    assert "psd_tool" in names


def test_psd_tool_in_shell_files_family():
    from src.turn_contract import FAMILY_TOOLS
    assert "psd_tool" in FAMILY_TOOLS["shell_files"]


def test_psd_tool_capabilities_read_by_default():
    from src.tool_capabilities import (
        ResultIntegrity, ToolEffect, capabilities_for_action, capabilities_for_tool,
    )
    base = capabilities_for_tool("psd_tool")
    assert base.known
    assert ToolEffect.READ_WORKSPACE in base.effects
    assert base.result_integrity is ResultIntegrity.WORKSPACE_UNTRUSTED
    read = capabilities_for_action("psd_tool", '{"path":"/workspace/a.psd","action":"info"}')
    assert ToolEffect.WRITE_WORKSPACE not in read.effects
    export = capabilities_for_action(
        "psd_tool",
        '{"path":"/workspace/a.psd","action":"export","output_path":"/workspace/a.png"}',
    )
    assert ToolEffect.WRITE_WORKSPACE in export.effects


async def test_unknown_arguments_rejected():
    result = await call({"path": "/workspace/a.psd", "bogus": 1})
    assert result["exit_code"] == 1
    assert "unknown argument" in result["error"]


async def test_invalid_action_rejected(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    result = await call({"path": str(source), "action": "flatten"})
    assert result["exit_code"] == 1
    assert "action must be info or export" in result["error"]


async def test_info_rejects_output_path(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    result = await call({"path": str(source), "action": "info", "output_path": "/workspace/x.png"})
    assert result["exit_code"] == 1
    assert "read-only" in result["error"]


async def test_export_requires_output_path(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    result = await call({"path": str(source), "action": "export"})
    assert result["exit_code"] == 1
    assert "requires output_path" in result["error"]


async def test_missing_psd_requires_workspace(tmp_path):
    result = await call({"path": "ghost.psd"})
    assert result["exit_code"] == 1
    assert "not found" in result["error"]


async def test_non_psd_suffix_rejected(tmp_path):
    (tmp_path / "notes.txt").write_text("not a psd", encoding="utf-8")
    result = await call({"path": "notes.txt"})
    assert result["exit_code"] == 1
    assert ".psd and .psb" in result["error"]


# ── Real PSD parsing: requires optional psd-tools ──────────────────────

psd_tools = pytest.importorskip("psd_tools", reason="psd-tool tests need psd-tools")


async def test_info_lists_layer_tree(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    result = await call({"path": str(source), "action": "info"})
    assert result["exit_code"] == 0
    assert result["width"] == 64 and result["height"] == 48
    assert result["layer_count"] == 2
    assert result["truncated"] is False
    names = [layer["name"] for layer in result["layers"]]
    assert names == ["Background Fill", "Overlay"]
    assert all(layer["kind"] == "pixel" for layer in result["layers"])


async def test_export_composite_preserves_original(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = await call({
        "path": str(source), "action": "export", "output_path": "/workspace/composite.png",
    })
    assert result["exit_code"] == 0, result
    exported = tmp_path / "composite.png"
    assert exported.is_file()
    assert result["exported_dimensions"] == [64, 48]
    assert result["images"] and result["images"][0]["mimeType"] == "image/jpeg"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


async def test_export_named_layer(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    result = await call({
        "path": str(source), "action": "export", "layer": "overlay",
        "output_path": "/workspace/layer.png",
    })
    assert result["exit_code"] == 0, result
    assert (tmp_path / "layer.png").is_file()
    assert "Overlay" in result["output"]


async def test_export_unknown_layer_reports_instead_of_flattening(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    result = await call({
        "path": str(source), "action": "export", "layer": "No Such Layer",
        "output_path": "/workspace/x.png",
    })
    assert result["exit_code"] == 1
    assert "found no layer" in result["error"]


async def test_export_cannot_overwrite_source_psd(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = await call({
        "path": str(source), "action": "export", "output_path": str(source),
    })
    assert result["exit_code"] == 1
    assert "must differ from the source PSD" in result["error"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


async def test_export_output_confined_to_workspace(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    result = await call({
        "path": str(source), "action": "export",
        "output_path": "../outside.png",
    })
    assert result["exit_code"] == 1
    assert "stay inside the active workspace" in result["error"]
    assert not (tmp_path.parent / "outside.png").exists()


async def test_export_output_must_be_png(tmp_path):
    source = _fixture_in_workspace(tmp_path)
    result = await call({
        "path": str(source), "action": "export", "output_path": "/workspace/out.jpg",
    })
    assert result["exit_code"] == 1
    assert "PNG" in result["error"]
