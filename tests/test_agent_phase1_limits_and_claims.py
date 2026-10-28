"""Phase 1 reliability contracts: limits, readback evidence, and honest claims.

Pins the acceptance criteria for the agent-reliability phase against the real
machinery: per-turn tool budget enforcement, artifact creation confirmed by a
subsequent read, the completion gate refusing false success after a failed
tool, and settings compatibility for the execution limits.
"""
import ast
import json
from pathlib import Path

import pytest

from src.agent_evidence import CompletionRequirements, EvidenceLedger
from src.agent_runtime.completion import with_completion_gate
from src.agent_runtime.journal import (
    ActionJournal, bind_journal, mark_dispatch, record_action,
)
from src.tool_types import ToolBlock

from tests.runtime_evidence_helpers import server_authorized_executor


@pytest.fixture(autouse=True)
def standalone_dispatch_authority(monkeypatch):
    from src import tool_execution
    monkeypatch.setattr(tool_execution, "execute_tool_block",
                        server_authorized_executor(tool_execution.execute_tool_block))


_AGENT_LOOP = Path(__file__).resolve().parent.parent / "src" / "agent_loop.py"


# ── Case 5: per-turn tool call limit is real ─────────────────────────────

def test_per_turn_tool_budget_enforcement_exists_in_loop():
    """The loop must cap executed calls when agent_max_tool_calls > 0 and emit
    a budget_exceeded event; 0 stays unlimited by design."""
    source = _AGENT_LOOP.read_text(encoding="utf-8")
    assert "max_tool_calls > 0 and total_tool_calls >= max_tool_calls" in source, (
        "per-turn tool budget check disappeared from the agent loop"
    )
    assert '"budget_exceeded"' in source, (
        "budget_exceeded event emission disappeared from the agent loop"
    )


def test_execution_limit_settings_keep_documented_defaults():
    from src.settings import DEFAULT_SETTINGS
    # 0 = unlimited tool calls; missing key falls back to these defaults.
    assert DEFAULT_SETTINGS["agent_max_tool_calls"] == 0
    assert DEFAULT_SETTINGS["agent_max_rounds"] == 20


# ── Case 4: created artifact is confirmed by a subsequent read ──────────

@record_action
async def _backend(block):
    mark_dispatch()
    return block.tool_type, {"exit_code": 0, "output": "OK"}


@pytest.mark.asyncio
async def test_created_artifact_confirmed_by_subsequent_read(tmp_path, monkeypatch):
    from src.tool_execution import execute_tool_block, NO_TOOL_SECURITY_CONTEXT
    monkeypatch.setattr("src.tool_execution.owner_is_admin_or_single_user", lambda owner: True)
    journal = ActionJournal(workspace=str(tmp_path), observed_artifacts=("app.py",))
    with bind_journal(journal):
        _, write_result = await execute_tool_block(
            ToolBlock("write_file", json.dumps({"path": "app.py", "content": 'print("ok")'})),
            workspace=str(tmp_path), security_context=NO_TOOL_SECURITY_CONTEXT,
        )
        _, read_result = await execute_tool_block(
            ToolBlock("read_file", "app.py"),
            workspace=str(tmp_path), security_context=NO_TOOL_SECURITY_CONTEXT,
        )
    assert write_result.get("exit_code") == 0, write_result
    assert read_result.get("exit_code") == 0, read_result
    # The readback, not the exit code alone, proves what was written.
    assert 'print("ok")' in str(read_result.get("output", ""))
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == 'print("ok")'
    ledger = EvidenceLedger.from_tool_events(
        journal.evidence_events(),
        CompletionRequirements(required_artifacts=("app.py",), workspace_root=str(tmp_path)),
    )
    assert ledger.evaluate().can_complete


# ── Case 6: a failed tool cannot yield a false completion claim ─────────

@record_action
async def _failing_backend(block):
    mark_dispatch()
    return block.tool_type, {"exit_code": 1, "error": "command failed"}


def decode(chunks):
    return [json.loads(c[6:]) for c in chunks if c.strip() != "data: [DONE]"]


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", ["All tests passed.", "The tests passed.", "Done."])
async def test_failed_tool_cannot_support_success_claim(claim):
    @with_completion_gate
    async def stream(messages):
        await _failing_backend(ToolBlock("bash", "python -m unittest"))
        yield 'data: ' + json.dumps({"delta": claim}) + '\n\n'
        yield "data: [DONE]\n\n"
    events = decode([chunk async for chunk in stream([{"role": "user", "content": "Run the tests."}])])
    assert claim not in json.dumps(events)
    final = next(e for e in events if e.get("type") == "final_response")
    assert final["content"].startswith("The task is incomplete:")
    decision = next(e["data"] for e in events if e.get("type") == "completion_decision")
    assert not decision["can_complete"]


# ── Case 7: compatibility with current agent configuration ──────────────

def test_settings_defaults_stay_superset_compatible():
    """Phase 1 must not remove or rename existing settings keys."""
    from src.settings import DEFAULT_SETTINGS
    for required in ("agent_max_tool_calls", "agent_max_rounds", "agent_input_token_budget"):
        assert required in DEFAULT_SETTINGS
    assert isinstance(DEFAULT_SETTINGS["agent_max_rounds"], int)
    assert DEFAULT_SETTINGS["agent_max_rounds"] >= 1


def test_classification_imports_do_not_break_settings_module():
    import src.settings  # noqa: F401
    from src.tool_capabilities import ToolOutcome, classify_tool_result  # noqa: F401
