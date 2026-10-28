"""Phase 1: shared per-result outcome taxonomy for executed tool calls.

``classify_tool_result`` distinguishes the five outcome categories the agent
reliability phase requires (success, failure, inconclusive, blocked, and
valid-but-unexpected), staying consistent with the existing binary
``tool_result_is_successful`` gate used across the agent loop.
"""
import pytest

from src.tool_capabilities import (
    ToolOutcome,
    classify_tool_result,
    tool_result_is_successful,
)


class TestSuccessfulExecution:
    def test_exit_zero_with_output_is_success(self):
        result = {"exit_code": 0, "output": "created app.py"}
        assert classify_tool_result(result) is ToolOutcome.SUCCESS

    def test_exit_code_none_with_payload_is_success(self):
        # Native/MCP style results often omit exit_code entirely.
        result = {"output": "1 row"}
        assert classify_tool_result(result) is ToolOutcome.SUCCESS

    def test_success_true_flag_with_content_is_success(self):
        result = {"success": True, "output": "OK"}
        assert classify_tool_result(result) is ToolOutcome.SUCCESS


class TestErroredExecution:
    def test_error_key_is_failed(self):
        result = {"error": "write_file: content must be a string", "exit_code": 1}
        assert classify_tool_result(result) is ToolOutcome.FAILED

    def test_nonzero_exit_code_is_failed(self):
        result = {"exit_code": 2, "output": "Traceback"}
        assert classify_tool_result(result) is ToolOutcome.FAILED

    def test_success_false_is_failed(self):
        result = {"success": False, "output": "provider refused"}
        assert classify_tool_result(result) is ToolOutcome.FAILED

    def test_non_dict_result_is_failed(self):
        # Dispatcher contract is a dict; anything else is a broken result.
        assert classify_tool_result(None) is ToolOutcome.FAILED
        assert classify_tool_result("ok") is ToolOutcome.FAILED
        assert classify_tool_result(0) is ToolOutcome.FAILED


class TestBlockedExecution:
    def test_policy_block_is_blocked_not_failed(self):
        # blocked_tool_result() shape (src.tool_capabilities.blocked_tool_result)
        result = {
            "error": "external untrusted context blocks write_workspace",
            "exit_code": 1,
            "blocked": True,
            "policy": "external_untrusted_context",
        }
        assert classify_tool_result(result) is ToolOutcome.BLOCKED

    def test_approval_required_is_blocked(self):
        result = {"approval_required": True}
        assert classify_tool_result(result) is ToolOutcome.BLOCKED


class TestInconclusiveExecution:
    def test_bare_exit_zero_proves_nothing(self):
        # {"exit_code": 0} with no observable payload cannot evidence a
        # completed task; it must not be treated the same as real success.
        assert classify_tool_result({"exit_code": 0}) is ToolOutcome.INCONCLUSIVE

    def test_status_only_dict_is_inconclusive(self):
        result = {"exit_code": 0, "success": True, "policy": "default"}
        assert classify_tool_result(result) is ToolOutcome.INCONCLUSIVE

    def test_empty_output_string_is_inconclusive(self):
        result = {"exit_code": 0, "output": ""}
        assert classify_tool_result(result) is ToolOutcome.INCONCLUSIVE

    def test_empty_payload_collections_are_inconclusive(self):
        result = {"exit_code": 0, "emails": [], "details": {}}
        assert classify_tool_result(result) is ToolOutcome.INCONCLUSIVE


class TestUnexpectedResult:
    def test_valid_output_failing_expectation_is_unexpected(self):
        result = {"exit_code": 0, "output": "config.ini"}
        expected = lambda r: str(r.get("output", "")).endswith(".json")
        assert classify_tool_result(result, expected=expected) is ToolOutcome.UNEXPECTED

    def test_matching_expectation_stays_success(self):
        result = {"exit_code": 0, "output": "answer.json"}
        expected = lambda r: str(r.get("output", "")).endswith(".json")
        assert classify_tool_result(result, expected=expected) is ToolOutcome.SUCCESS

    def test_expectation_never_masks_failure(self):
        result = {"exit_code": 1, "error": "boom"}
        assert classify_tool_result(result, expected=lambda r: True) is ToolOutcome.FAILED


class TestConsistencyWithExistingBinaryGate:
    """SUCCESS/INCONCLUSIVE must never disagree with the binary success gate.

    The agent loop branches on ``tool_result_is_successful`` in 80+ places;
    classify() refines, it must not contradict. FAIL/BLOCK/UNEXPECTED always
    map to not-successful.
    """

    @pytest.mark.parametrize("result", [
        {"exit_code": 0, "output": "done"},
        {"exit_code": 0},
        {"output": "no exit code"},
        {"exit_code": 1, "error": "nope"},
        {"blocked": True, "error": "policy", "exit_code": 1},
        {"approval_required": True},
        {"success": False, "output": "x"},
        {"exit_code": None, "output": ""},
        "not a dict",
        None,
    ])
    def test_classification_agrees_with_binary_gate(self, result):
        outcome = classify_tool_result(result)
        if outcome in (ToolOutcome.SUCCESS, ToolOutcome.INCONCLUSIVE):
            assert tool_result_is_successful(result)
        else:
            assert not tool_result_is_successful(result)

    @pytest.mark.parametrize("result", [
        {"exit_code": 0, "output": "done"},
        {"exit_code": 0},
    ])
    def test_inconclusive_refines_not_contradicts_success(self, result):
        # Both categories sit under the existing success umbrella; the
        # classifier's job is telling them apart for evidence purposes.
        assert tool_result_is_successful(result)
        outcomes = {classify_tool_result(result)}
        assert outcomes <= {ToolOutcome.SUCCESS, ToolOutcome.INCONCLUSIVE}
