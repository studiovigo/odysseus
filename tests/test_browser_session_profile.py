"""Phase 2: browser session profile mode for the built-in Playwright MCP.

``browser_session_profile`` controls whether the agent browser keeps a
persistent, Odysseus-owned Chromium profile (logins survive restarts) or stays
isolated (throwaway profile, default). The persistent profile lives under the
app's data directory — never the user's personal Chrome profile — and cookie
values never enter model context.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_builtin_mcp(monkeypatch):
    core = types.ModuleType("core")
    core.__path__ = []
    platform_compat = types.ModuleType("core.platform_compat")
    platform_compat.IS_WINDOWS = False
    platform_compat.which_tool = lambda name: None
    monkeypatch.setitem(sys.modules, "core", core)
    monkeypatch.setitem(sys.modules, "core.platform_compat", platform_compat)

    spec = importlib.util.spec_from_file_location(
        "builtin_mcp_profile_under_test",
        ROOT / "src" / "builtin_mcp.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_BROWSER_EXECUTABLE", raising=False)
    monkeypatch.delenv("ODYSSEUS_BROWSER_ISOLATED", raising=False)
    monkeypatch.delenv("ODYSSEUS_BROWSER_NO_SANDBOX", raising=False)


def _set_mode(monkeypatch, mode):
    monkeypatch.setattr(
        "src.settings.load_settings",
        lambda: {"browser_session_profile": mode, "browser_executable": ""},
    )


_BASE_ARGS = ["-y", "@playwright/mcp@latest", "--headless"]


def test_default_mode_is_isolated(monkeypatch):
    _set_mode(monkeypatch, "isolated")
    builtin_mcp = _load_builtin_mcp(monkeypatch)
    args = builtin_mcp._browser_mcp_args(list(_BASE_ARGS))
    assert "--isolated" in args
    assert "--user-data-dir" not in args


def test_unknown_mode_falls_back_to_isolated(monkeypatch):
    _set_mode(monkeypatch, "yes please")
    builtin_mcp = _load_builtin_mcp(monkeypatch)
    args = builtin_mcp._browser_mcp_args(list(_BASE_ARGS))
    assert "--isolated" in args
    assert "--user-data-dir" not in args


def test_missing_setting_load_falls_back_to_isolated(monkeypatch):
    # settings module failing entirely (early boot, corrupt file) must not
    # change the safe default.
    def boom():
        raise RuntimeError("settings unavailable")
    monkeypatch.setattr("src.settings.load_settings", boom)
    builtin_mcp = _load_builtin_mcp(monkeypatch)
    args = builtin_mcp._browser_mcp_args(list(_BASE_ARGS))
    assert "--isolated" in args
    assert "--user-data-dir" not in args


def test_persistent_mode_pins_odysseus_owned_profile(monkeypatch, tmp_path):
    _set_mode(monkeypatch, "persistent")
    builtin_mcp = _load_builtin_mcp(monkeypatch)
    monkeypatch.setattr(builtin_mcp, "get_app_root", lambda: str(tmp_path))
    args = builtin_mcp._browser_mcp_args(list(_BASE_ARGS))
    assert "--isolated" not in args
    idx = args.index("--user-data-dir")
    profile_dir = args[idx + 1]
    assert profile_dir == str(tmp_path / "data" / "browser-profile")
    # Directory is created before the server start so Chromium owns it.
    assert Path(profile_dir).is_dir()


def test_persistent_profile_dir_is_app_scoped_not_user_home(monkeypatch, tmp_path):
    _set_mode(monkeypatch, "persistent")
    builtin_mcp = _load_builtin_mcp(monkeypatch)
    monkeypatch.setattr(builtin_mcp, "get_app_root", lambda: str(tmp_path))
    assert "user-home" not in builtin_mcp._persistent_profile_dir()


def test_explicit_user_data_dir_wins_over_setting(monkeypatch):
    _set_mode(monkeypatch, "persistent")
    builtin_mcp = _load_builtin_mcp(monkeypatch)
    args = builtin_mcp._browser_mcp_args(
        list(_BASE_ARGS) + ["--user-data-dir", "/tmp/custom-profile"]
    )
    assert "--user-data-dir" in args
    assert "/tmp/custom-profile" in args
    # The explicit path appears exactly once (setting did not append another).
    assert args.count("--user-data-dir") == 1
    assert "--isolated" not in args


def test_legacy_env_opt_out_keeps_default_profile(monkeypatch):
    _set_mode(monkeypatch, "isolated")
    monkeypatch.setenv("ODYSSEUS_BROWSER_ISOLATED", "0")
    builtin_mcp = _load_builtin_mcp(monkeypatch)
    args = builtin_mcp._browser_mcp_args(list(_BASE_ARGS))
    assert "--isolated" not in args
    assert "--user-data-dir" not in args


def test_persistent_mode_makedirs_failure_falls_back_to_isolated(monkeypatch, tmp_path):
    _set_mode(monkeypatch, "persistent")
    builtin_mcp = _load_builtin_mcp(monkeypatch)
    monkeypatch.setattr(builtin_mcp, "get_app_root", lambda: str(tmp_path))

    def forbidden(path, exist_ok=False):
        raise PermissionError("read-only volume")
    monkeypatch.setattr(builtin_mcp.os, "makedirs", forbidden)
    args = builtin_mcp._browser_mcp_args(list(_BASE_ARGS))
    assert "--user-data-dir" not in args
    assert "--isolated" in args


def test_setting_default_value_in_settings_module():
    from src.settings import DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS["browser_session_profile"] == "isolated"
