import json

from core.session import SessionStore, _sanitize_cwd


def test_session_store_persists_mode(tmp_path, monkeypatch):
    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)

    store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="session-1",
        mode="coordinator",
    )
    store.append_message({"role": "user", "content": "hello"})

    meta_path = tmp_path / _sanitize_cwd("/tmp/project") / "session-1.meta.json"
    data = json.loads(meta_path.read_text())

    assert data["mode"] == "coordinator"

    sessions = SessionStore.list_sessions("/tmp/project")
    assert len(sessions) == 1
    assert sessions[0].mode == "coordinator"


def test_recent_conversation_messages_keeps_latest_rounds():
    from commands import _recent_conversation_messages

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "tool", "content": "ignored"},
        {"role": "user", "content": " second\n\nquestion "},
        {"role": "assistant", "content": [{"type": "text", "text": "second answer"}]},
        {"role": "user", "content": "third question"},
        {"role": "assistant", "content": "third answer"},
    ]

    assert _recent_conversation_messages(messages, rounds=2) == [
        ("user", "second question"),
        ("assistant", "second answer"),
        ("user", "third question"),
        ("assistant", "third answer"),
    ]


def test_resume_prints_latest_two_rounds(tmp_path, monkeypatch):
    from io import StringIO
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from rich.console import Console

    from commands import CommandContext, _cmd_resume

    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr("commands.os.getcwd", lambda: "/tmp/project")

    current_store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="current-session",
    )
    current_store.append_message({"role": "user", "content": "current"})

    resumed_store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="resumed-session",
    )
    resumed_store.append_message({"role": "user", "content": "old question"})
    resumed_store.append_message({"role": "assistant", "content": "old answer"})
    resumed_store.append_message({"role": "user", "content": "second question"})
    resumed_store.append_message({"role": "assistant", "content": "second answer"})
    resumed_store.append_message({"role": "user", "content": "third question"})
    resumed_store.append_message({"role": "assistant", "content": "third answer"})

    engine = MagicMock()
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    ctx = CommandContext(
        engine=engine,
        session_store=current_store,
        compact_service=MagicMock(),
        console=console,
        app_config=SimpleNamespace(model="test-model"),
        new_session_store=lambda: None,
        reconfigure_mode=lambda mode: None,
    )

    _cmd_resume(ctx, "resumed-session")

    rendered = output.getvalue()
    assert "Resumed session" in rendered
    assert "Recent conversation from resumed session" in rendered
    assert "2 rounds" in rendered
    preview_text = rendered.split("Recent conversation from resumed session", 1)[1]
    assert "old question" not in preview_text
    assert "second question" in preview_text
    assert "third answer" in preview_text
    engine.set_messages.assert_called_once()
    engine.set_session_store.assert_called_once()
    assert ctx.session_store is not None
    assert ctx.session_store.session_id == "resumed-session"


def test_session_store_writes_turn_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)

    store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="session-1",
    )

    path = store.append_turn_artifact({"status": "completed", "input": {"preview": "hi"}})

    assert path.name == "0001.json"
    assert path.parent == tmp_path / _sanitize_cwd("/tmp/project") / "session-1.turns"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["session_id"] == "session-1"
    assert payload["turn_index"] == 1
    assert payload["input"]["preview"] == "hi"


def test_session_store_persists_working_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)

    store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="session-1",
    )

    path = store.save_working_memory({"summary": "latest context", "recent_messages": []})

    assert path.name == "session-1.working-memory.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"] == "latest context"
    assert store.load_working_memory() == payload
