from unittest.mock import MagicMock, patch, PropertyMock
from core.engine import Engine, AbortedError
from core.tool import Tool, ToolResult
from core.permissions import PermissionChecker


class DummyTool(Tool):
    name = "Dummy"
    description = "A dummy tool for testing"
    input_schema = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }

    def execute(self, msg: str) -> ToolResult:
        return ToolResult(content=f"got: {msg}")


def _make_text_stream(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text

    final_msg = MagicMock()
    final_msg.content = [block]

    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.text_stream = iter([text])
    stream.get_final_message = MagicMock(return_value=final_msg)
    return stream


def _make_engine():
    return Engine(
        tools=[DummyTool()],
        system_prompt="test",
        permission_checker=PermissionChecker(auto_approve=True),
    )


class _FakeEscListener:
    """A no-op replacement for EscListener that doesn't touch the terminal."""
    pressed = False

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def check_esc_nonblocking(self):
        return False


@patch("tui.query.EscListener", _FakeEscListener)
def test_run_query_prints_text(capsys):
    """run_query should print text events to stdout in print_mode."""
    from tui.query import run_query

    engine = _make_engine()
    with patch.object(engine._client, "stream_messages", return_value=_make_text_stream("hello world")):
        run_query(engine, "hi", print_mode=True)

    captured = capsys.readouterr()
    assert "hello world" in captured.out


@patch("tui.query.EscListener", _FakeEscListener)
def test_run_query_handles_tool_call_event():
    """run_query should display tool call info via rich console."""
    from tui.query import run_query

    engine = _make_engine()

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tu_1"
    tool_block.name = "Dummy"
    tool_block.input = {"msg": "test"}

    first_final = MagicMock()
    first_final.content = [tool_block]
    first_stream = MagicMock()
    first_stream.__enter__ = MagicMock(return_value=first_stream)
    first_stream.__exit__ = MagicMock(return_value=False)
    first_stream.text_stream = iter([])
    first_stream.get_final_message = MagicMock(return_value=first_final)

    second_stream = _make_text_stream("done")

    with patch.object(engine._client, "stream_messages", side_effect=[first_stream, second_stream]):
        run_query(engine, "use tool", print_mode=True)


@patch("tui.query.EscListener", _FakeEscListener)
def test_run_query_handles_keyboard_interrupt():
    """run_query should gracefully handle KeyboardInterrupt."""
    from tui.query import run_query

    engine = _make_engine()

    def raise_interrupt(*a, **kw):
        raise KeyboardInterrupt()

    with patch.object(engine._client, "stream_messages", side_effect=raise_interrupt):
        run_query(engine, "hi", print_mode=True)
    # Should not propagate the exception


def test_maybe_auto_compact_announces_and_compresses(monkeypatch):
    from io import StringIO
    from types import SimpleNamespace

    from rich.console import Console

    from tui.app import _maybe_auto_compact

    class DummyEngine:
        def __init__(self, messages):
            self._messages = messages
            self.system_prompt = "system"

        def get_messages(self):
            return self._messages

        def set_messages(self, messages):
            self._messages = messages

    class DummyCompactService:
        def __init__(self, new_messages):
            self.new_messages = new_messages
            self.called_with = None

        def compact(self, messages, system_prompt):
            self.called_with = (messages, system_prompt)
            return self.new_messages, "summary"

    original_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    new_messages = [{"role": "assistant", "content": "summary"}]
    engine = DummyEngine(original_messages)
    compact_service = DummyCompactService(new_messages)

    monkeypatch.setattr(
        "tui.app.should_reduce_sections",
        lambda model, last_input_tokens: False,
    )
    monkeypatch.setattr(
        "tui.app.should_compact",
        lambda messages, model, last_input_tokens: True,
    )
    token_counts = iter([1234, 456])
    monkeypatch.setattr("tui.app.estimate_tokens", lambda messages: next(token_counts))

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    cost_tracker = SimpleNamespace(last_input_tokens=1500)

    info = _maybe_auto_compact(engine, compact_service, cost_tracker, "test-model", console)

    rendered = output.getvalue()
    assert "Auto-compacting conversation before the next turn." in rendered
    assert "Context compressed: 2" in rendered
    assert "1 messages, ~1,234" in rendered
    assert "~456 tokens." in rendered
    assert compact_service.called_with == (original_messages, "system")
    assert engine.get_messages() == new_messages
    assert info == {
        "triggered": True,
        "before_messages": 2,
        "before_tokens": 1234,
        "status": "completed",
        "after_messages": 1,
        "after_tokens": 456,
    }


def test_maybe_auto_compact_reduces_sections_before_compacting(monkeypatch):
    from io import StringIO
    from types import SimpleNamespace

    from rich.console import Console

    from tui.app import _maybe_auto_compact

    class DummyEngine:
        def __init__(self, messages):
            self._messages = messages
            self.system_prompt = "system"

        def get_messages(self):
            return self._messages

        def set_messages(self, messages):
            self._messages = messages

    class DummyCompactService:
        def __init__(self, new_messages):
            self.new_messages = new_messages
            self.called_with = None

        def compact(self, messages, system_prompt):
            self.called_with = (messages, system_prompt)
            return self.new_messages, "summary"

    engine = DummyEngine([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ])
    compact_service = DummyCompactService([{"role": "assistant", "content": "summary"}])

    monkeypatch.setattr("tui.app.should_reduce_sections", lambda model, last_input_tokens: True)
    monkeypatch.setattr("tui.app.section_budget_target_chars", lambda model: 5000)
    compact_checks = iter([True, True])
    monkeypatch.setattr(
        "tui.app.should_compact",
        lambda messages, model, last_input_tokens: next(compact_checks),
    )
    token_counts = iter([1234, 456])
    monkeypatch.setattr("tui.app.estimate_tokens", lambda messages: next(token_counts))

    rebuild_calls = []

    def rebuild_prompt_with_budget(max_dynamic_chars):
        rebuild_calls.append(max_dynamic_chars)
        return {
            "prompt": "reduced prompt",
            "layout": {
                "reductions": [{"name": "git", "action": "drop", "saved_chars": 1000}],
                "stats": {"budget_saved_chars": 1000, "dynamic_chars": 4200},
            },
        }

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    cost_tracker = SimpleNamespace(last_input_tokens=1500)

    info = _maybe_auto_compact(
        engine,
        compact_service,
        cost_tracker,
        "test-model",
        console,
        rebuild_prompt_with_budget=rebuild_prompt_with_budget,
    )

    rendered = output.getvalue()
    assert rebuild_calls == [5000]
    assert "Reduced prompt sections before compact" in rendered
    assert info["reduction"]["status"] == "reduced"
    assert info["reduction"]["reductions"][0]["name"] == "git"
    assert compact_service.called_with == (engine.get_messages(), "system") or compact_service.called_with == ([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ], "system")


def test_maybe_auto_compact_can_stop_after_section_reduction(monkeypatch):
    from io import StringIO
    from types import SimpleNamespace

    from rich.console import Console

    from tui.app import _maybe_auto_compact

    class DummyEngine:
        def __init__(self, messages):
            self._messages = messages
            self.system_prompt = "system"

        def get_messages(self):
            return self._messages

        def set_messages(self, messages):
            self._messages = messages

    class DummyCompactService:
        def __init__(self):
            self.called_with = None

        def compact(self, messages, system_prompt):
            self.called_with = (messages, system_prompt)
            return messages, "summary"

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    engine = DummyEngine(messages)
    compact_service = DummyCompactService()

    monkeypatch.setattr("tui.app.should_reduce_sections", lambda model, last_input_tokens: True)
    monkeypatch.setattr("tui.app.section_budget_target_chars", lambda model: 5000)
    compact_checks = iter([False])
    monkeypatch.setattr(
        "tui.app.should_compact",
        lambda messages, model, last_input_tokens: next(compact_checks),
    )

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    cost_tracker = SimpleNamespace(last_input_tokens=1500)

    info = _maybe_auto_compact(
        engine,
        compact_service,
        cost_tracker,
        "test-model",
        console,
        rebuild_prompt_with_budget=lambda max_dynamic_chars: {
            "prompt": "reduced prompt",
            "layout": {
                "reductions": [{"name": "git", "action": "drop", "saved_chars": 1000}],
                "stats": {"budget_saved_chars": 1000, "dynamic_chars": 4200},
            },
        },
    )

    rendered = output.getvalue()
    assert "Reduced prompt sections before compact" in rendered
    assert "Auto-compacting conversation before the next turn." not in rendered
    assert compact_service.called_with is None
    assert info == {
        "triggered": False,
        "status": "reduced",
        "saved_chars": 1000,
        "dynamic_chars": 4200,
        "reductions": [{"name": "git", "action": "drop", "saved_chars": 1000}],
    }


@patch("tui.query.EscListener", _FakeEscListener)
def test_run_query_writes_turn_artifact():
    from tui.query import run_query

    engine = _make_engine()
    engine.write_turn_artifact = MagicMock(return_value="turns/0001.json")

    with patch.object(engine._client, "stream_messages", return_value=_make_text_stream("hello world")):
        artifact = run_query(
            engine,
            "hi",
            print_mode=True,
            turn_context={"source": "test"},
        )

    assert artifact is not None
    assert artifact["status"] == "completed"
    assert artifact["context"] == {"source": "test"}
    assert artifact["input"]["preview"] == "hi"
    assert artifact["assistant"]["preview"] == "hello world"
    assert artifact["artifact_path"] == "turns/0001.json"
    assert artifact["timeline"][0]["type"] == "turn_started"
    assert any(item["type"] == "text_started" for item in artifact["timeline"])
    engine.write_turn_artifact.assert_called_once()
