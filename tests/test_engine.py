from unittest.mock import MagicMock, patch
import pytest
from core.engine import Engine
from core.config import default_max_tokens_for_model
from core.tool import Tool, ToolResult
from core.permissions import PermissionChecker


class EchoTool(Tool):
    name = "Echo"
    description = "Returns the input message"
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def execute(self, message: str) -> ToolResult:
        return ToolResult(content=f"Echo: {message}")


def _make_engine(auto_approve=True):
    return Engine(
        tools=[EchoTool()],
        system_prompt="You are a test assistant.",
        permission_checker=PermissionChecker(auto_approve=auto_approve),
    )


def _make_text_response(text: str):
    """Simulate an API response with just text (no tool calls)."""
    from core.llm import LLMMessage, LLMUsage

    final_msg = LLMMessage(
        content=[{"type": "text", "text": text}],
        usage=LLMUsage(),
    )

    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.text_stream = iter([text])
    stream.get_final_message = MagicMock(return_value=final_msg)
    return stream


def _make_tool_then_text_response(tool_name, tool_input, tool_use_id, text):
    """Simulate: first response has tool_use, second response has text."""
    from core.llm import LLMMessage, LLMUsage

    first_final = LLMMessage(
        content=[{
            "type": "tool_use",
            "id": tool_use_id,
            "name": tool_name,
            "input": tool_input,
        }],
        usage=LLMUsage(),
    )
    first_stream = MagicMock()
    first_stream.__enter__ = MagicMock(return_value=first_stream)
    first_stream.__exit__ = MagicMock(return_value=False)
    first_stream.text_stream = iter([])
    first_stream.get_final_message = MagicMock(return_value=first_final)

    second_stream = _make_text_response(text)
    return [first_stream, second_stream]


def test_engine_returns_text_events():
    engine = _make_engine()
    with patch.object(engine._client, "stream_messages", return_value=_make_text_response("hello")):
        events = list(engine.submit("hi"))
    text_events = [e for e in events if e[0] == "text"]
    assert any("hello" in e[1] for e in text_events)


def test_engine_executes_tool_and_loops():
    engine = _make_engine()
    streams = _make_tool_then_text_response("Echo", {"message": "world"}, "tu_1", "done")

    with patch.object(engine._client, "stream_messages", side_effect=streams):
        events = list(engine.submit("use the echo tool"))

    tool_result_events = [e for e in events if e[0] == "tool_result"]
    assert len(tool_result_events) == 1
    _, tool_name, _, result = tool_result_events[0]
    assert tool_name == "Echo"
    assert "Echo: world" in result.content


def test_engine_denied_tool_returns_error_result():
    engine = _make_engine(auto_approve=False)
    streams = _make_tool_then_text_response("Echo", {"message": "hi"}, "tu_2", "ok")

    with patch.object(engine._permissions, "_prompt_user", return_value="deny"):
        with patch.object(engine._client, "stream_messages", side_effect=streams):
            events = list(engine.submit("echo hi"))

    tool_result_events = [e for e in events if e[0] == "tool_result"]
    assert tool_result_events[0][3].is_error


def test_engine_unknown_tool_returns_error():
    engine = _make_engine()
    streams = _make_tool_then_text_response("UnknownTool", {}, "tu_3", "done")

    with patch.object(engine._client, "stream_messages", side_effect=streams):
        events = list(engine.submit("use unknown"))

    tool_result_events = [e for e in events if e[0] == "tool_result"]
    assert tool_result_events[0][3].is_error
    assert "Unknown tool" in tool_result_events[0][3].content


def test_engine_uses_model_specific_default_max_tokens():
    engine = Engine(
        tools=[EchoTool()],
        system_prompt="You are a test assistant.",
        permission_checker=PermissionChecker(auto_approve=True),
        model="claude-sonnet-4",
    )

    with patch.object(engine._client, "stream_messages", return_value=_make_text_response("hello")) as stream:
        list(engine.submit("hi"))

    assert stream.call_args.kwargs["model"] == "claude-sonnet-4"
    assert stream.call_args.kwargs["max_tokens"] == default_max_tokens_for_model("claude-sonnet-4")


def test_engine_normalizes_assistant_tool_use_blocks_before_retrying():
    engine = _make_engine()
    streams = _make_tool_then_text_response("Echo", {"message": "world"}, "tu_1", "done")

    with patch.object(engine._client, "stream_messages", side_effect=streams) as stream:
        list(engine.submit("use the echo tool"))

    second_messages = stream.call_args_list[1].kwargs["messages"]
    assistant_message = second_messages[1]
    assistant_block = assistant_message["content"][0]

    assert isinstance(assistant_block, dict)
    assert assistant_block == {
        "type": "tool_use",
        "id": "tu_1",
        "name": "Echo",
        "input": {"message": "world"},
    }


def test_engine_normalizes_tool_result_blocks_before_follow_up_request():
    engine = _make_engine()
    streams = _make_tool_then_text_response("Echo", {"message": "world"}, "tu_1", "done")

    with patch.object(engine._client, "stream_messages", side_effect=streams) as stream:
        list(engine.submit("use the echo tool"))

    second_messages = stream.call_args_list[1].kwargs["messages"]
    tool_result_message = second_messages[2]

    assert tool_result_message["content"] == [{
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "Echo: world",
        "is_error": False,
    }]


# ---------------------------------------------------------------------------
# 401 / 402 / 429 error handling
# ---------------------------------------------------------------------------

def test_engine_handles_auth_error_401():
    """401 authentication error stops the conversation with an error event."""
    engine = _make_engine()
    exc = Exception("Invalid API key")

    with patch.object(engine._client, "stream_messages", side_effect=exc), \
         patch.object(engine._client, "is_authentication_error", return_value=True):
        events = list(engine.submit("hi"))

    error_events = [e for e in events if e[0] == "error"]
    assert any("Authentication" in e[1] for e in error_events)


def test_engine_handles_payment_required_402():
    """402 payment-required is a non-retryable API error."""
    engine = _make_engine()
    exc = Exception("Insufficient credits")

    with patch.object(engine._client, "stream_messages", side_effect=exc), \
         patch.object(engine._client, "is_authentication_error", return_value=False), \
         patch.object(engine._client, "is_retryable_error", return_value=False), \
         patch.object(engine._client, "is_api_error", return_value=True):
        events = list(engine.submit("hi"))

    error_events = [e for e in events if e[0] == "error"]
    assert any("API error" in e[1] for e in error_events)


def test_engine_handles_rate_limit_429():
    """429 rate-limit triggers retry then succeeds."""
    engine = _make_engine()
    exc = Exception("Rate limited")
    text_stream = _make_text_response("recovered")

    with patch.object(engine._client, "stream_messages", side_effect=[exc, text_stream]), \
         patch.object(engine._client, "is_authentication_error", return_value=False), \
         patch.object(engine._client, "is_retryable_error", side_effect=lambda e: e is exc), \
         patch.object(engine._client, "is_api_error", return_value=False), \
         patch("core.engine.time.sleep"):
        events = list(engine.submit("hi"))

    text_events = [e for e in events if e[0] == "text"]
    error_events = [e for e in events if e[0] == "error"]
    assert any("recovered" in e[1] for e in text_events)
    assert any("retrying" in e[1].lower() for e in error_events)


# ---------------------------------------------------------------------------
# finish_reason validation
# ---------------------------------------------------------------------------

def test_engine_warns_on_max_tokens_finish_reason():
    """Engine emits an error event when stop_reason is max_tokens."""
    engine = _make_engine()
    from core.llm import LLMMessage, LLMUsage

    final_msg = LLMMessage(
        content=[{"type": "text", "text": "truncated"}],
        usage=LLMUsage(),
        stop_reason="max_tokens",
    )
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.text_stream = iter(["truncated"])
    stream.get_final_message = MagicMock(return_value=final_msg)

    with patch.object(engine._client, "stream_messages", return_value=stream):
        events = list(engine.submit("hi"))

    error_events = [e for e in events if e[0] == "error"]
    assert any("max_tokens" in e[1].lower() or "truncated" in e[1].lower()
               for e in error_events)


def test_engine_no_warning_on_end_turn_finish_reason():
    """No error event when stop_reason is a normal end_turn."""
    engine = _make_engine()
    from core.llm import LLMMessage, LLMUsage

    final_msg = LLMMessage(
        content=[{"type": "text", "text": "done"}],
        usage=LLMUsage(),
        stop_reason="end_turn",
    )
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.text_stream = iter(["done"])
    stream.get_final_message = MagicMock(return_value=final_msg)

    with patch.object(engine._client, "stream_messages", return_value=stream):
        events = list(engine.submit("hi"))

    error_events = [e for e in events if e[0] == "error"]
    assert len(error_events) == 0


# ---------------------------------------------------------------------------
# Mid-stream error
# ---------------------------------------------------------------------------

def test_engine_handles_mid_stream_error():
    """A retryable error mid-stream triggers retry and recovers."""
    engine = _make_engine()

    def failing_text_stream():
        yield "partial"
        raise Exception("Connection reset mid-stream")

    stream_fail = MagicMock()
    stream_fail.__enter__ = MagicMock(return_value=stream_fail)
    stream_fail.__exit__ = MagicMock(return_value=False)
    stream_fail.text_stream = failing_text_stream()

    stream_ok = _make_text_response("success")

    with patch.object(engine._client, "stream_messages", side_effect=[stream_fail, stream_ok]), \
         patch.object(engine._client, "is_authentication_error", return_value=False), \
         patch.object(engine._client, "is_retryable_error",
                      side_effect=lambda e: "Connection reset" in str(e)), \
         patch.object(engine._client, "is_api_error", return_value=False), \
         patch("core.engine.time.sleep"):
        events = list(engine.submit("hi"))

    text_events = [e for e in events if e[0] == "text"]
    assert any("success" in e[1] for e in text_events)


class EchoLongTool(Tool):
    name = "EchoLong"
    description = "Returns a long payload"
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def execute(self, message: str) -> ToolResult:
        return ToolResult(content=message)


class CountingEchoTool(Tool):
    name = "Echo"
    description = "Returns the input message and counts calls"
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def __init__(self):
        self.calls = 0

    def execute(self, message: str) -> ToolResult:
        self.calls += 1
        return ToolResult(content=f"Echo: {message}")


class NestedSchemaTool(Tool):
    name = "Nested"
    description = "Validates nested schema"
    input_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["label", "description"],
                            },
                            "minItems": 2,
                            "maxItems": 4,
                        },
                    },
                    "required": ["question", "options"],
                },
                "minItems": 1,
                "maxItems": 4,
            }
        },
        "required": ["questions"],
    }

    def execute(self, questions: list) -> ToolResult:
        return ToolResult(content=f"questions={len(questions)}")


def _make_engine_with_tools(*tools):
    return Engine(
        tools=list(tools),
        system_prompt="You are a test assistant.",
        permission_checker=PermissionChecker(auto_approve=True),
    )


def _make_tool_use_message(tool_uses):
    from core.llm import LLMMessage, LLMUsage

    final_msg = LLMMessage(content=tool_uses, usage=LLMUsage())
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.text_stream = iter([])
    stream.get_final_message = MagicMock(return_value=final_msg)
    return stream


def test_engine_rejects_missing_required_tool_input():
    engine = _make_engine()
    streams = _make_tool_then_text_response("Echo", {}, "tu_missing", "done")

    with patch.object(engine._client, "stream_messages", side_effect=streams):
        events = list(engine.submit("use echo"))

    tool_result = [e for e in events if e[0] == "tool_result"][0][3]
    assert tool_result.is_error
    assert "Invalid input for Echo" in tool_result.content
    assert "missing required field 'message'" in tool_result.content


def test_engine_rejects_unexpected_tool_input_field():
    engine = _make_engine()
    streams = _make_tool_then_text_response(
        "Echo",
        {"message": "hi", "extra": True},
        "tu_extra",
        "done",
    )

    with patch.object(engine._client, "stream_messages", side_effect=streams):
        events = list(engine.submit("use echo"))

    tool_result = [e for e in events if e[0] == "tool_result"][0][3]
    assert tool_result.is_error
    assert "unexpected field 'extra'" in tool_result.content


def test_engine_rejects_invalid_enum_tool_input():
    from tools.todo import TodoUpdateTool
    from features.todo import TodoManager

    engine = _make_engine_with_tools(TodoUpdateTool(TodoManager()))
    streams = _make_tool_then_text_response(
        "TodoUpdate",
        {"id": "1", "status": "paused"},
        "tu_enum",
        "done",
    )

    with patch.object(engine._client, "stream_messages", side_effect=streams):
        events = list(engine.submit("update todo"))

    tool_result = [e for e in events if e[0] == "tool_result"][0][3]
    assert tool_result.is_error
    assert "expected one of ['pending', 'in_progress', 'completed']" in tool_result.content


def test_engine_validates_nested_array_object_input():
    engine = _make_engine_with_tools(NestedSchemaTool())
    streams = _make_tool_then_text_response(
        "Nested",
        {
            "questions": [{
                "question": "q1",
                "options": [{"label": "a", "description": "A"}],
            }],
        },
        "tu_nested",
        "done",
    )

    with patch.object(engine._client, "stream_messages", side_effect=streams):
        events = list(engine.submit("use nested"))

    tool_result = [e for e in events if e[0] == "tool_result"][0][3]
    assert tool_result.is_error
    assert "expected at least 2 items for 'questions[0].options'" in tool_result.content


def test_engine_blocks_duplicate_tool_calls_in_same_turn():
    tool = CountingEchoTool()
    engine = _make_engine_with_tools(tool)
    first_stream = _make_tool_use_message([
        {"type": "tool_use", "id": "tu_1", "name": "Echo", "input": {"message": "same"}},
        {"type": "tool_use", "id": "tu_2", "name": "Echo", "input": {"message": "same"}},
    ])
    second_stream = _make_text_response("done")

    with patch.object(engine._client, "stream_messages", side_effect=[first_stream, second_stream]) as stream:
        events = list(engine.submit("use echo twice"))

    tool_results = [e for e in events if e[0] == "tool_result"]
    assert len(tool_results) == 2
    assert tool.calls == 1
    assert tool_results[0][3].is_error is False
    assert tool_results[1][3].is_error is True
    assert "Duplicate tool call blocked for Echo" in tool_results[1][3].content

    second_messages = stream.call_args_list[1].kwargs["messages"]
    tool_result_message = second_messages[2]
    assert tool_result_message["content"][1]["is_error"] is True
    assert "Duplicate tool call blocked for Echo" in tool_result_message["content"][1]["content"]


def test_engine_allows_same_tool_with_different_inputs():
    tool = CountingEchoTool()
    engine = _make_engine_with_tools(tool)
    first_stream = _make_tool_use_message([
        {"type": "tool_use", "id": "tu_1", "name": "Echo", "input": {"message": "one"}},
        {"type": "tool_use", "id": "tu_2", "name": "Echo", "input": {"message": "two"}},
    ])
    second_stream = _make_text_response("done")

    with patch.object(engine._client, "stream_messages", side_effect=[first_stream, second_stream]):
        events = list(engine.submit("use echo twice"))

    tool_results = [e for e in events if e[0] == "tool_result"]
    assert len(tool_results) == 2
    assert tool.calls == 2
    assert all(result[3].is_error is False for result in tool_results)


def test_engine_truncates_long_tool_results_before_follow_up_request():
    engine = _make_engine_with_tools(EchoLongTool())
    long_message = "A" * 9000 + "[stderr]\nboom\n[exit code: 7]" + "Z" * 4500
    streams = _make_tool_then_text_response("EchoLong", {"message": long_message}, "tu_long", "done")

    with patch.object(engine._client, "stream_messages", side_effect=streams) as stream:
        events = list(engine.submit("use long tool"))

    tool_result = [e for e in events if e[0] == "tool_result"][0][3]
    assert len(tool_result.content) < len(long_message)
    assert "tool result truncated" in tool_result.content
    assert "[stderr]\nboom\n[exit code: 7]" in tool_result.content

    second_messages = stream.call_args_list[1].kwargs["messages"]
    tool_result_message = second_messages[2]
    assert tool_result_message["content"][0]["content"] == tool_result.content


def test_engine_normalizes_empty_tool_result_to_no_output():
    class EmptyTool(Tool):
        name = "Empty"
        description = "Returns empty output"
        input_schema = {
            "type": "object",
            "properties": {},
        }

        def execute(self) -> ToolResult:
            return ToolResult(content="   \n")

    engine = _make_engine_with_tools(EmptyTool())
    streams = _make_tool_then_text_response("Empty", {}, "tu_empty", "done")

    with patch.object(engine._client, "stream_messages", side_effect=streams):
        events = list(engine.submit("use empty"))

    tool_result = [e for e in events if e[0] == "tool_result"][0][3]
    assert tool_result.content == "(no output)"
