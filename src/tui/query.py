"""Run a single query turn with TUI feedback (spinner, markdown streaming)."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rich.console import Console

from core.engine import AbortedError, Engine
from core.session import _extract_text
from tui.keylistener import EscListener
from core.permissions import PermissionChecker
from tui.rendering import (
    StreamingMarkdown,
    SpinnerManager,
    tool_preview,
    collapsed_tool_summary,
    render_todo_list,
)

if TYPE_CHECKING:
    from features.todo import TodoManager

console = Console()

_TODO_TOOL_NAMES = frozenset({"TodoWrite", "TodoUpdate"})
_TURN_PATH_KEYS = frozenset({
    "file_path", "path", "paths", "cwd", "notebook_path", "localDir", "scriptPath",
})


def _usage_to_dict(usage: Any) -> dict[str, int]:
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        "advisor_input_tokens": int(getattr(usage, "advisor_input_tokens", 0) or 0),
        "advisor_output_tokens": int(getattr(usage, "advisor_output_tokens", 0) or 0),
    }



def _extract_paths_from_value(value: Any) -> list[str]:
    found: set[str] = set()

    def _walk(current: Any, key: str | None = None) -> None:
        if isinstance(current, dict):
            for child_key, child_value in current.items():
                if child_key in _TURN_PATH_KEYS:
                    if isinstance(child_value, str) and child_value.strip():
                        found.add(child_value)
                    elif isinstance(child_value, list):
                        for item in child_value:
                            if isinstance(item, str) and item.strip():
                                found.add(item)
                            else:
                                _walk(item, child_key)
                    else:
                        _walk(child_value, child_key)
                else:
                    _walk(child_value, child_key)
        elif isinstance(current, list):
            for item in current:
                _walk(item, key)

    _walk(value)
    return sorted(found)



def _input_preview(user_input: str | list) -> tuple[str, int, str]:
    text = " ".join(_extract_text(user_input).split())
    preview = text if len(text) <= 200 else text[:197] + "..."
    kind = "text" if isinstance(user_input, str) else "content_blocks"
    return preview, len(text), kind



def _condense_text(text: str, limit: int = 200) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."



def run_query(engine: Engine, user_input: str | list, print_mode: bool,
              permissions: PermissionChecker | None = None,
              quiet: bool = False,
              todo_manager: TodoManager | None = None,
              turn_context: dict | None = None,
              record_turn: bool = True) -> dict | None:
    """
    执行单轮对话。按 Ctrl+C 或 Esc 键可取消当前正在进行的回合。
    如果将 quiet 设为 True, 则会隐藏所有终端输出（包括加载动画、工具调用信息和文本内容）。
    这个选项通常用于像 auto-dream 这样的后台任务。
    """
    listener = EscListener(on_cancel=engine.abort)
    if permissions:
        permissions.set_esc_listener(listener)

    spinner = SpinnerManager(console)
    md_stream = StreamingMarkdown(console)
    first_text = True
    streaming = False
    pending_tools: dict[str, tuple[str, str]] = {}

    started_at = datetime.now(timezone.utc)
    started_cost = engine.total_cost_usd()
    turn_t0 = time.monotonic()
    input_preview, input_chars, input_kind = _input_preview(user_input)
    timeline: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    paths: set[str] = set()
    errors: list[str] = []
    usage: dict[str, int] | None = None
    text_chunk_count = 0

    def _record(event_type: str, **details: Any) -> None:
        item = {
            "t_ms": int((time.monotonic() - turn_t0) * 1000),
            "type": event_type,
        }
        for key, value in details.items():
            if value in (None, "", [], {}):
                continue
            item[key] = value
        timeline.append(item)

    def _find_pending_tool(tool_name: str, preview: str) -> dict[str, Any] | None:
        for item in reversed(tool_events):
            if item.get("name") == tool_name and item.get("input_preview") == preview and item.get("status") in {"pending", "running"}:
                return item
        return None

    def _finalize(status: str) -> dict[str, Any] | None:
        assistant_text = engine.last_assistant_text()
        try:
            from features.memory import extract_memory_tags
            memory_tags = extract_memory_tags(assistant_text)
        except Exception:
            memory_tags = []

        final_status = status
        if final_status == "completed" and errors:
            final_status = "completed_with_errors"

        artifact = {
            "version": 1,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.monotonic() - turn_t0) * 1000),
            "status": final_status,
            "context": dict(turn_context or {}),
            "input": {
                "kind": input_kind,
                "char_count": input_chars,
                "preview": input_preview,
            },
            "assistant": {
                "char_count": len(assistant_text),
                "preview": _condense_text(assistant_text, limit=400),
                "text_chunk_count": text_chunk_count,
            },
            "timeline": timeline,
            "tools": tool_events,
            "paths": sorted(paths),
            "usage": usage,
            "cost_usd": round(max(engine.total_cost_usd() - started_cost, 0.0), 6),
            "errors": errors,
            "memory": {
                "tags": memory_tags,
                "triggered": bool(memory_tags),
            },
        }
        if record_turn:
            artifact_path = engine.write_turn_artifact(artifact)
            if artifact_path:
                artifact["artifact_path"] = artifact_path
        return artifact

    _record("turn_started", input_kind=input_kind, input_chars=input_chars)

    try:
        with listener:
            if not quiet:
                spinner.start("Thinking…")

            for event in engine.submit(user_input):
                event_type = event[0]

                if not quiet and streaming and listener.pressed:
                    md_stream.flush()
                    spinner.stop()
                    engine.cancel_turn()
                    console.print("\n[dim yellow]⏹ Turn cancelled (Esc)[/dim yellow]")
                    _record("cancelled", reason="esc")
                    return _finalize("cancelled")

                if event_type == "text":
                    text_chunk_count += 1
                    if text_chunk_count == 1:
                        _record("text_started")
                    if quiet:
                        continue
                    if first_text:
                        spinner.stop()
                        streaming = True
                        first_text = False
                    if print_mode:
                        print(event[1], end="", flush=True)
                    else:
                        md_stream.feed(event[1])

                elif event_type == "waiting":
                    _record("waiting")
                    if not quiet:
                        md_stream.flush()
                    streaming = False
                    if not quiet:
                        listener.resume()
                        spinner.start("Preparing tool call…")

                elif event_type == "tool_call":
                    _, tool_name, tool_input, activity = event
                    preview = tool_preview(tool_name, tool_input)
                    key = f"{tool_name}({preview})"
                    tool_paths = _extract_paths_from_value(tool_input)
                    paths.update(tool_paths)
                    tool_events.append({
                        "name": tool_name,
                        "input_preview": preview,
                        "activity": activity,
                        "paths": tool_paths,
                        "status": "pending",
                    })
                    _record("tool_call", tool_name=tool_name, activity=activity, paths=tool_paths)
                    if not quiet:
                        spinner.stop()
                        streaming = False
                        listener.pause()
                        pending_tools[key] = (tool_name, f"↳ {key}")

                elif event_type == "tool_executing":
                    _, tool_name, tool_input, activity = event
                    preview = tool_preview(tool_name, tool_input)
                    pending = _find_pending_tool(tool_name, preview)
                    if pending is not None:
                        pending["status"] = "running"
                    _record("tool_executing", tool_name=tool_name, activity=activity)
                    if not quiet:
                        n = len(pending_tools)
                        if tool_name == "AskUserQuestion":
                            spinner.stop()
                            _, line = next(iter(pending_tools.values()), ("", f"↳ {tool_name}"))
                            console.print(f"[dim]{line}[/dim]", highlight=False)
                        elif n > 1:
                            names = [tn for tn, _ in pending_tools.values()]
                            spinner.start(collapsed_tool_summary(names))
                        else:
                            _, line = next(iter(pending_tools.values()), ("", f"↳ {tool_name}"))
                            activity_text = activity or f"Running {tool_name}…"
                            spinner.start(f"{line} … {activity_text}")

                elif event_type == "tool_result":
                    _, tool_name, tool_input, result = event
                    preview = tool_preview(tool_name, tool_input)
                    pending = _find_pending_tool(tool_name, preview)
                    if pending is not None:
                        pending["status"] = "denied" if result.is_error and result.content == "Permission denied." else ("error" if result.is_error else "completed")
                        pending["result_preview"] = _condense_text(result.content, limit=240)
                    _record(
                        "tool_result",
                        tool_name=tool_name,
                        ok=not result.is_error,
                        result_preview=_condense_text(result.content, limit=160),
                    )
                    if not quiet:
                        spinner.stop()
                        key = f"{tool_name}({preview})"
                        _, line = pending_tools.pop(key, (tool_name, f"↳ {key}"))

                        if tool_name in _TODO_TOOL_NAMES and todo_manager is not None:
                            if result.is_error:
                                console.print(f"[dim]{line}[/dim] [red]✗[/red]", highlight=False)
                                console.print(f"  [red]{result.content[:200]}[/red]")
                            else:
                                render_todo_list(todo_manager.get_items(), console)
                        elif result.is_error:
                            console.print(f"[dim]{line}[/dim] [red]✗[/red]", highlight=False)
                            console.print(f"  [red]{result.content[:200]}[/red]")
                        else:
                            console.print(f"[dim]{line}[/dim] [green]✓[/green]", highlight=False)

                        if pending_tools:
                            names = [tn for tn, _ in pending_tools.values()]
                            spinner.start(collapsed_tool_summary(names))
                        else:
                            streaming = False
                            listener.resume()
                            spinner_text = "Thinking…"
                            if todo_manager is not None:
                                wip = todo_manager.in_progress_item()
                                if wip:
                                    label = wip.subject
                                    if len(label) > 60:
                                        label = label[:57] + "…"
                                    spinner_text = label
                            spinner.start(spinner_text)
                            first_text = True

                elif event_type == "usage":
                    usage = _usage_to_dict(event[1])
                    _record("usage", **{k: v for k, v in usage.items() if v})

                elif event_type == "error":
                    errors.append(event[1])
                    _record("error", message=_condense_text(event[1], limit=240))
                    if not quiet:
                        md_stream.flush()
                        spinner.stop()
                        console.print(f"\n[bold red]{event[1]}[/bold red]")

            md_stream.flush()
            spinner.stop()
    except (AbortedError, KeyboardInterrupt):
        md_stream.flush()
        spinner.stop()
        if not isinstance(sys.exc_info()[1], AbortedError):
            engine.cancel_turn()
        if not quiet:
            console.print("\n[dim yellow]⏹ Turn cancelled[/dim yellow]")
        _record("cancelled", reason="interrupt")
        return _finalize("cancelled")
    finally:
        md_stream.flush()
        spinner.stop()
        if permissions:
            permissions.set_esc_listener(None)

    if not print_mode:
        console.print()
    return _finalize("completed")
