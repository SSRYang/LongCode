from __future__ import annotations
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from .config import DEFAULT_MODEL, default_max_tokens_for_model, resolve_model
from .llm import LLMClient
from .tool import Tool, ToolResult
from .permissions import PermissionChecker

if TYPE_CHECKING:
    from features.cost_tracker import CostTracker
    from .session import SessionStore

_MAX_RETRIES = 10 #  最大重试次数常量
_BASE_DELAY = 0.5 #  基础延迟时间常量（秒）
_MAX_DELAY = 32.0 #  最大延迟时间常量（秒）
_JITTER_FACTOR = 0.25 #  抖动因子常量，用于在重试延迟中引入随机性
_MAX_TOOL_RESULT_CHARS = 12_000
_TOOL_RESULT_HEAD_CHARS = 8_000
_TOOL_RESULT_TAIL_CHARS = 3_000
_TOOL_RESULT_DIAGNOSTIC_MARKERS = ("[stderr]", "Traceback", "[exit code:", "Error:")
_TOOL_RESULT_TRUNCATION_TEMPLATE = "\n\n... (tool result truncated, showing first {head} characters and a diagnostic tail segment; original length {original})"


def _compute_retry_delay(attempt: int, retry_after: float | None = None) -> float:  # 计算重试延迟时间, 默认指数退避策略，或采用retry_after
    """Exponential backoff with jitter, respecting Retry-After if present."""
    if retry_after is not None and retry_after > 0:
        return retry_after
    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    jitter = delay * random.uniform(0, _JITTER_FACTOR)
    return delay + jitter


def _parse_retry_after(exc: Exception) -> float | None: # 从API错误响应头中提取Retry-After值
    """Extract Retry-After value from API error headers, if available."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


_CONTEXT_OVERFLOW_RE = re.compile( #  定义一个正则表达式模式，用于匹配上下文溢出的错误信息
    r"prompt is too long|max_tokens.*exceeds.*context|input.*too large",
    re.IGNORECASE,
)


class AbortedError(Exception):  # 定义了一个名为 AbortedError 的异常类，用于在用户通过 Esc 键或 Ctrl+C 中断当前操作时抛出
    """Raised when the current turn is aborted by the user (Esc / Ctrl+C)."""


class Engine:
    def __init__(self, tools: list[Tool], system_prompt: str,
                 permission_checker: PermissionChecker,
                 provider: str = "anthropic", #  默认使用Anthropic作为AI模型提供商
                 model: str = DEFAULT_MODEL, #  默认使用的模型
                 max_tokens: int | None = None, #  最大令牌数限制
                 api_key: str | None = None, #  API密钥
                 base_url: str | None = None, #  API基础URL
                 effort: str | None = None, #  努力程度参数
                 session_store: SessionStore | None = None, #  会话存储
                 cost_tracker: CostTracker | None = None, #  成本跟踪器
                 advisor_model: str | None = None, #  顾问模型
                 advisor_max_uses: int | None = None): #  顾问最大使用次数
        self._provider = provider #  初始化提供商和模型
        self._model = resolve_model(model, provider=provider)
        self._max_tokens = max_tokens or default_max_tokens_for_model( #  设置最大令牌数
            self._model,
            provider=provider,
        )
        self._effort = effort
        self._client = LLMClient( #  初始化LLM客户端
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
        self._tools = {t.name: t for t in tools} #  存储工具，以工具名为键
        self._system_prompt = system_prompt #  系统提示词
        self._permissions = permission_checker #  初始化权限检查器
        self._messages: list[dict] = [] #  初始化消息列表，存储对话消息
        self._aborted = False #  初始化中止标志，用于控制对话流程
        self._turn_start_len: int | None = None #  初始化回合开始时的消息长度，用于跟踪对话长度变化
        self._active_stream = None  # reference to current HTTP stream #  初始化当前HTTP流的引用
        self._session_store = session_store #  初始化会话存储，用于保存和管理会话状态
        self._cost_tracker = cost_tracker #  初始化成本跟踪器，用于计算API调用成本和耗时
        self._advisor_model = advisor_model or "claude-opus-4-6" #  初始化顾问模型，默认使用claude-opus-4-6
        self._advisor_max_uses = advisor_max_uses if advisor_max_uses is not None else 3 #  初始化顾问最大使用次数，默认为3
        self._advisor_enabled = False #  初始化顾问功能开关，默认为关闭状态

    # -- advisor toggle --------------------------------------------------------

    def toggle_advisor(self) -> bool:
        """Toggle advisor on/off. Returns new state."""
        self._advisor_enabled = not self._advisor_enabled
        return self._advisor_enabled

    @property
    def advisor_enabled(self) -> bool:
        return self._advisor_enabled

    # -- message accessors (for compact / resume / commands) ----------------

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def set_messages(self, messages: list[dict]) -> None:
        self._messages = [
            {
                "role": message["role"],
                "content": message.get("content", ""),
            }
            for message in messages
        ]

    def set_session_store(self, store: SessionStore | None) -> None:
        self._session_store = store

    def write_turn_artifact(self, artifact: dict) -> str | None:
        if self._session_store is None:
            return None
        try:
            return str(self._session_store.append_turn_artifact(artifact))
        except Exception:
            return None

    def total_cost_usd(self) -> float:
        if self._cost_tracker is None:
            return 0.0
        return self._cost_tracker.total_cost_usd

    def set_tools(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def get_model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        self._model = resolve_model(model, provider=self._provider)
        self._max_tokens = default_max_tokens_for_model(
            self._model,
            provider=self._provider,
        )

    def _persist(self, message: dict) -> None:
        """Append message to session store if available."""
        if self._session_store is not None:
            try:
                self._session_store.append_message(message)
            except Exception:
                pass  # don't break the conversation on I/O errors

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    def last_assistant_text(self) -> str:
        """Extract text from the last assistant message."""
        if not self._messages:
            return ""
        last = self._messages[-1]
        if last.get("role") != "assistant":
            return ""
        content = last.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    def abort(self):
        """Abort the current turn immediately.

        Matches claude-code-main's AbortController.abort(): sets flag and
        closes the active HTTP stream so the generator unblocks at once.
        """
        self._aborted = True
        if self._active_stream is not None:
            try:
                self._active_stream.close()
            except Exception:
                pass

    def cancel_turn(self):
        """Roll back messages to the state before the current turn started.

        Uses _turn_start_len (set at the beginning of submit()) to restore
        messages to the exact state before the turn. This is more robust than
        trying to walk back individual messages, especially when a turn has
        multiple tool_use/tool_result cycles.
        """
        if self._turn_start_len is not None:
            del self._messages[self._turn_start_len:]
            self._turn_start_len = None

    def submit(self, user_input: str | list) -> Iterator[tuple]:
        """Send user message; yield events until the conversation turn completes.

        Yields:
          ("text", str)                         — 流式输出的文本片段
          ("tool_call", name, input, activity)  — 在工具执行前触发
          ("tool_executing", name, input, activity) — 在获取权限后、工具实际运行中触发
          ("tool_result", name, input, result)  — 在工具执行完毕后触发
          ("waiting",)                          — 文本已输出完毕，正在等待工具调用
          ("error", str)                        — 非致命性的 API 错误，会展示给用户

        Raises:
          AbortedError — if abort() was called (by Esc listener or Ctrl+C)
        """
        self._aborted = False
        self._turn_start_len = len(self._messages)
        self._messages.append({
            "role": "user",
            "content": user_input,
        })
        self._persist(self._messages[-1])

        try:
            while True:
                if self._aborted:
                    raise AbortedError()

                tool_uses = []

                # API call with retry
                final = None    # 保存最终结果
                for attempt in range(_MAX_RETRIES):
                    try:
                        # 准备工具列表
                        _api_t0 = time.monotonic()  # 记录API调用开始时间
                        tools = [t.to_api_schema() for t in self._tools.values()]
                        if self._advisor_enabled:
                            tools.append({
                                "type": "advisor_20260301",
                                "name": "advisor",
                                "model": self._advisor_model,
                                "max_uses": self._advisor_max_uses,
                            })
                        stream_obj = self._client.stream_messages(
                            model=self._model,
                            max_tokens=self._max_tokens,
                            system=self._system_prompt,
                            tools=tools,
                            messages=self._messages,
                            effort=self._effort,
                        )
                        self._active_stream = stream_obj

                        with stream_obj as stream:
                            got_text = False    # 是否已输出过文本片段
                            for text in stream.text_stream:
                                if self._aborted:
                                    raise AbortedError()
                                got_text = True
                                yield ("text", text)

                            if self._aborted:
                                raise AbortedError()

                            if got_text:    # 文本输出完成，当前正在等待工具执行
                                yield ("waiting",)

                            final = stream.get_final_message()
                            _api_elapsed = time.monotonic() - _api_t0   # 获取API调用耗时
                            # Track token usage / cost
                            if final.usage and self._cost_tracker:
                                self._cost_tracker.add_usage(self._model, {
                                    "input_tokens": getattr(final.usage, "input_tokens", 0) or 0,
                                    "output_tokens": getattr(final.usage, "output_tokens", 0) or 0,
                                    "cache_read_input_tokens": getattr(final.usage, "cache_read_input_tokens", 0) or 0,
                                    "cache_creation_input_tokens": getattr(final.usage, "cache_creation_input_tokens", 0) or 0,
                                    "advisor_input_tokens": getattr(final.usage, "advisor_input_tokens", 0) or 0,
                                    "advisor_output_tokens": getattr(final.usage, "advisor_output_tokens", 0) or 0,
                                }, api_duration_s=_api_elapsed, advisor_model=self._advisor_model if self._advisor_enabled else None)
                                yield ("usage", final.usage)
                            # Warn if response was truncated by max_tokens
                            if final.stop_reason == "max_tokens":
                                yield ("error", "Response truncated: hit max_tokens limit.")
                            for block in final.content:
                                if _block_type(block) == "tool_use":
                                    tool_uses.append(block)
                        break  # success, exit retry loop
                    except AbortedError:
                        raise
                    except Exception as e:
                        if self._client.is_authentication_error(e): #  检查是否为认证错误
                            self._messages.pop() #  移除最后一条消息
                            yield ("error", f"Authentication failed: {self._client.error_message(e)}") #  返回认证失败错误信息
                            return #  直接返回
                        # Context overflow: reduce max_tokens and retry
                        err_msg = self._client.error_message(e)
                        if self._client.is_api_error(e) and _CONTEXT_OVERFLOW_RE.search(err_msg):
                            reduced = self._max_tokens // 2
                            if reduced >= 1024:
                                self._max_tokens = reduced
                                yield ("error", f"Context overflow, reducing max_tokens to {reduced} and retrying...")
                                continue
                            else:
                                self._messages.pop()
                                yield ("error", f"Context overflow and cannot reduce further: {err_msg}")
                                return
                        if self._client.is_retryable_error(e):
                            if attempt < _MAX_RETRIES - 1:
                                retry_after = _parse_retry_after(e)
                                wait = _compute_retry_delay(attempt, retry_after)
                                yield ("error", f"API error, retrying in {wait:.1f}s... ({err_msg})")
                                time.sleep(wait)
                            else:
                                self._messages.pop()
                                yield ("error", f"API error after {_MAX_RETRIES} retries: {err_msg}")
                                return
                            continue
                        if self._client.is_api_error(e):
                            self._messages.pop()
                            yield ("error", f"API error: {err_msg}")
                            return
                        if self._aborted:
                            raise AbortedError()
                        raise
                    finally:
                        self._active_stream = None

                if final is None:
                    self._messages.pop()
                    return

                # 更新历史消息
                self._messages.append({
                    "role": "assistant",
                    "content": final.content,
                })
                self._persist(self._messages[-1])

                if not tool_uses:
                    break

                tool_results = []
                duplicate_results = self._build_duplicate_results(tool_uses)

                # 将工具调用分批：连续的只读工具并行执行，非只读工具单独执行
                batches: list[list] = []    # [(布尔值, [工具调用对象列表]), ...]，其中布尔值代表该批次是否支持并发。
                for tu in tool_uses:
                    if _block_id(tu) in duplicate_results:
                        batches.append((False, [tu]))
                        continue
                    t = self._tools.get(_block_name(tu))
                    is_concurrent = t is not None and t.is_read_only()  # 只读工具
                    if batches and batches[-1][0] == is_concurrent and is_concurrent:   # 当前工具是只读的，且与上一个批次的并发属性一致。
                        batches[-1][1].append(tu)
                    else:
                        batches.append((is_concurrent, [tu]))

                for is_concurrent, batch in batches:
                    if self._aborted:
                        raise AbortedError()

                    if len(batch) == 1 and _block_id(batch[0]) in duplicate_results:
                        tu = batch[0]
                        tn = _block_name(tu)
                        ti = _block_input(tu)
                        tool = self._tools.get(tn)
                        act = self._safe_activity_description(tool, ti)
                        result = duplicate_results[_block_id(tu)]
                        yield ("tool_call", tn, ti, act)
                        yield ("tool_result", tn, ti, result)
                        tool_results.append(_tool_result_block(_block_id(tu), result))
                        continue

                    if is_concurrent and len(batch) > 1:
                        # --- parallel execution for read-only tools ---
                        # Phase 1: emit tool_call events + check permissions
                        approved: list[tuple] = []  # (tool_use, tool, activity)
                        denied_results: dict[str, ToolResult] = {}  # by tool_use_id
                        for tu in batch:
                            tn = _block_name(tu)
                            ti = _block_input(tu)
                            tool = self._tools.get(tn)
                            act = self._safe_activity_description(tool, ti)
                            yield ("tool_call", tn, ti, act)
                            preflight = self._preflight_tool_use(tool, tn, ti)
                            if preflight is not None:
                                denied_results[_block_id(tu)] = preflight
                            elif self._permissions.check(tool, ti) == "deny":
                                denied_results[_block_id(tu)] = self._normalize_tool_result(
                                    tn,
                                    ToolResult(content="Permission denied.", is_error=True),
                                )
                            else:
                                approved.append((tu, tool, act))

                        # Phase 2: emit tool_executing for approved, then run in parallel
                        executed_results: dict[str, ToolResult] = {}
                        if approved:
                            for tu, tool, act in approved:
                                tn = _block_name(tu)
                                ti = _block_input(tu)
                                yield ("tool_executing", tn, ti, act)

                            with ThreadPoolExecutor(max_workers=min(len(approved), 10)) as pool:
                                futures = {}
                                for tu, tool, act in approved:
                                    f = pool.submit(self._execute_tool, tu, skip_permission=True)
                                    futures[f] = tu
                                for f in as_completed(futures): # 在各个线程完成时捕获结果或异常
                                    tu = futures[f]
                                    try:
                                        executed_results[_block_id(tu)] = f.result()
                                    except Exception as exc:
                                        executed_results[_block_id(tu)] = self._normalize_tool_result(
                                            _block_name(tu),
                                            ToolResult(content=f"Error: Tool execution failed for {_block_name(tu)}: {exc}", is_error=True),
                                        )

                        # Phase 3: emit results in original batch order
                        for tu in batch:
                            tid = _block_id(tu)
                            tn = _block_name(tu)
                            ti = _block_input(tu)
                            result = denied_results.get(tid) or executed_results.get(tid)   # 按调用顺序输出结果
                            if result is None:
                                result = self._normalize_tool_result(
                                    tn,
                                    ToolResult(content=f"Error: Tool execution failed for {tn}: no result returned.", is_error=True),
                                )
                            yield ("tool_result", tn, ti, result)
                            tool_results.append(_tool_result_block(tid, result))
                    else:   # 串行执行
                        # --- sequential execution (single tool or non-read-only) ---
                        for tu in batch:
                            if self._aborted:
                                raise AbortedError()
                            tn = _block_name(tu)
                            ti = _block_input(tu)
                            tool = self._tools.get(tn)
                            act = self._safe_activity_description(tool, ti)
                            yield ("tool_call", tn, ti, act)

                            preflight = self._preflight_tool_use(tool, tn, ti)
                            if preflight is not None:
                                result = preflight
                            elif self._permissions.check(tool, ti) == "deny":
                                result = self._normalize_tool_result(
                                    tn,
                                    ToolResult(content="Permission denied.", is_error=True),
                                )
                            else:
                                yield ("tool_executing", tn, ti, act)
                                result = self._execute_tool(tu, skip_permission=True)

                            yield ("tool_result", tn, ti, result)
                            tool_results.append(_tool_result_block(_block_id(tu), result))

                self._messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                self._persist(self._messages[-1])
        except AbortedError:
            self.cancel_turn()
            raise

    def _build_duplicate_results(self, tool_uses: list[Any]) -> dict[str, ToolResult]:
        seen: dict[str, str] = {}
        duplicates: dict[str, ToolResult] = {}
        for tool_use in tool_uses:
            tool_name = _block_name(tool_use)
            tool_input = _block_input(tool_use)
            fingerprint = _tool_call_fingerprint(tool_name, tool_input)
            tool_use_id = _block_id(tool_use)
            if fingerprint in seen:
                duplicates[tool_use_id] = self._normalize_tool_result(
                    tool_name,
                    ToolResult(
                        content=(
                            f"Error: Duplicate tool call blocked for {tool_name} "
                            "with identical input in the same assistant response."
                        ),
                        is_error=True,
                    ),
                )
            else:
                seen[fingerprint] = tool_use_id
        return duplicates

    def _preflight_tool_use(self, tool: Tool | None, tool_name: str, tool_input: dict[str, Any]) -> ToolResult | None:
        if tool is None:
            return self._normalize_tool_result(
                tool_name,
                ToolResult(content=f"Error: Unknown tool: {tool_name}", is_error=True),
            )

        validation_error = tool.validate_input(tool_input)
        if validation_error is not None:
            return self._normalize_tool_result(
                tool_name,
                ToolResult(
                    content=f"Error: Invalid input for {tool_name}: {validation_error}.",
                    is_error=True,
                ),
            )
        return None

    def _safe_activity_description(self, tool: Tool | None, tool_input: dict[str, Any]) -> str | None:
        if tool is None:
            return None
        try:
            return tool.get_activity_description(**tool_input)
        except Exception:
            return None

    def _normalize_tool_result(self, tool_name: str, result: ToolResult) -> ToolResult:
        content = result.content
        if isinstance(content, str):
            text = content
        elif content is None:
            text = ""
        else:
            text = json.dumps(content, ensure_ascii=False)

        text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip()
        if not text:
            text = "(no output)"

        if len(text) > _MAX_TOOL_RESULT_CHARS:
            head = text[:_TOOL_RESULT_HEAD_CHARS]
            tail = _diagnostic_tail_segment(text)
            marker = _TOOL_RESULT_TRUNCATION_TEMPLATE.format(
                head=_TOOL_RESULT_HEAD_CHARS,
                original=len(text),
            )
            text = head + marker + "\n\n" + tail

        return ToolResult(content=text, is_error=result.is_error)

    def _execute_tool(self, tool_use, skip_permission: bool = False) -> ToolResult:
        tool_name = _block_name(tool_use)
        tool_input = _block_input(tool_use)
        tool = self._tools.get(tool_name)
        preflight = self._preflight_tool_use(tool, tool_name, tool_input)
        if preflight is not None:
            return preflight

        assert tool is not None

        if not skip_permission and self._permissions.check(tool, tool_input) == "deny":
            return self._normalize_tool_result(
                tool_name,
                ToolResult(content="Permission denied.", is_error=True),
            )

        try:
            # Snapshot file for diff if it's a write tool we want to track
            old_lines: list[str] | None = None
            if self._cost_tracker and tool_name in ("Edit", "Write"):
                fp = tool_input.get("file_path", "")
                try:
                    p = Path(fp)
                    old_lines = p.read_text().splitlines() if p.exists() else []
                except Exception:
                    old_lines = None

            result = tool.execute(**tool_input)

            # Track line changes for Edit/Write
            if self._cost_tracker and old_lines is not None and not result.is_error:
                fp = tool_input.get("file_path", "")
                try:
                    new_lines = Path(fp).read_text().splitlines()
                    added = max(len(new_lines) - len(old_lines), 0)
                    removed = max(len(old_lines) - len(new_lines), 0)
                    self._cost_tracker.add_lines_changed(added, removed)
                except Exception:
                    pass

            return self._normalize_tool_result(tool_name, result)
        except Exception as e:
            return self._normalize_tool_result(
                tool_name,
                ToolResult(content=f"Error: Tool execution failed for {tool_name}: {e}", is_error=True),
            )


def _tool_result_block(tool_use_id: str, result: ToolResult) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result.content,
        "is_error": result.is_error,
    }


def _tool_call_fingerprint(tool_name: str, tool_input: dict[str, Any]) -> str:
    try:
        normalized = json.dumps(tool_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        normalized = repr(tool_input)
    return f"{tool_name}:{normalized}"


def _diagnostic_tail_segment(text: str) -> str:
    for marker in _TOOL_RESULT_DIAGNOSTIC_MARKERS:
        index = text.rfind(marker)
        if index != -1:
            return text[index:]
    return text[-_TOOL_RESULT_TAIL_CHARS:]


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_name(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("name", ""))
    return str(getattr(block, "name", ""))


def _block_id(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("id", ""))
    return str(getattr(block, "id", ""))


def _block_input(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        value = block.get("input", {})
    else:
        value = getattr(block, "input", {})
    return value if isinstance(value, dict) else {}
