"""Generic OpenAI tool-use loop."""

from __future__ import annotations

import json
import time
from typing import Any

from .tools import Tool


DEFAULT_LLM_MODEL = "gpt-5.5"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    data = {"role": getattr(message, "role", "assistant")}
    content = getattr(message, "content", None)
    if content is not None:
        data["content"] = content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is not None:
        data["tool_calls"] = tool_calls
    return data


def _tool_call_parts(tool_call: Any) -> tuple[str, str, str]:
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        return (
            tool_call.get("id", ""),
            function.get("name", ""),
            function.get("arguments") or "{}",
        )
    function = getattr(tool_call, "function")
    return (
        getattr(tool_call, "id", ""),
        getattr(function, "name", ""),
        getattr(function, "arguments", None) or "{}",
    )


def _load_args(raw_args: str, tool_name: str) -> dict[str, Any]:
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool '{tool_name}' received invalid JSON args") from exc
    if not isinstance(args, dict):
        raise ValueError(f"Tool '{tool_name}' args must decode to an object")
    return args


def _default_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def run_agent(
    system: str,
    user: str,
    tools: list[Tool],
    *,
    model: str = DEFAULT_LLM_MODEL,
    client: Any | None = None,
    max_step_turns: int = 5,
    recorder: Any | None = None,
    agent_name: str = "agent",
) -> Any:
    """Run a tool-using agent until a terminal tool returns a payload."""

    if not tools:
        raise ValueError("run_agent requires at least one tool")
    tool_by_name = {tool.name: tool for tool in tools}
    if len(tool_by_name) != len(tools):
        raise ValueError("Tool names must be unique")
    for tool in tools:
        tool.validate()

    client = client or _default_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_skipped_terminal_result: Any | None = None
    last_step_key: str | None = None
    step_turns = 0

    def check_step_limit(step_key: str) -> None:
        nonlocal last_step_key, step_turns
        if step_key == last_step_key:
            step_turns += 1
        else:
            last_step_key = step_key
            step_turns = 1
        if step_turns > max_step_turns:
            raise RuntimeError(
                f"Agent exceeded max_step_turns={max_step_turns} for step: {step_key}"
            )

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool.openai_schema() for tool in tools],
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(_message_to_dict(message))
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        if not tool_calls:
            if last_skipped_terminal_result is not None:
                check_step_limit(
                    "text_after_skipped_terminal:"
                    + json.dumps(_jsonable(last_skipped_terminal_result), sort_keys=True)
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous terminal tool call was skipped because the "
                            "task is not complete. Continue using tools until a "
                            "terminal tool succeeds. Skipped result:\n"
                            + json.dumps(_jsonable(last_skipped_terminal_result))
                        ),
                    }
                )
                continue
            return getattr(message, "content", None)

        terminal_result: Any | None = None
        terminal_seen = False
        skipped_terminal_result: Any | None = None
        step_parts: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_call_id, tool_name, raw_args = _tool_call_parts(tool_call)
            if tool_name not in tool_by_name:
                raise ValueError(f"Unknown tool requested: {tool_name}")
            tool = tool_by_name[tool_name]
            args = _load_args(raw_args, tool_name)
            started = time.perf_counter()
            result = tool.handler(args)
            latency_ms = int((time.perf_counter() - started) * 1000)
            if recorder is not None:
                recorder.record_tool_call(
                    agent=agent_name,
                    tool=tool_name,
                    args=args,
                    result=result,
                    latency_ms=latency_ms,
                )
            skipped_terminal = (
                tool.terminal
                and isinstance(result, dict)
                and result.get("skipped") is True
            )
            if skipped_terminal:
                skipped_terminal_result = result
            if tool.terminal and not skipped_terminal:
                terminal_result = result
                terminal_seen = True
            step_parts.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(_jsonable(result)),
                }
            )
        if terminal_seen:
            return terminal_result
        check_step_limit(json.dumps(_jsonable(step_parts), sort_keys=True))
        last_skipped_terminal_result = skipped_terminal_result
