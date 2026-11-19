#!/usr/bin/env python3
"""Fallback transcript compiler (DECISIONS section 17: used when the `conversation-compiler` skill's
VCC.py is not available; it was not found on any machine of this project on 2026-11-18).

Reads Claude Code stream-json (`claude -p --output-format stream-json --verbose`) or the session JSONL
files Claude Code writes under ~/.claude/projects, and produces a redacted Markdown narrative plus a
JSON summary: one entry per assistant turn with the text, every tool call (name, redacted input summary),
every tool result (length and a redacted head), token usage and cost where present, and subagent turns
grouped under their parent tool call. The raw records are never modified; this is a derived view.

Redaction: strings that look like API keys or OAuth tokens (sk-ant-..., long base64/hex runs), IPv4
addresses, hostnames of the form pod4-*/gb300-*, and absolute paths under /mnt/shared or /home/<user>
are replaced by placeholders (section 18). The summary reports how many replacements were made so a
reviewer can see redaction happened.

    compile_transcript.py INPUT... --out DIR [--title TITLE]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REDACTIONS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "<redacted-token>"),
    (re.compile(r"\b[A-Za-z0-9+/_\-]{48,}={0,2}\b"), "<redacted-long-string>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b(?:pod4|gb300)[A-Za-z0-9_\-]*"), "<host>"),
    (re.compile(r"/mnt/shared/home/[A-Za-z0-9_\-]+"), "/mnt/shared/home/<user>"),
    (re.compile(r"/home/(?!agent\b)[A-Za-z0-9_\-]+"), "/home/<user>"),
]


class Redactor:
    def __init__(self):
        self.counts = Counter()

    def __call__(self, text: str) -> str:
        for pattern, replacement in REDACTIONS:
            text, n = pattern.subn(replacement, text)
            self.counts[replacement] += n
        return text


def read_events(path: Path) -> list[dict]:
    events = []
    for number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            events.append({"type": "unparsable", "line": number})
    return events


def message_of(event: dict) -> dict | None:
    """Both stream-json ({type: assistant, message: {...}}) and session JSONL ({type: assistant, message: {...}})."""
    if event.get("type") in ("assistant", "user") and isinstance(event.get("message"), dict):
        return event["message"]
    return None


def summarize_input(value, redact: Redactor, limit: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = redact(text)
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


def compile_events(events: list[dict], redact: Redactor) -> dict:
    turns = []
    tool_calls = Counter()
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    cost = None
    result = None
    pending_tools: dict[str, dict] = {}
    for event in events:
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            turns.append({"kind": "init", "model": event.get("model"), "tools": len(event.get("tools", [])),
                          "permission_mode": event.get("permissionMode"), "claude_code_version": event.get("version")})
            continue
        if kind == "result":
            result = {k: event.get(k) for k in ("subtype", "duration_ms", "duration_api_ms", "num_turns", "total_cost_usd",
                                                "is_error", "stop_reason")}
            cost = event.get("total_cost_usd")
            continue
        message = message_of(event)
        if message is None:
            continue
        parent = event.get("parent_tool_use_id")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for block in content or []:
            btype = block.get("type")
            if btype == "text" and message.get("role") == "assistant":
                turns.append({"kind": "assistant_text", "parent_tool_use_id": parent, "text": redact(block["text"])})
            elif btype == "thinking":
                turns.append({"kind": "thinking", "parent_tool_use_id": parent, "chars": len(block.get("thinking", ""))})
            elif btype == "tool_use":
                tool_calls[block.get("name")] += 1
                entry = {"kind": "tool_use", "parent_tool_use_id": parent, "id": block.get("id"), "name": block.get("name"),
                         "input": summarize_input(block.get("input"), redact)}
                pending_tools[block.get("id")] = entry
                turns.append(entry)
            elif btype == "tool_result":
                raw = block.get("content")
                text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False) if raw is not None else ""
                turns.append({"kind": "tool_result", "parent_tool_use_id": parent, "tool_use_id": block.get("tool_use_id"),
                              "is_error": block.get("is_error", False), "chars": len(text),
                              "head": summarize_input(text, redact, 400)})
        for key in usage:
            usage[key] += int((message.get("usage") or {}).get(key) or 0)
    return {"turns": turns, "tool_calls": dict(tool_calls), "usage": usage, "total_cost_usd": cost, "result": result,
            "events": len(events), "unparsable": sum(1 for e in events if e.get("type") == "unparsable")}


def to_markdown(title: str, compiled: dict, redactions: Counter) -> str:
    lines = [f"# {title}", "", f"Events: {compiled['events']} (unparsable: {compiled['unparsable']})",
             f"Tool calls: {json.dumps(compiled['tool_calls'], sort_keys=True)}",
             f"Tokens: {json.dumps(compiled['usage'])}", f"Cost (USD): {compiled['total_cost_usd']}",
             f"Result: {json.dumps(compiled['result'])}", f"Redactions: {json.dumps(dict(redactions))}", ""]
    for turn in compiled["turns"]:
        sub = " (subagent)" if turn.get("parent_tool_use_id") else ""
        if turn["kind"] == "init":
            lines.append(f"## Session init{sub}: model {turn['model']}, Claude Code {turn['claude_code_version']}, "
                         f"{turn['tools']} tools, permission mode {turn['permission_mode']}")
        elif turn["kind"] == "assistant_text":
            lines += [f"### Assistant{sub}", "", turn["text"], ""]
        elif turn["kind"] == "thinking":
            lines.append(f"_thinking: {turn['chars']} chars (not reproduced)_")
        elif turn["kind"] == "tool_use":
            lines += [f"- **{turn['name']}**{sub}: `{turn['input'].replace('`', "'")}`"]
        elif turn["kind"] == "tool_result":
            flag = " ERROR" if turn["is_error"] else ""
            lines += [f"  - result{flag} ({turn['chars']} chars): {turn['head'].replace(chr(10), ' ')[:400]}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Goal agent trajectory (compiled, redacted)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    redact = Redactor()
    events = []
    for path in args.inputs:
        events += read_events(path)
    compiled = compile_events(events, redact)
    (args.out / "trajectory.json").write_text(json.dumps(compiled | {"redactions": dict(redact.counts)}, indent=1))
    (args.out / "trajectory.md").write_text(to_markdown(args.title, compiled, redact.counts))
    print(json.dumps({"events": compiled["events"], "turns": len(compiled["turns"]), "tool_calls": sum(compiled["tool_calls"].values()),
                      "redactions": dict(redact.counts), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
