"""
Tolerant ReAct output parser for noisy Gemini output.

Real-world failures we have observed and have to handle here:

1. Gemini-2.5-flash sometimes emits TWO consecutive Action blocks in
   one step, with no newline separating the JSON Action Input of the
   first block from the next "Thought:" line. The previous greedy
   regex slurped everything from "Action Input:" to end-of-text into
   the tool input, which then got fed verbatim into the RAG
   pipeline's query rewriter and confused it.

2. The model writes "Final Answer formulation:" (or markdown-wrapped
   variants like "**Final Answer:**") before the actual final answer.
   The previous parser only checked for the literal substring
   "Final Answer:" and fell through to a fallback that dumped the
   entire chain-of-thought as the user-visible response.

3. Action Input arrives as `{"query": "..."}` JSON with trailing junk
   (more "Thought:" text directly concatenated). `json.loads` raises
   and we used to keep the entire blob as a plain-string tool input.

This parser deliberately makes only minimal, defensive changes. It
does NOT try to second-guess the model — it just refuses to forward
obviously malformed output downstream.
"""

from __future__ import annotations

import json
import re

from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain.schema import AgentAction, AgentFinish

# Matches the "Final Answer:" marker line ITSELF (not the answer body).
# We intentionally do NOT include a capture group for the answer body
# here, because a greedy DOTALL capture would lock onto the first match
# and prevent us from finding a later marker that has the real answer.
# The body is sliced from text[match.end():] in the parser.
#
# Tolerates leading markdown noise (`*`, `#`, `>`, `-`), an arbitrary
# word between "Final Answer" and the colon (e.g. "formulation"), and
# trailing markdown after the colon (e.g. `**` from `**Final Answer:**`).
_FINAL_ANSWER_MARKER_RE = re.compile(
    r"(?im)^[\s>*#\-]*final\s*answer\b[^:\n]*:[\s*]*",
)

# Match "Action: <name>" followed by "Action Input:" non-greedily.
# We capture only the action name here; the input is extracted separately
# from the tail because the model often forgets to put a newline between
# the input and the next Thought/Action block.
_ACTION_RE = re.compile(
    r"Action:\s*([^\n]+?)\s*\n+\s*Action\s*Input:\s*",
    re.IGNORECASE,
)


def _extract_json_object(text: str) -> str | None:
    """
    If `text` (after lstrip) starts with `{`, return the substring up
    to and including the matching closing `}`. Otherwise return None.

    This intentionally does not use json.loads to find the boundary,
    so a malformed object still gets a chance at parsing further down.
    """
    text = text.lstrip()
    if not text.startswith("{"):
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]
    return None


class SafeReActOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str):
        text = text.strip()

        # --- 1. Try Action / Action Input (FIRST occurrence only). ---
        action_match = _ACTION_RE.search(text)
        if action_match:
            action = action_match.group(1).strip()
            tail = text[action_match.end() :]

            json_blob = _extract_json_object(tail)
            if json_blob is not None:
                action_input_raw = json_blob
                try:
                    tool_input = json.loads(action_input_raw)
                except json.JSONDecodeError:
                    # JSON-shaped but malformed — keep just the blob, not
                    # the trailing chain-of-thought.
                    tool_input = action_input_raw
            else:
                # Plain-string Action Input — keep the first line only.
                first_line = tail.split("\n", 1)[0].strip()
                tool_input = first_line.strip('"').strip("'")

            return AgentAction(tool=action, tool_input=tool_input, log=text)

        # --- 2. Final Answer (tolerant). ---
        # Slice from after the LAST marker so a "Final Answer formulation:"
        # preamble doesn't shadow the real "Final Answer:" that follows.
        markers = list(_FINAL_ANSWER_MARKER_RE.finditer(text))
        if markers:
            body = text[markers[-1].end() :]
            # Strip enclosing quotes and trailing markdown noise.
            answer = body.strip().strip('"').strip("'").strip("*").strip()
            if answer:
                return AgentFinish(return_values={"output": answer}, log=text)

        # --- 3. Last-resort fallback. ---
        # The model emitted neither a clean Action block nor a Final Answer
        # marker. Returning the entire blob leaks chain-of-thought to the
        # end user, so take just the trailing paragraph (text after the
        # last blank line) as a best-effort answer.
        last_paragraph = text.rsplit("\n\n", 1)[-1].strip()
        return AgentFinish(
            return_values={"output": last_paragraph or text},
            log=text,
        )
