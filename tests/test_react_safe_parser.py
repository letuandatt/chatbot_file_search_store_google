"""
Tests for the SafeReActOutputParser.

The parser has to be tolerant of the noisy output that Gemini-2.5-flash
actually produces in production, so most of these tests are pinned to
real captured traces (slightly anonymised) rather than synthetic
happy-path samples.
"""

from __future__ import annotations

from langchain.schema import AgentAction, AgentFinish

from chatbot.llm.react_safe_parser import SafeReActOutputParser


def _parser() -> SafeReActOutputParser:
    return SafeReActOutputParser()


# --- Happy path -----------------------------------------------------------


def test_parses_clean_action_block() -> None:
    text = (
        "Thought: tra cứu pháp luật.\n"
        "Action: tool_search_law\n"
        'Action Input: {"query": "Chỉ thị 35/CT-TTg"}'
    )
    out = _parser().parse(text)
    assert isinstance(out, AgentAction)
    assert out.tool == "tool_search_law"
    assert out.tool_input == {"query": "Chỉ thị 35/CT-TTg"}


def test_parses_clean_final_answer() -> None:
    text = (
        "Thought: I have the info.\n"
        "Final Answer: Theo Điều 7, Luật Doanh nghiệp 2020..."
    )
    out = _parser().parse(text)
    assert isinstance(out, AgentFinish)
    assert out.return_values["output"].startswith("Theo Điều 7")


def test_parses_plain_string_action_input() -> None:
    text = "Action: tool_search_law\n" 'Action Input: "Chỉ thị 35/CT-TTg năm 2025"'
    out = _parser().parse(text)
    assert isinstance(out, AgentAction)
    assert out.tool == "tool_search_law"
    assert out.tool_input == "Chỉ thị 35/CT-TTg năm 2025"


# --- Malformed Gemini output: real captures -------------------------------


def test_two_action_blocks_only_first_is_executed() -> None:
    # Real capture: Gemini emitted TWO consecutive Action blocks with
    # no newline between the JSON Action Input of the first block and
    # the "Thought:" of the second block. The parser must NOT slurp
    # the second block into the first tool input.
    text = (
        "Thought: search.\n"
        "Action: tool_search_law\n"
        'Action Input: {"query": "Chỉ thị số 35/CT-TTg"}'
        "Thought: thử lại lần nữa.\n"
        "Action: tool_search_law\n"
        'Action Input: {"query": "Chỉ thị số 35/CT-TTg"}'
    )
    out = _parser().parse(text)
    assert isinstance(out, AgentAction)
    assert out.tool == "tool_search_law"
    assert out.tool_input == {"query": "Chỉ thị số 35/CT-TTg"}


def test_final_answer_formulation_preamble() -> None:
    # Real capture: Gemini writes "Final Answer formulation:" before
    # the actual answer (sometimes followed by a quoted version, then
    # the unquoted version). Take the LAST Final-Answer-shaped block.
    text = (
        "Thought: blah blah English chain-of-thought.\n\n"
        'Final Answer formulation:\n'
        '"Tôi xin lỗi, hiện tại tôi không tìm thấy thông tin."\n\n'
        "Final Answer: Tôi xin lỗi, hiện tại tôi không tìm thấy thông tin."
    )
    out = _parser().parse(text)
    assert isinstance(out, AgentFinish)
    assert out.return_values["output"] == (
        "Tôi xin lỗi, hiện tại tôi không tìm thấy thông tin."
    )


def test_markdown_wrapped_final_answer() -> None:
    text = (
        "Thought: I know the answer.\n\n"
        "**Final Answer:** Đây là câu trả lời cuối cùng."
    )
    out = _parser().parse(text)
    assert isinstance(out, AgentFinish)
    assert out.return_values["output"] == "Đây là câu trả lời cuối cùng."


def test_no_final_answer_marker_falls_back_to_last_paragraph() -> None:
    # When the model omits the Final-Answer marker entirely, returning
    # the entire blob leaks chain-of-thought into the UI. Take just the
    # last paragraph instead.
    text = (
        "I was thinking about this for a while.\n"
        "First, I considered approach A.\n"
        "Then approach B.\n\n"
        "Câu trả lời ngắn gọn cho bạn."
    )
    out = _parser().parse(text)
    assert isinstance(out, AgentFinish)
    assert out.return_values["output"] == "Câu trả lời ngắn gọn cho bạn."


def test_action_input_with_trailing_thought_block() -> None:
    # Even with a SINGLE Action block, the model sometimes glues a
    # "Thought:" or extra commentary right after the JSON. Strip it.
    text = (
        "Action: tool_search_law\n"
        'Action Input: {"query": "Chỉ thị 35/CT-TTg"}Thought: maybe I should retry...'
    )
    out = _parser().parse(text)
    assert isinstance(out, AgentAction)
    assert out.tool == "tool_search_law"
    assert out.tool_input == {"query": "Chỉ thị 35/CT-TTg"}


def test_empty_input_does_not_crash() -> None:
    out = _parser().parse("")
    assert isinstance(out, AgentFinish)
    # Whatever it returns should not be the empty string masquerading
    # as a real answer — but we must NOT crash.
    assert "output" in out.return_values


def test_json_with_escaped_quotes_in_string() -> None:
    # The JSON boundary detector must not get confused by escaped
    # quotes inside string values.
    text = (
        "Action: tool_search_law\n"
        'Action Input: {"query": "she said \\"hi\\""}Trailing junk.'
    )
    out = _parser().parse(text)
    assert isinstance(out, AgentAction)
    assert out.tool_input == {"query": 'she said "hi"'}
