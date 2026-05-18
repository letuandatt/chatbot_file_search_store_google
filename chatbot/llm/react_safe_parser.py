import json
import re
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish


class SafeReActOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str):
        text = text.strip()

        # 1. Nếu chứa Final Answer -> trả luôn
        if "Final Answer:" in text:
            # Lấy phần sau Final Answer
            answer = text.split("Final Answer:", 1)[1].strip()
            return AgentFinish(
                return_values={"output": answer},
                log=text
            )

        # 2. Regex để bắt Action và Action Input
        # Format chuẩn: 
        # Action: tool_name
        # Action Input: input_value
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)

        if not match:
            # 3. Fallback: Nếu Bot nói lung tung không đúng format Action -> Coi như là câu trả lời
            return AgentFinish(
                return_values={"output": text},
                log=text
            )

        action = match.group(1).strip()
        action_input = match.group(2).strip()

        # 4. XỬ LÝ QUAN TRỌNG: Thử parse JSON cho Multi-argument Tool
        # Tool recall_history cần 2 tham số (user_id, session_id), nên input bắt buộc phải là Dict
        try:
            # Cố gắng parse chuỗi thành JSON Dict
            tool_input = json.loads(action_input)
        except json.JSONDecodeError:
            # Nếu không phải JSON, giữ nguyên là string (cho các tool search đơn giản)
            tool_input = action_input.strip(" ").strip('"')

        return AgentAction(tool=action, tool_input=tool_input, log=text)
