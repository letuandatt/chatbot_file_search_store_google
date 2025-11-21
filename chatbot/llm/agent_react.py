from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from chatbot.llm.prompts import AGENT_SYSTEM_PROMPT, REACT_PROMPT_TEMPLATE
from chatbot.core.history import load_session_messages
from chatbot.llm.react_safe_parser import SafeReActOutputParser


def create_agent_executor(llm, tools):
    if llm is None:
        print("[llm.agent_react] Missing llm.")
        return None
    prompt = PromptTemplate(
        template=REACT_PROMPT_TEMPLATE,
        input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"system_message": AGENT_SYSTEM_PROMPT},
    )
    agent = create_react_agent(
        llm,
        tools,
        prompt,
        output_parser=SafeReActOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)

    def _prepare_agent_input(input_dict, config):
        session_id = config.get("configurable", {}).get("session_id")
        return {"input": f"[Session: {session_id}]\n{input_dict['question']}", "chat_history": input_dict.get("chat_history", [])}

    def get_history_wrapper(session_id: str, user_id: str):
        return load_session_messages(session_id, user_id)

    agent_chain = (RunnablePassthrough() | RunnableLambda(_prepare_agent_input) | agent_executor)
    agent_with_history = RunnableWithMessageHistory(
        agent_chain,
        get_history_wrapper,
        input_messages_key="question",
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(id="user_id", annotation=str, name="User ID"),
            ConfigurableFieldSpec(id="session_id", annotation=str, name="Session ID"),
        ]
    )
    return agent_with_history
