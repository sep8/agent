import re
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import BaseModel
from agents.agent import BaseSingleActionAgent
from agents.callback_manager import CallbackManager
from agents.schema import AgentAction, BaseTool, PlanOutputParser
from models.chat_model import ChatModel
from langchain.prompts import ChatPromptTemplate


PREFIX = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""

SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:"""

HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"


class StructuredChatAgent(object):
    def __init__(self, **kwargs):
        self.observation_prefix = kwargs.get('observation_prefix', "Observation: ")
        self.llm_prefix = kwargs.get('llm_prefix', "Thought:")

    def _init_construct_scratchpad(self, intermediate_steps):
      thoughts = ""
      for action, observation in intermediate_steps:
          thoughts += action.log
          thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"

    def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        agent_scratchpad = self._init_construct_scratchpad(intermediate_steps)
        return (
              f"This was your previous work "
              f"(but I haven't seen any of it! I only see what "
              f"you return as final answer):\n{agent_scratchpad}"
          )

    @property
    def _stop(self) -> List[str]:
        return ["Observation:"]

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX
    ):
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE
        format_instructions: str = FORMAT_INSTRUCTIONS

        tool_strings = []
        for tool in tools:
            args_schema = re.sub("}", "}}}}", re.sub("{", "{{{{", str(tool.args)))
            tool_strings.append(f"{tool.name}: {tool.description}, args: {args_schema}")

        formatted_tools = "\n".join(tool_strings)

        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, formatted_tools, format_instructions, suffix])

        def format_message(input_variables):
            human_message = human_message_template.format(**input_variables)
            messages = [
                { 'role': 'system', 'content': template },
                { 'role': 'user', 'content': human_message },
            ]
            return messages

        return format_message

    @classmethod
    def from_llm_and_tools(
        cls,
        model: ChatModel,
        tools: Sequence[BaseTool],
        output_parser: Optional[PlanOutputParser] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix
        )
        messages = prompt(input_variables)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(llm=llm)
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @property
    def _agent_type(self) -> str:
        raise ValueError("This should be implemented by subclasses.")