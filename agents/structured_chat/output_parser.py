import json
import re
from typing import List, Union
from agents.base import AgentOutputParser
from schema import AgentAction, AgentFinish
from schema.output import Generation

class StructuredChatOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""
        try:
            action_match = re.search(r"```(.*?)```?", text, re.DOTALL)
            if action_match is not None:
                response = json.loads(
                    action_match.group(1).strip().replace('json\n', '\n'), strict=False)
                if isinstance(response, list):
                    # gpt turbo frequently ignores the directive to emit a single action
                    response = response[0]
                if response["action"] == "Final Answer":
                    return AgentFinish({"output": response["action_input"]}, text)
                else:
                    return AgentAction(
                        response["action"], response.get("action_input", {}), text
                    )                    
            else:
               return AgentFinish({"output": text}, text)
        except Exception as e:
            raise Exception(f"Could not parse LLM output: {text}") from e
    
    def parse_result(self, result: List[Generation]):
        """Parse a list of candidate model Generations into a specific format.

        The return value is parsed from only the first Generation in the result, which
            is assumed to be the highest-likelihood Generation.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        return self.parse(result[0].text)

