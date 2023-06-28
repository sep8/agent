
from abc import abstractmethod
from typing import Any, List
from pydantic import BaseModel
from agents.schema import BaseTool, Callbacks, StepResponse
from models.chat_model import ChatModel

HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""

TASK_PREFIX = """{objective}

"""

class BaseExecutor(BaseModel):
    @abstractmethod
    def step(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""


class Executor(BaseExecutor):
    model: ChatModel
    tools: List[BaseTool]
    input_variables = ["previous_steps", "current_step", "agent_scratchpad"]
    template = HUMAN_MESSAGE_TEMPLATE

    def step(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""
        input = inputs["input"]
        messages = [{"role": "user", "content": input}]
        response = self.model(messages, callbacks=callbacks)
        return StepResponse(response=response)