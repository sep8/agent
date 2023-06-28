
from abc import abstractmethod
import re
from typing import Any, List, Optional

from pydantic import BaseModel
from agents.schema import Plan, PlanOutputParser, Callbacks, Step
from models.chat_model import ChatModel

system_prompt = (
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "to accurately complete the task. If the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "At the end of your plan, say '<END_OF_PLAN>'"
)

class PlanningOutputParser(PlanOutputParser):
    def parse(self, text: str) -> Plan:
        steps = [Step(value=v) for v in re.split("\n\s*\d+\. ", text)[1:]]
        return Plan(steps=steps)


class BasePlanner(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    @abstractmethod
    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""


class Planner(BasePlanner):
    model: ChatModel
    output_parser: PlanOutputParser = PlanningOutputParser()
    stop: Optional[List] = None
    verbose: bool = False

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""
        input = inputs["input"]
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": input}]
        if (self.verbose):
            print("planner prompt: ", messages)
        response = self.model(messages, stop=self.stop, callbacks=callbacks)
        return self.output_parser.parse(response)
