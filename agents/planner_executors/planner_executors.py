
from typing import Any, Dict, List
from agents.planner_executors.planner import BasePlanner
from agents.planner_executors.executors import BaseExecutor
from pydantic import BaseModel, Field

from agents.schema import BaseStepContainer, ListStepContainer


class PlanAndExecute(BaseModel):
    planner: BasePlanner
    executor: BaseExecutor
    step_container: BaseStepContainer = Field(default_factory=ListStepContainer)
    input_key: str = "input"
    output_key: str = "output"
    verbose: bool = False

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        plan = self.planner.plan(inputs)
        for step in plan.steps:
            _new_inputs = {
                "previous_steps": self.step_container,
                "current_step": step,
                "objective": inputs[self.input_key],
            }
            new_inputs = {**_new_inputs, **inputs}
            response = self.executor.step(
                new_inputs
            )
            if self.verbose is True:
                print(f"*****\n\nStep: {step.value}")
                print(f"\n\nResponse: {response.response}")
            self.step_container.add_step(step, response)
        return {self.output_key: self.step_container.get_final_response()}
