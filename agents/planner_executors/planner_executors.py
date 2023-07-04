
from typing import Any, Dict
from agents.planner_executors.schema import ListStepContainer



class PlanAndExecute(object):
    def __init__(self, **kwargs) -> None:
        self.planner = kwargs.get('planner')
        self.executor = kwargs.get('executor')
        self.step_container = ListStepContainer()
        self.input_key: str = "input"
        self.output_key: str = "output"
        self.input_keys = [self.input_key]
        self.output_keys = [self.output_key]

    def run(self, inputs: Dict[str, Any]) -> Any:
        plan = self.planner.plan(inputs)
        for step in plan.steps:
            _new_inputs = {
                "previous_steps": self.step_container,
                "current_step": step,
                "objective": inputs[self.input_key],
            }
            new_inputs = {**_new_inputs, **inputs}
            response = self.executor.step(new_inputs)
            self.step_container.add_step(step, response)
            print(f"*****\n\nStep: {step.value}")
            print(f"\n\nResponse: {response.response}")
        return {self.output_key: self.step_container.get_final_response()}
