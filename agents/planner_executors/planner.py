
import re
from schema.agent import Plan, PlanOutputParser, Callbacks, Step
from chains.chain import Chain
from prompts.chart_prompt_template import ChatPromptTemplate

SYSTEM_PROMPT = (
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "to accurately complete the task. If the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "At the end of your plan, say '<END_OF_PLAN>'"
)

HUMAN_MESSAGE_TEMPLATE = "{input}"

class PlanningOutputParser(PlanOutputParser):
    def parse(self, text: str) -> Plan:
        steps = [Step(value=v) for v in re.split("\n\s*\d+\. ", text)[1:]]
        return Plan(steps=steps)


class Planner(object):
    def __init__(self, **kwargs):
        prompt = self.create_prompt()
        verbose = kwargs.get('verbose', False)
        callback = kwargs.get('callback', None)

        self.chain = Chain(prompt=prompt, stop=["<END_OF_PLAN>"], callback = callback, verbose=verbose)
        self.output_parser = kwargs.get('output_parser', PlanningOutputParser())

    def create_prompt(self, system_prompt: str = SYSTEM_PROMPT,human_message_template: str = HUMAN_MESSAGE_TEMPLATE):
        input_variables=['input']
        system_prompt_template = "\n".join(system_prompt)
        messages = [{"role": "system", "template": system_prompt_template}, {"role": "user","template": human_message_template}]
        prompt = ChatPromptTemplate(input_variables, messages)
        return prompt

    def plan(self, inputs: dict) -> Plan:
        """Given input, decide what to do."""
        response = self.chain(inputs)
        return self.output_parser.parse(response)
