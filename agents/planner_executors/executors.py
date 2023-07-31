from agents.planner_executors.schema import StepResponse
from agents.structured_chat.structured_chat import StructuredChatAgent

HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""

TASK_PREFIX = """{objective}

"""


class Executor(object):
    def __init__(self, **kwargs):
        template = HUMAN_MESSAGE_TEMPLATE
        input_variables = ["previous_steps",
                           "current_step", "agent_scratchpad"]

        include_task_in_prompt = kwargs.get('include_task_in_prompt', False)
        if include_task_in_prompt:
            input_variables.append("objective")
            template = TASK_PREFIX + template

        verbose = kwargs.get('verbose', False)
        tools = kwargs.get('tools')
        print_prompt = kwargs.get('print_prompt', False)
        self.agent = StructuredChatAgent(
            tools=tools,
            human_message_template=template,
            input_variables=input_variables,
            verbose=verbose,
            print_prompt=print_prompt
        )

    def step(
        self, inputs: dict
    ) -> StepResponse:
        """Take step."""
        response = self.agent.run(**inputs)
        return StepResponse(response=response)
