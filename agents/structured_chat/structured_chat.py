import re
from agents.base import Agent
from agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from agents.structured_chat.output_parser import StructuredChatOutputParser
from chains.chain import Chain
from prompts.chart_prompt_template import ChatPromptTemplate


HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"

class StructuredChatAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stop = kwargs.get('stop', ['Observation:'])
        tools = kwargs.get('tools')

        self.prefix = kwargs.get('prefix', PREFIX)
        human_message_template = HUMAN_MESSAGE_TEMPLATE
        format_instructions = FORMAT_INSTRUCTIONS
        input_variables = kwargs.get('input_variables', None)
        prompt = self.create_prompt(tools, self.prefix, SUFFIX, human_message_template, format_instructions, input_variables)
        
        print_prompt = kwargs.get('print_prompt', False)
        self.chain = Chain(prompt=prompt, stop=self._stop, verbose=self.verbose, print_prompt=print_prompt)
        self.output_parse = StructuredChatOutputParser()

    def create_prompt(
        self,
        tools,
        prefix: str,
        suffix: str,
        human_message_template: str,
        format_instructions: str,
        input_variables=None
    ):
        tool_strings = []
        for tool in tools:
            args_schema = re.sub("}", "}}}}", re.sub("{", "{{{{", str(tool.args)))
            tool_strings.append(
                f"{tool.name}: {tool.description}, args: {args_schema}")
        formatted_tools = "\n".join(tool_strings)
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join(
            [prefix, formatted_tools, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]

        messages = [{
            "role": "system",
            "template": template,
        }, {
            "role": "user",
            "template": human_message_template
        }]

        prompt = ChatPromptTemplate(input_variables, messages)
        return prompt
    
    def run(self, **kwargs):
        output = self(**kwargs)
        return output[self.return_values[0]]