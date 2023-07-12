
from agents.base import Agent
from agents.mrkl.output_parser import MRKLOutputParser
from agents.mrkl.prompt import PREFIX, SUFFIX, FORMAT_INSTRUCTIONS
from chains.chain import Chain
from prompts.prompt_template import PromptTemplate


class ZeroShotAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tools = kwargs.get('tools')
        self.prefix = kwargs.get('prefix', PREFIX)

        prompt = self.create_prompt(tools, self.prefix, SUFFIX, FORMAT_INSTRUCTIONS)
        
        print_prompt = kwargs.get('print_prompt', False)
        self.chain = Chain(prompt=prompt, stop=self._stop, verbose=self.verbose, print_prompt=print_prompt)

        self.output_parse = MRKLOutputParser()

    def create_prompt(self,
        tools,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables = None
    ):
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]

        prompt = PromptTemplate(input_variables, template)
        return prompt
    
    def run(self, **kwargs):
        output = self(**kwargs)
        return output[self.return_values[0]]
