from abc import abstractmethod
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from schema.agent import AgentAction, AgentFinish
from schema.output_parser import BaseOutputParser
from utils import plog

class AgentOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""

class Agent(object):
    def __init__(self, **kwargs):
        self.max_iterations = kwargs.get('max_iterations', 15)
        self.tools = kwargs.get('tools', [])
        self.name_to_tool_map = self._register_tools(self.tools)

        self.verbose = kwargs.get('verbose', False)
        self.chain = kwargs.get('chain', None)

        self.observation_prefix = kwargs.get('observation_prefix', "Observation: ")
        self.llm_prefix = kwargs.get('llm_prefix', "Thought:")

        self._stop = kwargs.get('stop', [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ])

        self.output_parse: AgentOutputParser = kwargs.get('output_parse')

        self.return_values = kwargs.get('return_values', ['output'])
        self.return_intermediate_steps = kwargs.get('return_intermediate_steps', False)

    def create_prompt(self, tools):
        raise NotImplementedError
        

    def _register_tools(self, tools):
        name_to_tool_map = {}
        for tool in tools:
            name_to_tool_map[tool.name] = tool
        return name_to_tool_map

    def _construct_scratchpad(self, intermediate_steps):
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        
        agent_scratchpad = thoughts
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")

        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad


    def get_full_inputs(self, intermediate_steps, **kwargs):
        """Create the full inputs for the llm from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    def _track_steps_verbose(self, step_output: Union[dict[str, str], AgentFinish, List[Tuple[AgentAction, str]]]):
         if self.verbose:
                if 'input' in step_output:
                    plog(f"> Start: \n{step_output['input']}\n", 'bold')   
                else:
                    if isinstance(step_output, AgentFinish):
                        plog(f"> Finished: \n{step_output.return_values}\n", 'bold')
                    action, observation = step_output[0]
                    though = action.log
                    if not though.startswith(self.llm_prefix):
                        though = f"{self.llm_prefix}{though}"
                    plog(f"{though}", 'green')
                    plog(f"{self.observation_prefix}{observation}\n", 'blue')
                    if action.tool == 'Final Answer':
                        plog(f"> Finished: \n{observation}\n", 'bold')

    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        """Check if the tool is a returning tool."""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # Invalid tools won't be in the map, so we return False.
        if agent_action.tool in name_to_tool_map:
            if name_to_tool_map[agent_action.tool].return_direct:
                return AgentFinish(
                    {self.return_values[0]: observation},
                    "",
                )
        return None
    
    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
    ) -> Dict[str, Any]:
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output
    
    def run(self, **kwargs):
        raise NotImplementedError
    
    def __call__(self, **kwargs):
        intermediate_steps = []
        iterations = 0
        self._track_steps_verbose(kwargs)
        while iterations <= self.max_iterations:
            next_step_output = self._take_next_step(intermediate_steps, kwargs)
            if isinstance(next_step_output, AgentFinish):
                return self._return(next_step_output, intermediate_steps)
            self._track_steps_verbose(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(tool_return, intermediate_steps)

            intermediate_steps.extend(next_step_output)
            iterations += 1

    def _take_next_step(self, intermediate_steps, inputs) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        full_inputs = self.get_full_inputs(intermediate_steps, **inputs)
        response = self.chain.run(full_inputs)
        output = self.output_parse.parse(response)

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions = []
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for action in actions:
            if action.tool in self.name_to_tool_map:
                tool = self.name_to_tool_map[action.tool]
                observation = tool.run(action.tool_input)
                result.append((action, observation))
        return result