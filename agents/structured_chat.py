import re
import json
from chains.chain import Chain
from models.chat_model import ChatModel
from prompts.chart_prompt_template import ChatPromptTemplate
from utils import dotdict, plog


PREFIX = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""

SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:"""

HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"


class StructuredChatAgent(object):
    def __init__(self, stop=['Observation:'], max_iterations=15, **kwargs):
        self._stop = stop
        tools = kwargs.get('tools')
        self.name_to_tool_map = self._register_tools(tools)
        self.verbose = kwargs.get('verbose', False)

        human_message_template = kwargs.get('human_message_template', HUMAN_MESSAGE_TEMPLATE)
        format_instructions = FORMAT_INSTRUCTIONS
        input_variables = kwargs.get('input_variables', None)
        prompt = self.create_prompt(tools, PREFIX, SUFFIX, human_message_template, format_instructions, input_variables)
        
        print_prompt = kwargs.get('print_prompt', False)
        self.chain = Chain(prompt=prompt, stop=stop, verbose=self.verbose, print_prompt=print_prompt)

        self.max_iterations = max_iterations
        self.observation_prefix = kwargs.get(
            'observation_prefix', "Observation: ")
        self.llm_prefix = kwargs.get('llm_prefix', "Thought:")

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

    def _output_parse(self, text):
        try:
            action_match = re.search(r"```(.*?)```?", text, re.DOTALL)
            if action_match is not None:
                response = json.loads(
                    action_match.group(1).strip().replace('json\n', '\n'), strict=False)
                if isinstance(response, list):
                    # gpt turbo frequently ignores the directive to emit a single action
                    response = response[0]
                return dotdict({
                    "type": "action",
                    "tool": response["action"],
                    "tool_input": response.get("action_input", {}),
                    "state" : 'finished' if response["action"] == 'Final Answer' else 'intermediate',
                    "output": response.get("action_input", {}),
                    "log": text
                })
            else:
                return dotdict({
                    "type": "action",
                    "state": "finished",
                    "output": text,
                    "log": text
                })
        except Exception as e:
            raise Exception(f"Could not parse LLM output: {text}") from e

    def get_full_inputs(self, intermediate_steps, **kwargs):
        """Create the full inputs for the llm from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    def _track_steps_verbose(self, step_output):
         if self.verbose:
                if 'input' in step_output:
                    plog(f"> Start: \n{step_output['input']}\n", 'bold')   
                else:
                    action, observation = step_output
                    though = action.log
                    if not though.startswith(self.llm_prefix):
                        though = f"{self.llm_prefix}{though}"
                    plog(f"{though}", 'green')
                    plog(f"{self.observation_prefix}{observation}\n", 'blue')
                    if action.tool == 'Final Answer':
                        plog(f"> Finished: \n{observation}\n", 'bold')

    def run(self, **kwargs):
        intermediate_steps = []
        iterations = 0
        self._track_steps_verbose(kwargs)
        while iterations <= self.max_iterations:
            next_step_outputs = self._take_next_step(intermediate_steps, kwargs)
            next_step_output = next_step_outputs[0]
            self._track_steps_verbose(next_step_output)
            if (next_step_output[0].state == 'finished'):
                return next_step_output[1]

            intermediate_steps.extend(next_step_outputs)
            iterations += 1

    def _take_next_step(self, intermediate_steps, inputs):
        full_inputs = self.get_full_inputs(intermediate_steps, **inputs)
        response = self.chain.run(full_inputs)
        output = self._output_parse(response)

        if (output.state == 'finished'):
            return [(output, output.output)]
        actions = []
        if (output.tool is not None):
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