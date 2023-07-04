
from typing import Any, Dict, List, Optional, Union
from callbacks.custom import CallbackHandler
from models.chat_model import ChatModel
from schema.output import LLMResult
from schema.output_parser import NoOpOutputParser

default_callback = CallbackHandler()


class Chain(object):
    def __init__(self, **kwargs):
        self.llm = kwargs.get('llm', ChatModel())
        self.prompt = kwargs.get('prompt')
        self._stop = kwargs.get('stop')

        self.input_keys = self.prompt.input_variables
        self.output_key = kwargs.get('output_key', 'text')
        self.output_keys = [self.output_key]
        self.output_parser = kwargs.get('output_parser', NoOpOutputParser())

        self.return_final_only = kwargs.get('return_final_only', True)
        self.callback = kwargs.get('callback') or default_callback

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Check that all inputs are present."""
        missing_keys = set(self.input_keys).difference(inputs)
        if missing_keys:
            raise ValueError(f"Missing some input keys: {missing_keys}")
        
    def _validate_outputs(self, outputs: Dict[str, Any]) -> None:
        missing_keys = set(self.output_keys).difference(outputs)
        if missing_keys:
            raise ValueError(f"Missing some output keys: {missing_keys}")

    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        """Validate and prep inputs."""
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if len(_input_keys) != 1:
                raise ValueError(
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
            inputs = {list(_input_keys)[0]: inputs}
        self._validate_inputs(inputs)
        return inputs

    def prep_prompts(self, input_list):
        """Prepare prompts from inputs."""
        stop = self._stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k]
                               for k in self.prompt.input_variables}
            messages = self.prompt.format(**selected_inputs)
            prompts.append(messages)

        return prompts, stop
    
    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.llm.model_name}

    def _create_llm_result(self, response: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create llm result from response."""
        llm_output = self._combine_llm_outputs([res.llm_output for res in response])
        generations = [res.generations for res in response]
        llm_result = LLMResult(generations=generations, llm_output=llm_output)
        return llm_result
    
    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)
        if self.memory is not None:
            self.memory.save_context(inputs, outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}
    
    def create_outputs(self, response: List[Dict[str, Any]]):
        llm_result: LLMResult = self._create_llm_result(response)
        results = [
            # Get the text of the top generated string.
            {
                self.output_key: self.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if self.return_final_only:
            results = [{self.output_key: r[self.output_key]} for r in results]
        return results

    
    def __call__(self, inputs, return_only_outputs: bool = False):
        inputs = self.prep_inputs(inputs)
        responses = self.generate([inputs])
        outputs = self.create_outputs(responses)
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs[0], return_only_outputs
        )
        return final_outputs

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        **kwargs: Any,
    ):
        prompts, stop = self.prep_prompts(input_list)
        results = []
        for messages in prompts:
            try:
                self.callback.on_start(messages)
                response = self.llm(messages, stop=stop, **kwargs)
                self.callback.on_end(response)
                results.append(response)
            except (KeyboardInterrupt, Exception) as e:
                self.callback.on_error(e)
                raise e

        return results
