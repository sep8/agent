
from typing import Any, Dict, Mapping
import openai


class ChatModel(object):
    def __init__(self, model_name='gpt-3.5-turbo', **kwargs):
        self.model_name = model_name
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', None)
        self.top_p = kwargs.get('top_p', 1.0)
        self.n = kwargs.get('n', 1)

    def _create_chat_result(self, response: Mapping[str, Any]) -> Dict[str, Any]:
        generations = []
        for res in response["choices"]:
            message = res["message"]
            content = message['content'] or ""
            generations.append(content)
        llm_output = {
            "token_usage": response["usage"], 'model_name': self.model_name}
        return {"llm_output": llm_output, "generations": generations}

    def __call__(self, messages, stop):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=self.n,
            stop=stop
        )
        return self._create_chat_result(response)
