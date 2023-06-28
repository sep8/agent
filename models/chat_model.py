
import openai

def default_callback(response):
    return response['choices'][0]['message']['content'].strip()

class ChatModel(object):
    def __init__(self, model='gpt-3.5-turbo', temperature=0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = kwargs.get('max_tokens', 2000)
        self.top_p = kwargs.get('top_p', 1.0)
        self.n = kwargs.get('n', 1)

    def __call__(self, messages, stop, callbacks = None):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=2000,
            top_p=1.0,
            n=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        response = response['choices'][0]['message']['content'].strip()
        if (callbacks is not None):
            for callback in self.callbacks:
                response = callback(response)
        return response
