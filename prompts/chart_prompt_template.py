class ChatPromptTemplate(object):
    def __init__(self, input_variables, messages):
        self.input_variables = input_variables
        self.messages = messages

    def format(self, **kwargs):
        messages = []
        for message in self.messages:
            text = message['template'].format(**kwargs)
            messages.append({
                "role": message['role'],
                "content": text
            })
        return messages
