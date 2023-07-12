class PromptTemplate(object):
    def __init__(self, input_variables, template, role='user'):
        self.role = role
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        text = self.template.format(**kwargs)
        return [{
                "role": self.role,
                "content": text
            }]
