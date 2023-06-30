from utils import plog


class Tool(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.func = kwargs.get('func')
        self.description = kwargs.get('description')
        self.args = kwargs.get('args')

    def run(self, args, verbose=False):
        if verbose:
            plog(f"Running {self.name} with args {args}", 'cyan')
        return self.func(args)