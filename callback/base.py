class CallbackHandler(object):
    def __init__(self, **kwargs) -> None:
        self.raise_error = kwargs.get('raise_error', False)
        self.start_callbacks = kwargs.get('start_callback', [])
        self.end_callbacks = kwargs.get('end_callback', [])
        self.error_callbacks = kwargs.get('error_callback', [])

    def on_start(self, args) -> None:
        for callback in self.start_callbacks:
            callback(args)
    def on_end(self, args) -> None:
        for callback in self.end_callbacks:
            callback(args)
    def on_error(self, args) -> None:
        if self.raise_error:
          for callback in self.error_callbacks:
              callback(args)

