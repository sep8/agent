from typing import Any
from agents.schema import Callbacks


class CallbackManager(object):
    def __init__(self, callbacks: Callbacks = []) -> None:
        self.callbacks = callbacks

    def configure(self, callbacks: Callbacks = []) -> None:
        self.callbacks = [*self.callbacks, callbacks]
    
    def __call__(self, input: Any) -> Any:
        output = input
        for callback in self.callbacks:
                output = callback(output)
        return output