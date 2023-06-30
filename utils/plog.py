class Plog(object):
    def __init__(self) -> None:
        self.bcolors = {
            'purple': '\033[95m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'ENDC': '\033[0m',
            'bold': '\033[1m',
            'underline': '\033[4m'
        }
    """Print log messages with color."""

    def __call__(self, message, color=None):
        if color is None:
            print(message)
        else:
            pcolor = self.bcolors[color] if color in self.bcolors else self.bcolors['ENDC']
            print(f"{pcolor}{message}{self.bcolors['ENDC']}")

plog = Plog()