import difflib
from IPython.display import clear_output
from loguru import logger

def diff_strings(a: str, b: str, *, use_loguru_colors: bool = False) -> str:
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    if use_loguru_colors:
        green = '<GREEN><black>'
        red = '<RED><black>'
        endgreen = '</black></GREEN>'
        endred = '</black></RED>'
    else:
        green = '\x1b[38;5;16;48;5;2m'
        red = '\x1b[38;5;16;48;5;1m'
        endgreen = '\x1b[0m'
        endred = '\x1b[0m'

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            output.append(a[a0:a1])
        elif opcode == 'insert':
            output.append(f'{green}{b[b0:b1]}{endgreen}')
        elif opcode == 'delete':
            output.append(f'{red}{a[a0:a1]}{endred}')
        elif opcode == 'replace':
            output.append(f'{green}{b[b0:b1]}{endgreen}')
            output.append(f'{red}{a[a0:a1]}{endred}')
    return ''.join(output)

def print_diff_strings(a: str, b: str, *, use_loguru_colors: bool = True) -> None:
    diffs = diff_strings(a, b, use_loguru_colors=use_loguru_colors)
    logger.opt(raw=True, colors=True).info(diffs)

def print_clean_diff_strings(a: str, b: str, *, use_loguru_colors: bool = True) -> None:
    clear_output()
    print_diff_strings(a, b, use_loguru_colors=use_loguru_colors)