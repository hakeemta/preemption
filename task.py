from collections import namedtuple

class Task:
    Parameters = namedtuple('Parameters', 'O C D T A')

    def __init__(self, parameters) -> None:
        self.parameters = parameters

    def __str__(self) -> str:
        return str(self.parameters)
