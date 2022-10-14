from collections import namedtuple
from enum import Enum
import math


class Status(Enum):
    IDLE = 1
    RESUMING = 2
    RUNNING = 3
    COMPLETED = 4

class Task:
    Parameters = namedtuple('Parameters', 'O C D T A')

    def __init__(self, parameters) -> None:
        self.parameters = parameters
        self.Ct = self.parameters.C
        self.At = 0
        self.Dt = self.parameters.D
        self.releases = 0
        self.status = Status.IDLE
        self.U = self.parameters.C / self.parameters.T

        self.t = 0
        self.reset()

    def get_util(self) -> float:
        return self.U

    def reset(self, start: bool = True) -> None:
        self.status = Status.IDLE
        self.At = 0
        self.update()

        if start:
            self.releases = 1
            self.t = 0

            if self.parameters.O > 0:
                self.status = Status.COMPLETED
                self.Dt = self.parameters.O
                self.Ct = 0
                self.releases = 0
        
    def is_ready(self) -> bool:
        return self.status != Status.COMPLETED

    def update(self, reload: bool = True) -> None:
        if reload:
            self.Ct = self.parameters.C
            self.Dt = self.parameters.D

    def step(self, selected: bool = False, delta: int = 1) -> None:
        if not selected:
            if self.status in [Status.RUNNING, Status.RESUMING]:
                self.At = self.parameters.A
                self.status = Status.IDLE
        elif self.At > 0:
            self.status = Status.RESUMING
            self.At -= 1
        else:
            if (not self.is_ready()) or (delta > self.Ct):
                raise Exception('Task execution overrun!')

            self.Ct -= delta
            self.status = Status.RUNNING if self.Ct > 0 else Status.COMPLETED

        self.t += delta
        self.Dt -= delta
        self.update(False)

        next_r = self.parameters.O + (self.releases * self.parameters.T)
        if self.t >= next_r:
            self.releases += 1
            self.reset(False)

    def __str__(self) -> str:
        return str(self.parameters)


