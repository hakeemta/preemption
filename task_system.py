import math

from utils import *
from task import Status


class TaskSystem:
    def __init__(self) -> None:
        self.n = 0
        self.util = 0

        self.tasks = []
        self.dispatched = []
        self.completed = []

        self.t = 0
        self.dt = 0
        self.L = 1

        self.trace = []

    def add_task(self, task) -> None:
        util = task.get_util()
        if util == 0:
            return

        self.util += util
        assert(self.util <= 1.0)

        self.dt = math.gcd(self.dt, task.parameters.C, task.parameters.T, task.parameters.D)
        self.L = math.lcm(self.L, task.parameters.T)
        self.tasks.append(task)
        self.n += 1

    def load_tasks(self, tasks=None, filename=None):
        if tasks is None:
            tasks = read_tasks(filename)
        for task in tasks:
            self.add_task(task)

    def reset(self) -> None:
        self.t = 0

        for task in self.tasks:
            task.reset()

    def step(self, index) -> bool:
        for i, task in enumerate(self.tasks):
            selected = i == index
            task.step(selected)

            if selected:
                self.trace.append(i + 1)
                if task.status == Status.RESUMING:
                    self.trace[-1] *= -1

        if index == None:
            self.trace.append(0)

