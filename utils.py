import numpy as np
from task import Task


def read_tasks(filename):
    tasks = []

    with open(filename) as fp:
        n = fp.readline()
        n = int(n)

        for i in range(n):
            line = fp.readline()
            line = line.strip()
            params = [int(v) for v in line.split()]

            task = Task( Task.Parameters(*params) )
            tasks.append(task)

    return tasks


def read_traces(filename):
    traces = []

    with open(filename) as fp:
        for line in fp.readlines():
            traces += line.split()

    traces = np.array(traces, dtype=int)
    return traces


def compute_H(tasks):
    periods = [task.parameters.T for task in tasks]
    H = (periods and np.lcm.reduce(periods)) or 0
    return H


def get_ticks(task, H, for_deadlines=False, n_repeats=1):
    ticks = []
    offset = task.parameters.O
    upper = H
    if for_deadlines:
        offset += task.parameters.D
        upper += task.parameters.D

    ticks = np.array( range(offset, (upper * n_repeats), 
                            task.parameters.T ) )
    return ticks


