from task import Status

def EDF(tasks, atomic=False):
    tasks = [(i, t) for i, t in enumerate(tasks)]

    if atomic:
        for i, t in tasks:
            if t.status == Status.RESUMING:
                return i

    deadlines = [(i, t.Dt) for i, t in tasks if t.is_ready()]

    deadlines = sorted(deadlines, key=lambda v: v[1])
    if len(deadlines) < 1:
        return None
    return deadlines[0][0]

