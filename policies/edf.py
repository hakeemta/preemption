from task import Status


def EDF(tasks, atomic=False):
    _tasks = [(i, t) for i, t in enumerate(tasks)]

    if atomic:
        for i, t in _tasks:
            if t.status == Status.RESUMING:
                return i
    
    deadlines = [(i, t.Dt) for i, t in _tasks 
            if t.is_ready()]

    ordering = sorted(deadlines, key=lambda v: v[1])
    if len(ordering) < 1:
        return None
    return ordering[0][0]  
    



