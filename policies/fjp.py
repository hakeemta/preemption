from task import Status

def FJP(priorities):
    def choose(tasks, atomic=False):
        _tasks = [(i, t) for i, t in enumerate(tasks)]

        if not atomic:
            for i, t in _tasks:
                if t.resuming():
                    return i

        _priorities = [(i, priorities[i]) for i, t in _tasks 
                            if t.is_ready()]

        ordering = sorted(_priorities, key=lambda v: -v[1])
        if len(ordering) < 1:
            return None
        
        return ordering[0][0]
        

    return choose

