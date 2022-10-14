from task import Status


def EDF(fifo=True):

    def choose(tasks, atomic=False):
        _tasks = [(i, t) for i, t in enumerate(tasks)]

        if not atomic:
            for i, t in _tasks:
                if t.resuming():
                    return i
        
        criteria = []
        for i, t in _tasks:
            if not t.is_ready():
                continue
            
            r = 1 if t.status == Status.RUNNING else 0
            d = (i, t.Dt, r)
            criteria.append(d)

        if fifo:
            ordering = sorted(criteria, key=lambda c: (c[1], -c[2]))
        else:
            ordering = sorted(criteria, key=lambda c: (c[1], c[0]))
        
        if len(ordering) < 1:
            return None

        return ordering[0][0]

    return choose



