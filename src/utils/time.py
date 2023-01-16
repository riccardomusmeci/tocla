import time
import datetime
import numpy as np
from functools import wraps

STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"

def now():
    STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
    return datetime.datetime.now().strftime(STRFTIME_FORMAT)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        return result, total_time
    return timeit_wrapper

class TimeMonitor:
    """Time monitor class
    """
    def __init__(
        self,
        num_iters: int
    ) -> None:
        self.num_iters = num_iters
        self.t_iters = []
    
    def update(self, t):
        self.t_iters.append(t)
        
    def estimate(self) -> float:
        if len(self.t_iters) == 0:
            est_sec = 0
        else:
            est_sec = np.mean(self.t_iters)*self.num_iters
        return str(datetime.timedelta(seconds=est_sec)).split(".")[0]
    
    def elapsed(self):
        tot_sec = sum(self.t_iters)
        return str(datetime.timedelta(seconds=tot_sec)).split(".")[0]
    
    @property
    def status(self) -> str:
        return f"{self.elapsed()}<{self.estimate()}"
        
    
    