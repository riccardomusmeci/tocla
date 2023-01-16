import sys
from src.utils import TimeMonitor
from src.metrics import ClassificationMetrics
from typing import Iterator, Dict, Tuple, TextIO

class ProgressBar():
    
    def __init__(
        self,
        iterable: Iterator,
        prefix: str = " ",
        bar_filler: str = "#",
        size: int = 30,
        out: TextIO = sys.stdout
    ) -> None:
        """Custom Progress Bar

        Args:
            iterable (Iterator): iterable object
            prefix (str, optional): prefix in the progress bar. Defaults to "\t".
            bar_filler (str, optional): filler of the progress bar. Defaults to "#".
            size (int, optional): progress bar size. Defaults to 60.
            out (TextIO, optional): where to render progress bar. Defaults to sys.stdout.
        """
        
        self.iterable = iterable
        self.len_iterable = len(iterable)
        self.prefix = prefix
        self.bar_filler = bar_filler
        self.size = size
        self.out = out
        
    def show(
        self,
        epoch: int,
        index: int,
        time_monitor: TimeMonitor,
        metrics: ClassificationMetrics,
        lr: float
    ):
        """progress bar show method 

        Args:
            epoch (int): epoch
            index (int): batch index
            time_monitor (TimeMonitor): time monitor class instance with status properoty
            metrics (ClassificationMetrics): classification metrics class instance with status propery
            lr (float): learning rate
        """
        progress = int((self.size*index) / self.len_iterable)
        progress_perc = 100*(index/self.len_iterable)
        print("Epoch: {}{}{:.2f}% [{}{}] {}/{} [ {}, {}, lr={:.5f} ]".format(
            # epoch
            epoch,
            self.prefix,
            # percentage overall
            progress_perc,
            # progress bar
            self.bar_filler*progress,
            '.'*(self.size-progress),
            # batch progress
            index,
            self.len_iterable,
            # elapsed time and estimated time ( {}<{})
            time_monitor.status,
            # metrics vals
            ", ".join([f"{m}={v:.5f}" if v is not None else f"{m}=null" for m, v in metrics.status.items()]),
            # learning rate
            lr
        ), end="\r", file=self.out, flush=True)
               
    def __call__(
        self, 
        epoch: int,
        metrics: ClassificationMetrics,
        time_monitor: TimeMonitor,
        lr: float
    ) -> Tuple:
        """progress bar call method 

        Args:
            epoch (int): epoch
            time_monitor (TimeMonitor): time monitor class instance with status properoty
            metrics (ClassificationMetrics): classification metrics class instance with status propery
            lr (float): learning rate
    
        Returns:
            Tuple: batch x, target

        Yields:
            Iterator[Tuple]: batch x, target
        """
          
        self.show(
            epoch=0,
            index=0,
            time_monitor=time_monitor,
            metrics=metrics,
            lr=lr
        )
        for i, item in enumerate(self.iterable):
            yield item
            self.show(
                epoch=epoch,
                index=i+1,
                time_monitor=time_monitor,
                metrics=metrics,
                lr=lr
            )
        print("", flush=True, file=self.out)
    