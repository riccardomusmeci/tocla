from typing import Union

class LossMetric:
    
    def __init__(
        self,
        avg_every_n_iter: int = 5
    ) -> None:
        """loss monitor class

        Args:
            avg_every_n_iter (int, optional): how many iterations to average loss. Defaults to 5.
        """
        self.loss = 0
        self._avg_loss = None
        self.avg_every_n_iter = avg_every_n_iter
        self.iter = 0
        
    def update(
        self,
        val: float
    ):
        """updates loss monitor data structure

        Args:
            val (float): loss iteration value
        """
        self.loss += val
        self.iter += 1
        
    def reset(self):
        self.loss = 0
        self.iter = 0
        
    @property
    def avg_loss(self) -> Union[None, float]:
        """returns average loss over n iter

        Returns:
            Union[None, float]: either an average loss or a None value (at the beginning)
        """
        if self.iter == self.avg_every_n_iter:
            self._avg_loss = self.loss / self.avg_every_n_iter
            self.reset()

        return self._avg_loss