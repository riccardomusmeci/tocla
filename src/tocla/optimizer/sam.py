"""Implementation of Sharpness-Aware Minimization (SAM) optimizer from:

https://github.com/davda54/sam.
"""
from typing import Dict, Iterable

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer.

    Args:
        params (Iterable): model params
        base_optimizer (Optimizer): base optimizer
        rho (float, optional): SAM rho. Defaults to 0.05.
        adaptive (bool, optional): SAM adaptive. Defaults to False.
        bn_to_zero (bool, optional): SAM bn_to_zero. Defaults to True.
    """

    def __init__(  # type: ignore
        self,
        params: Iterable,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        bn_to_zero: bool = True,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)  # type: ignore
        self.param_groups = self.base_optimizer.param_groups
        self.bn_to_zero = bn_to_zero
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False) -> None:  # type: ignore
        """First step of SAM optimizer.

        Args:
            zero_grad (bool, optional): whether to zero grad. Defaults to False.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False) -> None:  # type: ignore
        """Second step of SAM optimizer.

        Args:
            zero_grad (bool, optional): whether to zero grad. Defaults to False.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore
        """Step of SAM optimizer.

        Args:
            closure (optional): closure. Defaults to None.
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict.

        Args:
            state_dict (Dict): state dict
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
