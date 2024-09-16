"""Base class for obsidian constraints"""


from abc import abstractmethod, ABC
from torch.nn import Module


class Constraint(ABC, Module):
    """
    Base class for constraints, which restrict the input or output space
    of a model or optimization problem
    """

    def __init__(self) -> None:
        super().__init__()
        return

    @abstractmethod
    def forward(self):
        pass  # pragma: no cover
