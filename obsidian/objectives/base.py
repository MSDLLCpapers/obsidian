
"""Base class for obsidian objective functions"""

from torch import Tensor
from botorch.acquisition.multi_objective.objective import MCAcquisitionObjective, MCMultiOutputObjective
from abc import abstractmethod
from obsidian.surrogates.utils import tensordict_to_dict


class Objective(MCMultiOutputObjective, MCAcquisitionObjective):
    """
    An objective function which is calculated from the model output(s) and input(s).

    Attributes:
        _is_mo (bool): A flag indicating whether the final output is multi-output.

    """

    def __init__(self,
                 mo: bool = False) -> None:

        super().__init__()
        self._is_mo = mo

    def output_shape(self,
                     samples: Tensor) -> Tensor:
        return samples.squeeze(-1) if not self._is_mo else samples

    @abstractmethod
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        
        pass  # pragma: no cover
    
    def save_state(self) -> dict:
        
        obj_dict = {'name': self.__class__.__name__,
                    'state_dict': tensordict_to_dict(self.state_dict())}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
                
        return cls(**obj_dict['state_dict'])

    def __repr__(self):
        return f'{self.__class__.__name__} (mo={self._is_mo})'
