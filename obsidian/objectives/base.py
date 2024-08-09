
"""Base class for obsidian objective functions"""

from obsidian.utils import tensordict_to_dict

from botorch.acquisition.multi_objective.objective import MCAcquisitionObjective, MCMultiOutputObjective
from abc import abstractmethod

from torch import Tensor


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
        """Converts the output to the correct Torch shape based on the multi-output flag"""
        return samples.squeeze(-1) if not self._is_mo else samples

    @abstractmethod
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        
        pass  # pragma: no cover
    
    def save_state(self) -> dict:
        """Saves the objective to a state dictionary"""
        obj_dict = {'name': self.__class__.__name__,
                    'state_dict': tensordict_to_dict(self.state_dict())}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
        """Loads the objective from a state dictionary"""
        return cls(**obj_dict['state_dict'])

    def __repr__(self):
        """String representation of object"""
        return f'{self.__class__.__name__} (mo={self._is_mo})'
