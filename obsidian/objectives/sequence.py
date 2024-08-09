"""Wrapper class for combining multiple objectives into a single objective."""

from .base import Objective
from .config import obj_class_dict

from torch import Tensor


class Objective_Sequence(Objective):
    """
    A composite objective that combines multiple acquisition
    objectives into a single objective, potentially single or multi-output

    Attributes:
        obj_list (list[Objective]): The list of acquisition objectives.

    Raises:
        TypeError: If the input is not a list of Objective objects.
    """

    def __init__(self,
                 obj_list: list[Objective]) -> None:
        if not isinstance(obj_list, list):
            raise TypeError('Objective_Sequence input must be a list of objectives')
        if not all(isinstance(obj, Objective) for obj in obj_list):
            raise TypeError('All elements in the list must be Objective objects')
        
        super().__init__(mo=obj_list[-1]._is_mo)

        self.obj_list = obj_list

        # None but the last objective can squeeze the multi-output dimension (if at all)
        for obj in obj_list[:-1]:
            obj._is_mo = True
    
    def __repr__(self):
        """String representation of object"""
        return f'{self.__class__.__name__} (obj_list={self.obj_list})'
    
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        """
        Evaluate the objective function(s) on the candidate set samples, X
        """
        for obj in self.obj_list:
            samples = obj.forward(samples, X=X)

        return samples
    
    def save_state(self) -> dict:
        """Saves the objective(s) to a state dictionary"""
        obj_dict = {'name': self.__class__.__name__,
                    'obj_list': [obj.save_state() for obj in self.obj_list]}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
        """Loads the objective(s) from a state dictionary"""
        new_obj_list = []
        
        for obj_dict_i in obj_dict['obj_list']:
            obj_class = obj_class_dict[obj_dict_i['name']]
            new_obj_list.append(obj_class.load_state(obj_dict_i))
 
        return cls(new_obj_list)
