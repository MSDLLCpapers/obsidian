"""Parameter class definition"""

from abc import ABC, abstractmethod


class Parameter(ABC):
    """
    Base class for obsidian parameters.
    """

    def __init__(self,
                 name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    @abstractmethod
    def _validate_value(self,
                        value: int | float | str):
        pass  # pragma: no cover

    @abstractmethod
    def encode(X):
        pass  # pragma: no cover
    
    @abstractmethod
    def decode(X):
        pass  # pragma: no cover

    def save_state(self) -> dict:
        """
        Save the state of the Parameter object.

        Returns:
            dict: A dictionary containing the state of the object.
        """
        obj_dict = vars(self)
        return obj_dict

    @classmethod
    def load_state(cls,
                   obj_dict: dict):
        """
        Load the state of the Parameter object from a dictionary.

        Args:
            obj_dict (dict): A dictionary containing the state of the object.

        Returns:
            Parameter: A new instance of the Parameter class with the loaded state.
        """
        return cls(**obj_dict)
