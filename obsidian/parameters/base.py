"""Parameter class definition"""

from abc import ABC, abstractmethod


class Parameter(ABC):
    """
    Base class for obsidian parameters.
    """

    def __init__(self,
                 name: str):
        self.name = name

    @abstractmethod
    def __repr__(self):
        pass  # pragma: no cover

    @abstractmethod
    def _validate_value(self,
                        value: int | float | str):
        """Validate data inputs"""
        pass  # pragma: no cover

    @abstractmethod
    def set_search(self):
        """Set the search space for the parameter"""
        pass  # pragma: no cover

    @abstractmethod
    def encode(X):
        """Encode parameter to a format that can be used for training"""
        pass  # pragma: no cover
    
    @abstractmethod
    def decode(X):
        """Decode parameter from transformed space"""
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
