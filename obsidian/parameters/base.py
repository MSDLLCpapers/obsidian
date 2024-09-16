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


class IParamSpace(ABC):
    """
    Interface for parameter space classes.
    """

    def __init__(self, params: list[Parameter]):
        self.params = tuple(params)
    
    def __iter__(self):
        """Iterate over the parameters in the parameter space"""
        return iter(self.params)

    def __len__(self):
        """Number of parameters in the parameter space"""
        return len(self.params)
        
    def __repr__(self):
        """String representation of object"""
        return f"{self.__class__.__name__}(params={[p.name for p in self]})"

    def __getitem__(self, index: int | str) -> Parameter:
        """Retrieve a parameter by index"""
        if isinstance(index, str):
            index = self.X_names.index(index)
        return self.params[index]
