"""Method pointers and config for experiment designers"""

from .design import ExpDesigner
from .advanced_design import AdvExpDesigner

designer_class_dict = {"ExpDesigner": ExpDesigner, "AdvExpDesigner": AdvExpDesigner}
