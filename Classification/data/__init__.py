"""
Data package for ERA5 processing and labeling.

This file marks the directory as a Python package so that relative imports like
`from .label import data_label_dict` resolve correctly in editors (e.g., Pylance)
and at runtime when executed as a module.
"""

from .label import data_label_dict

__all__ = ["data_label_dict"]
