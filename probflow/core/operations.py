"""Distribution operations for composing distributions.

This module is imported by types.py to avoid circular imports.
"""

# The SumDist, ProductDist, and JointDist classes are defined in types.py
# This module exists to support the import structure.
from .types import SumDist, ProductDist, JointDist

__all__ = ["SumDist", "ProductDist", "JointDist"]
