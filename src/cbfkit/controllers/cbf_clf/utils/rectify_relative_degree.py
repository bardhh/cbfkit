"""Compatibility wrapper for relative-degree rectification utilities.

This legacy module path is kept for backward compatibility with older examples
that import rectification helpers from `cbfkit.controllers.cbf_clf.utils`.
The canonical implementation lives in `cbfkit.certificates.rectifiers`.
"""

from cbfkit.certificates.rectifiers import (
    compute_function_list,
    polynomial_coefficients_from_roots,
    rectify_relative_degree,
)

__all__ = [
    "rectify_relative_degree",
    "compute_function_list",
    "polynomial_coefficients_from_roots",
]
