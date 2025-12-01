"""Numerical integration module.

This module redirects to cbfkit.integration for numerical integration schemes.
Kept for backward compatibility.
"""

from cbfkit.integration import forward_euler, runge_kutta_4, solve_ivp

__all__ = ["forward_euler", "runge_kutta_4", "solve_ivp"]
