"""Environment exports for the hackrl package."""

from .gridworld import Action, BoxProgressGridWorld, Tile

__all__ = ["Action", "BoxProgressGridWorld", "Tile"]

try:
    from .lunar_lander import LunarProgressEnv, LunarStepResult
except ModuleNotFoundError:
    # Allow gridworld-only workflows to run without the optional Box2D stack.
    pass
else:
    __all__.extend(["LunarProgressEnv", "LunarStepResult"])
