from hackrl.envs import Action


def scripted_goal_actions() -> list[Action]:
    """Shortest completion path from the default initial state."""

    return [
        Action.DOWN,
        Action.RIGHT,
        Action.RIGHT,
    ]


def scripted_exploit_actions(cycles: int = 5) -> list[Action]:
    """Exploit the shaping reward by repeatedly undoing and regaining progress."""

    actions: list[Action] = [
        Action.DOWN,
        Action.RIGHT,
    ]
    single_cycle = [
        Action.UP,
        Action.RIGHT,
        Action.RIGHT,
        Action.DOWN,
        Action.LEFT,
        Action.UP,
        Action.LEFT,
        Action.LEFT,
        Action.DOWN,
        Action.RIGHT,
    ]
    for _ in range(cycles):
        actions.extend(single_cycle)
    return actions
