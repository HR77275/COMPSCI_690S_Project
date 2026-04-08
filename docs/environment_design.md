# Environment Design Notes

## Part 1 MVP: progress-shaping box world

This project starts with a deliberately exploitable box-pushing grid-world because it gives us a compact, fully controllable setting for reward hacking while staying aligned with the later continuous-control task.

### Intended task

- Start with the agent to the left of a movable box
- Push the box onto the goal tile `G`
- Use `goal_reached` as the true success metric

### Deliberate reward bug

- The agent receives shaping reward whenever the box-goal Manhattan distance decreases
- Moving the box away from the goal is not symmetrically penalized beyond the generic step cost
- The map includes open space below the top corridor, so the agent can route around the box and push it backward

This creates a clean exploit path:

1. Push the box one cell closer to the goal to collect positive shaping reward.
2. Move around the box through the lower corridor.
3. Push the box one cell away from the goal.
4. Move back around and push it closer again to re-earn the same shaping reward.
5. Repeat until episode truncation, often outscoring honest completion.

### Why this is useful

The exploit is intentional and legible:

- A standard optimizer can discover the loop because it yields dense progress reward.
- The policy can achieve high return while completely failing the real task.
- We can quantify the gap between `total_reward` and `true_objective`.
- We can label progress and reversals directly from simulator state for later SAE analysis.

### Proposed experiment metrics

- Episode reward
- Goal completion rate
- Number of progress events
- Number of reversal events
- Number of exploit cycles
- Reward without completion: fraction of episodes with positive return and `goal_reached = False`

## Planned continuous extension

The second stage can mirror the same idea in a simple continuous-control task and later in MuJoCo:

- Define a true task such as moving an object to a target pose.
- Add shaping reward for positive progress toward that goal.
- Deliberately leave a loophole where an agent can maximize shaping reward through repeated partial progress without ever completing the task.

Examples:

- A simple 2D puck-pushing task where moving the puck closer to the target yields reward, but the agent can repeatedly undo and regain progress.
- A MuJoCo manipulation task where moving an object closer to the target yields dense reward, but the agent can repeatedly move it in and out of a reward-rich zone instead of placing it correctly.

## Research framing

This setup supports a simple narrative for the report:

- The reward is not the objective.
- Good optimization can still produce bad behavior.
- Environment design should make that failure mode measurable rather than accidental.
- If SAE features line up with exploit cycles, we get a clean first test of exploit detection.
