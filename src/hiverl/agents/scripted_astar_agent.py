"""Non-learning baseline: every robot follows its A* waypoints."""
from __future__ import annotations

from hiverl.env.warehouse_env import WarehouseEnv
from hiverl.env.warehouse_grid import ACTION_DELTAS

WAIT_ACTION = 4
_REVERSE_LOOKUP = {v: k for k, v in ACTION_DELTAS.items()}


def scripted_actions(env: WarehouseEnv) -> dict[str, int]:
    """Discrete action that moves each robot toward its next A* waypoint."""
    actions = {}
    for robot_id, robot in env.robots.items():
        waypoint = robot.next_waypoint
        if waypoint is None:
            actions[env._agent_name(robot_id)] = WAIT_ACTION
            continue
        delta = (waypoint[0] - robot.pos[0], waypoint[1] - robot.pos[1])
        actions[env._agent_name(robot_id)] = _REVERSE_LOOKUP.get(delta, WAIT_ACTION)
    return actions
