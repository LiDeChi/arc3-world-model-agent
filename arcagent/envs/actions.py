from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActionKind(IntEnum):
    RESET = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5
    CLICK = 6
    UNDO = 7


ACTION_NAMES = {
    ActionKind.RESET: "RESET",
    ActionKind.UP: "ACTION1_UP",
    ActionKind.DOWN: "ACTION2_DOWN",
    ActionKind.LEFT: "ACTION3_LEFT",
    ActionKind.RIGHT: "ACTION4_RIGHT",
    ActionKind.INTERACT: "ACTION5_INTERACT",
    ActionKind.CLICK: "ACTION6_CLICK",
    ActionKind.UNDO: "ACTION7_UNDO",
}


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    x: int | None = None
    y: int | None = None

    def clipped(self, grid_size: int) -> "Action":
        if self.kind != ActionKind.CLICK:
            return self
        x = 0 if self.x is None else max(0, min(grid_size - 1, int(self.x)))
        y = 0 if self.y is None else max(0, min(grid_size - 1, int(self.y)))
        return Action(self.kind, x=x, y=y)


# The learner uses a compact discrete action set. CLICK defaults to center.
COMPACT_ACTIONS = [
    Action(ActionKind.RESET),
    Action(ActionKind.UP),
    Action(ActionKind.DOWN),
    Action(ActionKind.LEFT),
    Action(ActionKind.RIGHT),
    Action(ActionKind.INTERACT),
    Action(ActionKind.CLICK, x=0, y=0),
    Action(ActionKind.UNDO),
]


def legal_action_indices(include_reset: bool = False, include_undo: bool = False, include_click: bool = False) -> list[int]:
    allowed = [1, 2, 3, 4, 5]
    if include_reset:
        allowed.insert(0, 0)
    if include_click:
        allowed.append(6)
    if include_undo:
        allowed.append(7)
    return allowed


def index_to_action(index: int, grid_size: int = 16, click_xy: tuple[int, int] | None = None) -> Action:
    action = COMPACT_ACTIONS[int(index)]
    if action.kind == ActionKind.CLICK:
        if click_xy is None:
            center = grid_size // 2
            return Action(ActionKind.CLICK, x=center, y=center)
        return Action(ActionKind.CLICK, x=click_xy[0], y=click_xy[1]).clipped(grid_size)
    return action


def action_to_index(action: Action | int) -> int:
    if isinstance(action, int):
        return int(action)
    return int(action.kind)
