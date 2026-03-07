"""SpaceMouse hardware abstraction.

This module wraps pyspacemouse and provides a small, testable interface.
"""

from typing import List, Optional, Tuple

try:
    import pyspacemouse
except ImportError:  # pragma: no cover - runtime dependency
    pyspacemouse = None


class SpaceMouseState:
    """Current SpaceMouse axes/buttons."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        buttons: Optional[List[bool]] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.buttons = buttons if buttons is not None else [False, False]


class SpaceMouseHardware:
    """Low-level SpaceMouse interface."""

    def __init__(self) -> None:
        self._is_open = False
        self._device = None
        self._prev_button_states = [False, False]

    def open(self) -> bool:
        """Open SpaceMouse device connection."""
        if pyspacemouse is None:
            return False
        try:
            opened = pyspacemouse.open()
            # pyspacemouse>=1.1 returns a SpaceMouseDevice, while older
            # versions return bool and expose module-level read/close.
            self._device = opened if hasattr(opened, "read") else None
            self._is_open = bool(opened)
            return self._is_open
        except Exception:
            self._device = None
            self._is_open = False
            return False

    def close(self) -> None:
        """Close SpaceMouse device connection."""
        if not self._is_open or pyspacemouse is None:
            return
        try:
            if self._device is not None and hasattr(self._device, "close"):
                self._device.close()
            elif hasattr(pyspacemouse, "close"):
                pyspacemouse.close()
        finally:
            self._device = None
            self._is_open = False

    def read(self) -> Optional[SpaceMouseState]:
        """Read current state from device."""
        if not self._is_open or pyspacemouse is None:
            return None

        try:
            if self._device is not None and hasattr(self._device, "read"):
                raw_state = self._device.read()
            elif hasattr(pyspacemouse, "read"):
                raw_state = pyspacemouse.read()
            else:
                return None
            if raw_state is None:
                return None

            raw_buttons = (
                list(raw_state.buttons) if hasattr(raw_state, "buttons") else [False, False]
            )

            return SpaceMouseState(
                x=float(raw_state.x),
                y=float(raw_state.y),
                z=float(raw_state.z),
                roll=float(raw_state.roll),
                pitch=float(raw_state.pitch),
                yaw=float(raw_state.yaw),
                buttons=[bool(b) for b in raw_buttons],
            )
        except Exception:
            return None

    def get_button_transitions(
        self, current_buttons: List[bool]
    ) -> Tuple[List[int], List[int]]:
        """Return (pressed, released) button indexes since previous call."""
        if len(self._prev_button_states) != len(current_buttons):
            self._prev_button_states = [False] * len(current_buttons)

        pressed: List[int] = []
        released: List[int] = []

        for idx, (prev, curr) in enumerate(
            zip(self._prev_button_states, current_buttons)
        ):
            if (not prev) and curr:
                pressed.append(idx)
            elif prev and (not curr):
                released.append(idx)

        self._prev_button_states = current_buttons.copy()
        return pressed, released

    def is_open(self) -> bool:
        """Whether device is open."""
        return self._is_open
