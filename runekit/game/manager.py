import abc
from typing import List, Union

from PySide6.QtCore import QObject, Property, Signal

from .instance import GameInstance


class GameManager(QObject):
    @abc.abstractmethod
    def get_instances(self) -> List[GameInstance]:
        """Return a list of active game instance. The instances returned should be stable (same instance for all invocation)"""
        ...

    @abc.abstractmethod
    def get_active_instance(self) -> Union[GameInstance, None]:
        ...

    instance_added = Signal(GameInstance)
    instance_removed = Signal(GameInstance)
    instance_changed = Signal()
    # Alt (Option on macOS) went down or up while the game was in front
    alt_changed = Signal(bool)
    # While set_mouse_capture(True) is on, mouse events are offered to
    # mouse_hook(kind, x, y) in screen coordinates: kind is "down", "drag"
    # or "up" for the left button and "right" for a right press. A truthy
    # return swallows the event (and the matching right release). Platforms
    # without a way to intercept input leave set_mouse_capture as a no-op.
    mouse_hook = None

    def set_mouse_capture(self, on: bool):
        pass
    instances = Property(list, get_instances, notify=instance_changed)

    def stop(self):
        """Stop the GameManager and any GameInstances"""
        pass
