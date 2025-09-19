"""Experience replay buffer for Soft Actor-Critic (SAC).

This module defines lightweight container types for the state, action,
reward, and next-state tuples collected from the environment as well as an
abstract replay buffer interface. Concrete implementations can inherit from
:class:`BaseReplayBuffer` to provide storage and sampling logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol


@dataclass
class Transition:
    """Container representing a single transition.

    Attributes:
        state: The observed state prior to taking ``action``.
        action: The action executed by the agent.
        reward: The scalar reward received from the environment.
        next_state: The observed state after executing ``action``.
        done: Boolean flag indicating whether the episode terminated.
    """

    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


class SupportsAppend(Protocol):
    """Protocol for buffers that can store transitions.

    Implementations should append the provided :class:`Transition` to the
    underlying storage. The method signature mirrors :meth:`list.append` to
    keep the interface familiar.
    """

    def append(self, transition: Transition) -> None:
        """Append a transition to the buffer."""


class BaseReplayBuffer:
    """Base class defining the public API for replay buffers.

    Subclasses should store transitions and implement a sampling strategy.
    The default implementation maintains an in-memory list of transitions
    and exposes convenience methods that concrete buffers can reuse.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._storage: List[Transition] = []

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer.

        Args:
            transition: Transition collected from the environment.
        """

        raise NotImplementedError("Subclasses must implement `add`.")

    def sample(self, batch_size: int) -> Iterable[Transition]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Iterable of transitions sampled from the buffer.
        """

        raise NotImplementedError("Subclasses must implement `sample`.")

    def __len__(self) -> int:
        return len(self._storage)


__all__ = ["Transition", "SupportsAppend", "BaseReplayBuffer"]
