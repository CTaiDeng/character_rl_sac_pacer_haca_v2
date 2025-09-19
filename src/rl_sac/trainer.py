"""Training loop utilities for SAC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Protocol

from .agent import SACAgent
from .replay_buffer import Transition


class Environment(Protocol):
    """Minimal environment protocol compatible with the trainer."""

    def reset(self) -> Any:
        """Reset the environment and return the initial state."""

    def step(self, action: Any) -> Transition:
        """Advance the environment using ``action`` and return a transition."""


@dataclass
class TrainerConfig:
    """High-level configuration controlling the training procedure."""

    total_steps: int = 1_000_000
    warmup_steps: int = 1_000
    batch_size: int = 256
    updates_per_step: int = 1


class Trainer:
    """Skeleton implementation of an offline training loop."""

    def __init__(
        self,
        agent: SACAgent,
        environment: Environment,
        config: TrainerConfig,
        logger: MutableMapping[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.config = config
        self.logger = logger if logger is not None else {}

    def run(self) -> None:
        """Execute the training loop.

        This placeholder outlines the warmup, interaction, and update phases of
        a typical SAC training loop. Implementations should fill in the missing
        logic for collecting transitions, updating the agent, and logging
        metrics.
        """

        raise NotImplementedError("Implement trainer execution loop.")

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        """Record ``metrics`` at a given ``step`` in the logger."""

        self.logger.setdefault(step, {}).update(metrics)


__all__ = ["Environment", "TrainerConfig", "Trainer"]
