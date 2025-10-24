# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""Agent abstraction orchestrating the SAC algorithm components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

from .networks import NetworkFactory, PolicyNetwork, QNetwork
from .replay_buffer import BaseReplayBuffer, Transition


@dataclass
class AgentConfig:
    """Configuration hyper-parameters for the SAC agent."""

    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    target_entropy: Optional[float] = None
    alpha_lr: float = 1e-4
    top_p: float = 0.98
    entropy_kappa: float = 0.9


class SACAgent:
    """Soft Actor-Critic agent skeleton.

    The agent coordinates policy/value networks, optimizers, and replay buffer.
    This placeholder exposes methods that later implementations can fill in
    with the actual SAC update logic.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        q1: QNetwork,
        q2: QNetwork,
        target_q1: QNetwork,
        target_q2: QNetwork,
        replay_buffer: BaseReplayBuffer,
        config: AgentConfig,
    ) -> None:
        self.policy = policy
        self.q1 = q1
        self.q2 = q2
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        self.replay_buffer = replay_buffer
        self.config = config

    def act(self, state: Any, deterministic: bool = False) -> Any:
        """Select an action for ``state``.

        Args:
            state: Environment observation.
            deterministic: Whether to return a deterministic action.
        """

        raise NotImplementedError("Implement policy action selection.")

    def record(self, transition: Transition) -> None:
        """Store a transition in the replay buffer."""

        self.replay_buffer.add(transition)

    def update(self) -> Mapping[str, float]:
        """Perform one gradient update step.

        Returns:
            Mapping of metric names to scalar values useful for logging.
        """

        raise NotImplementedError("Implement SAC update step.")

    @classmethod
    def from_factory(
        cls,
        factory: NetworkFactory,
        replay_buffer: BaseReplayBuffer,
        config: AgentConfig,
        **network_kwargs: Any,
    ) -> "SACAgent":
        """Instantiate a SAC agent using :class:`NetworkFactory` helpers."""

        policy = factory.build_policy(**network_kwargs)
        q1, q2 = factory.build_q_functions(**network_kwargs)
        target_q1, target_q2 = factory.build_q_functions(**network_kwargs)
        return cls(policy, q1, q2, target_q1, target_q2, replay_buffer, config)

    def save(self, destination: MutableMapping[str, Any]) -> None:
        """Serialize model parameters into ``destination``."""

        raise NotImplementedError("Implement serialization logic.")

    def load(self, source: Mapping[str, Any]) -> None:
        """Load model parameters from ``source``."""

        raise NotImplementedError("Implement deserialization logic.")


__all__ = ["AgentConfig", "SACAgent"]
