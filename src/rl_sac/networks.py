"""Neural network definitions for SAC agents.

This module declares abstract base classes for policy and value networks and
provides factory helpers that later implementations can extend. The SAC
algorithm typically employs stochastic policies and twin Q-value networks; the
interfaces exposed here accommodate those patterns without prescribing a
particular deep learning framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple


class PolicyNetwork(Protocol):
    """Protocol describing the policy network API."""

    def forward(self, state: Any) -> Tuple[Any, Any]:
        """Compute the action distribution given ``state``.

        Implementations should return a tuple containing the sampled action and
        auxiliary information (e.g., log probabilities) required by the SAC
        loss functions.
        """

    def parameters(self) -> Any:
        """Return trainable parameters for optimization."""


class QNetwork(Protocol):
    """Protocol describing the state-action value network API."""

    def forward(self, state: Any, action: Any) -> Any:
        """Estimate the Q-value for the provided ``state`` and ``action``."""

    def parameters(self) -> Any:
        """Return trainable parameters for optimization."""


@dataclass
class NetworkFactory:
    """Convenience structure bundling network construction callables."""

    policy_builder: Any
    q1_builder: Any
    q2_builder: Any

    def build_policy(self, *args: Any, **kwargs: Any) -> PolicyNetwork:
        """Instantiate a policy network using ``policy_builder``."""

        raise NotImplementedError("Override to instantiate policy network.")

    def build_q_functions(self, *args: Any, **kwargs: Any) -> Tuple[QNetwork, QNetwork]:
        """Instantiate the twin Q-networks used by SAC."""

        raise NotImplementedError("Override to instantiate Q-networks.")


__all__ = ["PolicyNetwork", "QNetwork", "NetworkFactory"]
