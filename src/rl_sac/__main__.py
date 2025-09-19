"""Command-line entry point for SAC training."""

from __future__ import annotations

import argparse
from typing import Any

from .agent import AgentConfig, SACAgent
from .replay_buffer import BaseReplayBuffer
from .trainer import Trainer, TrainerConfig


def build_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for configuring the training script."""

    parser = argparse.ArgumentParser(description="Run SAC training loop")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment id")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    return parser


def build_environment(env_id: str) -> Any:
    """Construct the training environment placeholder."""

    raise NotImplementedError("Provide environment construction logic (e.g., Gym).")


def build_replay_buffer(env: Any) -> BaseReplayBuffer:
    """Create a replay buffer compatible with the environment."""

    raise NotImplementedError("Instantiate a replay buffer implementation.")


def main(args: list[str] | None = None) -> None:
    """Parse configuration and run the training loop."""

    parser = build_argument_parser()
    parsed = parser.parse_args(args=args)

    environment = build_environment(parsed.env)
    replay_buffer = build_replay_buffer(environment)

    agent_config = AgentConfig()
    trainer_config = TrainerConfig(
        total_steps=parsed.total_steps,
        warmup_steps=parsed.warmup_steps,
        batch_size=parsed.batch_size,
    )

    # Placeholders: replace with actual network factory, replay buffer, etc.
    network_factory = None  # type: ignore[assignment]
    agent = SACAgent.from_factory(network_factory, replay_buffer, agent_config)  # type: ignore[arg-type]
    trainer = Trainer(agent, environment, trainer_config)
    trainer.run()


if __name__ == "__main__":
    main()
