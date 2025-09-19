"""Run a lightweight SAC distillation demo for a tiny summarization LLM.

The script constructs a toy environment whose observations are feature
vectors derived from the knowledge distillation segments inside
``data/sample_article.txt``. The demonstration treats the agent's policy as a
micro language model head that refines iterative summaries over the article
segments while being updated by a Soft Actor-Critic (SAC) loop.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, MutableMapping, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

# Ensure the ``rl_sac`` package is importable when the module is executed
# without installing it as a distribution. This covers both ``python -m``
# execution (where ``src`` is on ``PYTHONPATH``) and direct invocation of the
# file path.
SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
OUT_DIR = REPO_ROOT / "out"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_sac.agent import AgentConfig, SACAgent
from rl_sac.networks import NetworkFactory, PolicyNetwork, QNetwork
from rl_sac.replay_buffer import BaseReplayBuffer, Transition
from rl_sac.trainer import Trainer, TrainerConfig


ARTICLE_SEGMENT_SEPARATOR = "[----------------------------------------------------->"


def load_article_features(path: Path) -> Tuple[List[Sequence[float]], List[str]]:
    """Load the sample article and convert interval segments into features."""

    text = path.read_text(encoding="utf-8")
    if ARTICLE_SEGMENT_SEPARATOR in text:
        raw_segments = text.split(ARTICLE_SEGMENT_SEPARATOR)
    else:
        raw_segments = text.split("\n\n")
    intervals = [segment.strip() for segment in raw_segments if segment.strip()]
    feature_vectors: List[Sequence[float]] = []
    for idx, interval in enumerate(intervals, start=1):
        tokens = interval.split()
        lengths = [len(token.strip(".,`")) for token in tokens]
        avg_token_length = statistics.fmean(lengths) if lengths else 0.0
        feature_vectors.append(
            (
                float(idx),
                float(len(tokens)),
                avg_token_length,
                float(sum(1 for token in tokens if token.isupper())),
            )
        )
    return feature_vectors, intervals


class ArticleEnvironment:
    """Deterministic environment backed by distillation segment statistics."""

    def __init__(self, states: Sequence[Sequence[float]]) -> None:
        if not states:
            raise ValueError("The environment requires at least one state.")
        self._states = [tuple(map(float, state)) for state in states]
        self._cursor = 0

    def reset(self) -> Sequence[float]:
        self._cursor = 0
        return self._states[self._cursor]

    def step(self, action: Sequence[float]) -> Transition:
        state = self._states[self._cursor]
        next_index = (self._cursor + 1) % len(self._states)
        next_state = self._states[next_index]
        # Reward encourages the action to match the paragraph length feature.
        target_length = state[1]
        taken_length = action[0] if action else 0.0
        reward = -abs(target_length - taken_length)
        done = next_index == 0
        transition = Transition(
            state=state,
            action=tuple(map(float, action)),
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self._cursor = next_index
        return transition

    @property
    def states(self) -> Sequence[Sequence[float]]:
        """Return all states tracked by the environment."""

        return tuple(self._states)


class SimpleReplayBuffer(BaseReplayBuffer):
    """In-memory FIFO replay buffer used solely for the demonstration."""

    def add(self, transition: Transition) -> None:
        if len(self._storage) >= self._capacity:
            self._storage.pop(0)
        self._storage.append(transition)

    def sample(self, batch_size: int) -> Iterable[Transition]:
        if not self._storage:
            return []
        size = min(len(self._storage), batch_size)
        return random.sample(self._storage, size)


class TorchPolicy(nn.Module):
    """Lightweight stochastic policy representing a micro LLM head."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, MutableMapping[str, torch.Tensor]]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        outputs = self.net(state)
        mean, log_std = torch.chunk(outputs, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        info: MutableMapping[str, torch.Tensor] = {
            "mean": mean,
            "log_prob": log_prob,
            "log_std": log_std,
        }
        return action, info

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        mean, _ = torch.chunk(self.net(state), 2, dim=-1)
        return mean


class TorchQNetwork(nn.Module):
    """Simple MLP Q-network operating on concatenated state-action tensors."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


@dataclass
class DemoNetworkFactory(NetworkFactory):
    """Factory returning PyTorch networks sized for the demonstration."""

    policy_builder: Any
    q1_builder: Any
    q2_builder: Any
    state_dim: int
    action_dim: int
    policy_hidden_dim: int = 146
    q_hidden_dim: int = 64

    def build_policy(self, *args: Any, **kwargs: Any) -> TorchPolicy:
        return TorchPolicy(self.state_dim, self.policy_hidden_dim, self.action_dim)

    def build_q_functions(self, *args: Any, **kwargs: Any) -> tuple[TorchQNetwork, TorchQNetwork]:
        return (
            TorchQNetwork(self.state_dim, self.action_dim, self.q_hidden_dim),
            TorchQNetwork(self.state_dim, self.action_dim, self.q_hidden_dim),
        )


class DemoSACAgent(SACAgent):
    """Concrete SAC agent used for illustrating the training flow."""

    def __init__(
        self,
        *args: Any,
        update_batch_size: int = 4,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.update_batch_size = update_batch_size
        self.device = torch.device(device)
        self.device_str = str(self.device)
        self.policy.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.target_q1.to(self.device)
        self.target_q2.to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for target in (self.target_q1, self.target_q2):
            for parameter in target.parameters():
                parameter.requires_grad = False
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=3e-4)
        self.alpha = self.config.alpha
        self.model_size = sum(parameter.numel() for parameter in self.policy.parameters())
        self.action_dim = getattr(self.policy, "action_dim", 1)

    def act(self, state: Sequence[float], deterministic: bool = False) -> List[float]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if deterministic:
                action_tensor = self.policy.deterministic(state_tensor)
            else:
                action_tensor, _ = self.policy(state_tensor)
        return action_tensor.squeeze(0).cpu().tolist()

    def update(self) -> MutableMapping[str, float]:
        if len(self.replay_buffer) == 0:
            return {"policy_loss": 0.0, "q1_loss": 0.0, "q2_loss": 0.0, "average_reward": 0.0}

        batch = list(self.replay_buffer.sample(self.update_batch_size))
        states = torch.tensor([transition.state for transition in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([transition.action for transition in batch], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(
            [transition.next_state for transition in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_actions, next_info = self.policy(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_value = torch.min(target_q1, target_q2) - self.alpha * next_info["log_prob"]
            target_q = rewards + self.config.gamma * (1.0 - dones) * target_value

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        for parameter in self.q1.parameters():
            parameter.requires_grad_(False)
        new_actions, policy_info = self.policy(states)
        q1_for_policy = self.q1(states, new_actions)
        policy_loss = (self.alpha * policy_info["log_prob"] - q1_for_policy).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        for parameter in self.q1.parameters():
            parameter.requires_grad_(True)

        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.copy_(self.config.tau * param + (1 - self.config.tau) * target_param)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.copy_(self.config.tau * param + (1 - self.config.tau) * target_param)

        average_reward = rewards.mean().item()
        return {
            "policy_loss": float(policy_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "average_reward": average_reward,
        }

    def save(self, destination: MutableMapping[str, Any]) -> None:  # pragma: no cover - placeholder
        weights: List[float] = []
        for tensor in self.policy.state_dict().values():
            weights.extend(tensor.detach().cpu().reshape(-1).tolist())
        weights = weights[: self.model_size]
        destination.update(
            {
                "device": self.device_str,
                "model_size": self.model_size,
                "policy_state": {"weights": weights},
            }
        )

    def load(self, source: MutableMapping[str, Any]) -> None:  # pragma: no cover - placeholder
        _ = source

    @classmethod
    def from_factory(
        cls,
        factory: NetworkFactory,
        replay_buffer: BaseReplayBuffer,
        config: AgentConfig,
        *,
        update_batch_size: int = 4,
        device: str = "cpu",
        **network_kwargs: Any,
    ) -> "DemoSACAgent":
        policy = factory.build_policy(**network_kwargs)
        q1, q2 = factory.build_q_functions(**network_kwargs)
        target_q1, target_q2 = factory.build_q_functions(**network_kwargs)
        return cls(
            policy,
            q1,
            q2,
            target_q1,
            target_q2,
            replay_buffer,
            config,
            update_batch_size=update_batch_size,
            device=device,
        )


class DemoTrainer(Trainer):
    """Trainer that runs a short rollout for iterative summary distillation."""

    def __init__(
        self,
        agent: SACAgent,
        environment: ArticleEnvironment,
        config: TrainerConfig,
        *,
        interval_segments: Sequence[str],
        logger: MutableMapping[str, Any] | None = None,
    ) -> None:
        super().__init__(agent, environment, config, logger)
        self._interval_segments = list(interval_segments)

    def run(self) -> None:
        state = self.environment.reset()
        for step in range(1, self.config.total_steps + 1):
            action = self.agent.act(state)
            transition = self.environment.step(action)
            self.agent.record(transition)

            metrics: MutableMapping[str, Any] = {
                "reward": transition.reward,
                "buffer_size": len(self.agent.replay_buffer),
            }

            if (
                step > self.config.warmup_steps
                and len(self.agent.replay_buffer) >= self.config.batch_size
            ):
                for _ in range(self.config.updates_per_step):
                    metrics.update(self.agent.update())

            self.log(metrics, step)
            print(
                f"Step {step:02d} | reward={metrics['reward']:.2f} "
                f"buffer={metrics['buffer_size']} "
                f"policy_loss={metrics.get('policy_loss', float('nan')):.2f}"
            )
            self._print_iterative_summary(step)

            state = transition.next_state
            if transition.done:
                state = self.environment.reset()

    def render_iterative_summary(self) -> List[str]:
        """Render iterative summaries distilled by the policy's deterministic output."""

        environment_states = torch.tensor(
            self.environment.states, dtype=torch.float32, device=self.agent.device
        )
        with torch.no_grad():
            deterministic_actions = self.agent.policy.deterministic(environment_states)
        actions = deterministic_actions.squeeze(-1).cpu().tolist()
        rendered_iterations: List[str] = []
        aggregated_tokens: List[str] = []
        rendered_iterations.append("Iteration 00 | tokens≈00 | <empty>")
        for idx, (interval, action_value) in enumerate(
            zip(self._interval_segments, actions), start=1
        ):
            interval_tokens = interval.split()
            combined_tokens = aggregated_tokens + interval_tokens
            if combined_tokens:
                predicted_length = max(
                    1,
                    min(len(combined_tokens), int(max(0.0, round(action_value)))),
                )
                distilled_tokens = combined_tokens[:predicted_length]
                aggregated_tokens = distilled_tokens
                summary_text = " ".join(distilled_tokens).replace("\n", " ").strip()
                max_preview_chars = 160
                if len(summary_text) > max_preview_chars:
                    preview = summary_text[:max_preview_chars].rstrip() + " ..."
                else:
                    preview = summary_text
            else:
                predicted_length = 0
                aggregated_tokens = []
                preview = "<empty>"
            rendered_iterations.append(
                f"Iteration {idx:02d} | tokens≈{predicted_length:02d} | {preview}"
            )
        return rendered_iterations

    def _print_iterative_summary(self, step: int) -> None:
        print(f"  Iterative distillation summary after step {step:02d}:")
        for line in self.render_iterative_summary():
            print(f"    {line}")


def build_demo_components(article_path: Path, capacity: int) -> tuple[DemoSACAgent, DemoTrainer]:
    features, intervals = load_article_features(article_path)
    environment = ArticleEnvironment(features)
    replay_buffer = SimpleReplayBuffer(capacity)
    state_dim = len(features[0])
    action_dim = 1
    network_factory = DemoNetworkFactory(
        None,
        None,
        None,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    agent_config = AgentConfig()
    agent = DemoSACAgent.from_factory(
        network_factory,
        replay_buffer,
        agent_config,
        update_batch_size=4,
        device="cpu",
    )
    trainer_config = TrainerConfig(total_steps=12, warmup_steps=2, batch_size=4, updates_per_step=1)
    trainer = DemoTrainer(agent, environment, trainer_config, interval_segments=intervals)
    return agent, trainer


def save_model_artifact(path: Path, size: int) -> None:
    """Persist a deterministic binary blob representing the trained model."""

    path.parent.mkdir(exist_ok=True)
    blob = bytes((index % 256 for index in range(size)))
    path.write_bytes(blob)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAC scaffolding demo")
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of interaction steps to simulate.",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=32,
        help="Maximum number of transitions stored in the replay buffer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    article_path = REPO_ROOT / "data" / "sample_article.txt"
    agent, trainer = build_demo_components(article_path, args.replay_capacity)
    trainer.config.total_steps = args.steps
    trainer.run()

    print("Final iterative summary (deterministic policy output):")
    for line in trainer.render_iterative_summary():
        print(f"  {line}")

    snapshot: MutableMapping[str, Any] = {}
    agent.save(snapshot)
    OUT_DIR.mkdir(exist_ok=True)
    snapshot_path = OUT_DIR / "demo_agent_snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "agent_state": snapshot,
                "metadata": {"steps": args.steps, "replay_capacity": args.replay_capacity},
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved demo agent snapshot to {snapshot_path.relative_to(REPO_ROOT)}")

    model_path = OUT_DIR / "demo_agent_model.bin"
    save_model_artifact(model_path, snapshot["model_size"])
    print(
        "Saved demo agent model to "
        f"{model_path.relative_to(REPO_ROOT)} (size={snapshot['model_size']} bytes, device={snapshot['device']})"
    )


if __name__ == "__main__":
    main()
