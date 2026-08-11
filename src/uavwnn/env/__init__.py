"""Ambiente fiel ao problema de Lamenza (2025). SPEC-00."""

from .bridge import EpisodeRunner, GreedyPolicy, Policy, RandomPolicy
from .observation import PADDING_VALUE, StateView, build_observation
from .rewards import reward_exploratory, reward_punish
from .scenario import Scenario, ScenarioConfig, episode_seed, generate_scenario
from .state import EpisodeResult, EpisodeState, TerminationReason, TruthStateView

__all__ = [
    "EpisodeResult",
    "EpisodeRunner",
    "EpisodeState",
    "GreedyPolicy",
    "PADDING_VALUE",
    "Policy",
    "RandomPolicy",
    "Scenario",
    "ScenarioConfig",
    "StateView",
    "TerminationReason",
    "TruthStateView",
    "build_observation",
    "episode_seed",
    "generate_scenario",
    "reward_exploratory",
    "reward_punish",
]
