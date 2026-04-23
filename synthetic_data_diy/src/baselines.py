"""
Baseline registry — loads baselines.yaml and exposes typed helpers.

Usage:
  from baselines import active_baselines, load_baselines, Baseline
  for b in active_baselines():
      print(b.label, b.strategy)
"""

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "baselines.yaml"


@dataclass(frozen=True)
class Baseline:
    label: str
    strategy: str
    active: bool = True


@functools.lru_cache(maxsize=None)
def load_baselines() -> tuple[Baseline, ...]:
    """Return all baselines from baselines.yaml (cached after first read)."""
    data = yaml.safe_load(_CONFIG_PATH.read_text())
    return tuple(Baseline(**b) for b in data["baselines"])


def active_baselines() -> list[Baseline]:
    """Return baselines with active=true, in config order."""
    return [b for b in load_baselines() if b.active]


def baseline_labels() -> set[str]:
    """Return the set of all labels defined in the config (active or not)."""
    return {b.label for b in load_baselines()}


def active_labels() -> set[str]:
    """Return the set of active labels."""
    return {b.label for b in load_baselines() if b.active}
