"""
Prompt template loader.

Templates are stored as YAML files directly in prompts/:
  prompts/appliance_repair.yaml
  prompts/electrical_repair.yaml
  ...

Each YAML file contains a top-level `category` key plus one key per strategy:

  category: appliance_repair

  zero_shot:
    system: "..."
    user:   "..."

  few_shot:
    system: "..."
    user:   "..."   # includes a worked example

  chain_of_thought:
    system: "..."
    user:   "..."   # includes explicit reasoning steps

Pass a strategy name to load_prompt_templates() to select which version to use.
"""

from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"

STRATEGIES = ("zero_shot", "few_shot", "chain_of_thought")


def load_prompt_templates(strategy: str = "zero_shot") -> list[dict]:
    """Load all prompt templates from prompts/ for the given strategy.

    Args:
        strategy: One of 'zero_shot', 'few_shot', or 'chain_of_thought'.

    Returns:
        List of dicts with keys: category, system, user.

    Raises:
        ValueError: If strategy is not recognised.
        FileNotFoundError: If the prompts directory or YAML files are missing.
        ValueError: If a YAML file is missing required keys.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of: {STRATEGIES}")

    if not PROMPTS_DIR.exists():
        raise FileNotFoundError(f"Prompts directory not found: {PROMPTS_DIR}")

    yaml_files = sorted(PROMPTS_DIR.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml files found in {PROMPTS_DIR}")

    templates: list[dict] = []
    for path in yaml_files:
        with path.open() as f:
            data = yaml.safe_load(f)

        if "category" not in data:
            raise ValueError(f"{path.name} is missing 'category' key")
        if strategy not in data:
            raise ValueError(f"{path.name} is missing strategy '{strategy}'")

        block = data[strategy]
        missing = [k for k in ("system", "user") if k not in block]
        if missing:
            raise ValueError(f"{path.name}[{strategy}] is missing keys: {missing}")

        templates.append({
            "category": data["category"],
            "system": block["system"],
            "user": block["user"],
        })

    return templates
