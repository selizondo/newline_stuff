"""
Prompt template loader.

Templates are stored per strategy under prompts/<strategy>/:

  prompts/zero_shot/appliance_repair.yaml
  prompts/zero_shot/electrical_repair.yaml
  ...
  prompts/few_shot/appliance_repair.yaml
  ...

Each YAML file contains three keys:

  category: appliance_repair
  system: "..."
  user:   "..."

To add a new strategy, create a new subdirectory under prompts/ and populate it
with one YAML file per category. No code changes required.
"""

from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt_templates(strategy: str = "zero_shot") -> list[dict]:
    """Load all prompt templates for the given strategy.

    Args:
        strategy: Name of a subdirectory under prompts/ (e.g. 'zero_shot',
                  'few_shot', 'chain_of_thought', 'human_feedback').

    Returns:
        List of dicts with keys: category, system, user.

    Raises:
        ValueError: If strategy directory does not exist.
        FileNotFoundError: If no YAML files are found in the strategy directory.
        ValueError: If a YAML file is missing required keys.
    """
    strategy_dir = PROMPTS_DIR / strategy
    if not strategy_dir.is_dir():
        available = sorted(d.name for d in PROMPTS_DIR.iterdir() if d.is_dir())
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available strategies (subdirs of prompts/): {available}"
        )

    yaml_files = sorted(strategy_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml files found in {strategy_dir}")

    templates: list[dict] = []
    for path in yaml_files:
        with path.open() as f:
            data = yaml.safe_load(f)

        if "category" not in data:
            raise ValueError(f"{path.name} is missing 'category' key")

        missing = [k for k in ("system", "user") if k not in data]
        if missing:
            raise ValueError(f"{strategy}/{path.name} is missing keys: {missing}")

        templates.append({
            "category": data["category"],
            "system": data["system"],
            "user": data["user"],
        })

    return templates
