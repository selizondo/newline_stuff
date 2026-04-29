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


def load_answer_templates(strategy: str = "zero_shot") -> list[dict]:
    """Load answer-only templates for Ph1b (controlled comparison mode).

    System prompt comes from the strategy directory (preserving strategy flavor).
    User prompt comes from answer_only/ and contains {question}/{equipment_problem}
    placeholders that are formatted at generation time.

    Args:
        strategy: Answering strategy (zero_shot, few_shot, chain_of_thought, etc.)

    Returns:
        List of dicts with keys: category, system, user.
        The user string contains {question} and {equipment_problem} format placeholders.
    """
    strategy_dir = PROMPTS_DIR / strategy
    if not strategy_dir.is_dir():
        available = sorted(d.name for d in PROMPTS_DIR.iterdir() if d.is_dir())
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available strategies (subdirs of prompts/): {available}"
        )

    answer_only_dir = PROMPTS_DIR / "answer_only"
    if not answer_only_dir.is_dir():
        raise FileNotFoundError(f"answer_only prompt directory not found: {answer_only_dir}")

    strategy_files = {f.stem: f for f in sorted(strategy_dir.glob("*.yaml"))}
    answer_files = {f.stem: f for f in sorted(answer_only_dir.glob("*.yaml"))}

    if not answer_files:
        raise FileNotFoundError(f"No .yaml files found in {answer_only_dir}")

    templates: list[dict] = []
    for stem, answer_path in sorted(answer_files.items()):
        with answer_path.open() as f:
            answer_data = yaml.safe_load(f)

        system = ""
        if stem in strategy_files:
            with strategy_files[stem].open() as f:
                strategy_data = yaml.safe_load(f)
            system = strategy_data.get("system", "")

        if "user" not in answer_data:
            raise ValueError(f"answer_only/{answer_path.name} is missing 'user' key")
        if "category" not in answer_data:
            raise ValueError(f"answer_only/{answer_path.name} is missing 'category' key")

        templates.append({
            "category": answer_data["category"],
            "system": system,
            "user": answer_data["user"],
        })

    return templates


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
