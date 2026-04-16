"""
Prompt template loader.

Templates are stored as YAML files in subdirectories of prompts/:
  prompts/baseline/   — one .yaml file per repair category
  prompts/corrected/  — improved prompts targeting common failure modes
  prompts/<trial>/    — any future trial directory

Each YAML file must have three keys:
  category : repair domain label (e.g. "appliance_repair")
  system   : LLM system message
  user     : user-turn instruction including the JSON output schema
"""

from pathlib import Path

import yaml

DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"
BASELINE_DIR = DEFAULT_PROMPTS_DIR / "baseline"
CORRECTED_DIR = DEFAULT_PROMPTS_DIR / "corrected"


def load_prompt_templates(prompts_dir: Path) -> list[dict]:
    """Load all prompt templates from .yaml files in prompts_dir.

    Files are loaded in alphabetical order for determinism.
    Raises FileNotFoundError if the directory or its YAML files are missing.
    Raises ValueError if a file is missing required keys.
    """
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    yaml_files = sorted(prompts_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml files found in {prompts_dir}")

    templates: list[dict] = []
    for path in yaml_files:
        with path.open() as f:
            data = yaml.safe_load(f)
        missing = [k for k in ("category", "system", "user") if k not in data]
        if missing:
            raise ValueError(f"{path.name} is missing required keys: {missing}")
        templates.append({"category": data["category"], "system": data["system"], "user": data["user"]})

    return templates
