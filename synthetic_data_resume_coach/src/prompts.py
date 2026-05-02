"""Prompt template and failure mode config loaders."""

from pathlib import Path
from typing import Optional

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"
FAILURE_MODES_DIR = Path(__file__).parent / "failure_modes"


def load_resume_prompt_templates(template_name: Optional[str] = None) -> dict[str, dict]:
    """Load resume generation prompt templates from YAML files.

    Args:
        template_name: If given, load only this template. Otherwise load all.

    Returns:
        Dict mapping template_name → {system, user}.
    """
    if not PROMPTS_DIR.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {PROMPTS_DIR}")

    yaml_files = sorted(PROMPTS_DIR.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml files found in {PROMPTS_DIR}")

    templates: dict[str, dict] = {}
    for path in yaml_files:
        with path.open() as f:
            data = yaml.safe_load(f)

        name = data.get("template")
        if not name:
            raise ValueError(f"{path.name} is missing 'template' key")

        missing = [k for k in ("system", "user") if k not in data]
        if missing:
            raise ValueError(f"{path.name} is missing keys: {missing}")

        templates[name] = {"system": data["system"], "user": data["user"]}

    if template_name is not None:
        if template_name not in templates:
            available = sorted(templates.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
        return {template_name: templates[template_name]}

    return templates


def load_failure_mode_configs() -> list[dict]:
    """Load failure mode definitions from YAML files.

    Returns:
        List of failure mode config dicts, each with:
        name, label, description, threshold, flag_if, hint.
    """
    if not FAILURE_MODES_DIR.is_dir():
        raise FileNotFoundError(f"Failure modes directory not found: {FAILURE_MODES_DIR}")

    yaml_files = sorted(FAILURE_MODES_DIR.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml files found in {FAILURE_MODES_DIR}")

    configs = []
    for path in yaml_files:
        with path.open() as f:
            data = yaml.safe_load(f)

        required_keys = ("name", "label", "description", "threshold")
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"{path.name} is missing required keys: {missing}")

        configs.append(data)

    return configs
