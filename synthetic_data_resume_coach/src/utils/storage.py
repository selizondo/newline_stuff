"""Storage utilities for JSONL format and data persistence."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel
import logfire

T = TypeVar("T", bound=BaseModel)


def save_jsonl(
    data: list[dict | BaseModel],
    file_path: str | Path,
    append: bool = False,
) -> Path:
    """Save data to a JSONL file.

    Args:
        data: List of dictionaries or Pydantic models to save.
        file_path: Path to the output file.
        append: If True, append to existing file instead of overwriting.

    Returns:
        Path to the saved file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(file_path, mode) as f:
        for item in data:
            if isinstance(item, BaseModel):
                line = item.model_dump_json()
            else:
                line = json.dumps(item, default=str)
            f.write(line + "\n")

    logfire.info(f"Saved {len(data)} records to {file_path}", append=append)
    return file_path


def load_jsonl(file_path: str | Path) -> list[dict]:
    """Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dictionaries loaded from the file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return []

    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    logfire.info(f"Loaded {len(data)} records from {file_path}")
    return data


def iter_jsonl(file_path: str | Path) -> Iterator[dict]:
    """Iterate over records in a JSONL file without loading all into memory.

    Args:
        file_path: Path to the JSONL file.

    Yields:
        Dictionary for each line in the file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_as_models(
    file_path: str | Path,
    model_class: type[T],
) -> list[T]:
    """Load JSONL data and parse as Pydantic models.

    Args:
        file_path: Path to the JSONL file.
        model_class: Pydantic model class to parse records into.

    Returns:
        List of parsed Pydantic model instances.
    """
    data = load_jsonl(file_path)
    return [model_class.model_validate(item) for item in data]


def save_invalid_records(
    records: list[dict],
    errors: list[dict],
    output_dir: str | Path = "data/labeled",
    filename: str = "invalid.jsonl",
) -> Path:
    """Save invalid records with their validation errors.

    Args:
        records: List of invalid record data.
        errors: List of corresponding validation errors.
        output_dir: Output directory path.
        filename: Output filename.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename

    invalid_data = []
    for record, error in zip(records, errors):
        invalid_data.append({
            "record": record,
            "errors": error,
            "labeled_at": datetime.utcnow().isoformat(),
        })

    return save_jsonl(invalid_data, file_path, append=True)


def get_timestamped_filename(prefix: str, extension: str = "jsonl") -> str:
    """Generate a timestamped filename.

    Args:
        prefix: Filename prefix.
        extension: File extension (without dot).

    Returns:
        Timestamped filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


class JSONLWriter:
    """Context manager for writing JSONL files incrementally."""

    def __init__(self, file_path: str | Path, append: bool = False):
        """Initialize the writer.

        Args:
            file_path: Path to the output file.
            append: If True, append to existing file.
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.append = append
        self._file = None
        self._count = 0

    def __enter__(self) -> "JSONLWriter":
        mode = "a" if self.append else "w"
        self._file = open(self.file_path, mode)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._file:
            self._file.close()
        logfire.info(f"Wrote {self._count} records to {self.file_path}")

    def write(self, item: dict | BaseModel) -> None:
        """Write a single record to the file.

        Args:
            item: Dictionary or Pydantic model to write.
        """
        if self._file is None:
            raise RuntimeError("Writer not initialized. Use with context manager.")

        if isinstance(item, BaseModel):
            line = item.model_dump_json()
        else:
            line = json.dumps(item, default=str)

        self._file.write(line + "\n")
        self._count += 1

    def write_batch(self, items: list[dict | BaseModel]) -> None:
        """Write multiple records to the file.

        Args:
            items: List of dictionaries or Pydantic models to write.
        """
        for item in items:
            self.write(item)
