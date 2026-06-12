import json
import os
from pathlib import Path


def _pipeline_path(project_dir: Path) -> Path:
    return project_dir / "pipeline.json"


def read_pipeline(project_dir: Path) -> dict:
    path = _pipeline_path(project_dir)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_pipeline(project_dir: Path, data: dict) -> None:
    """Atomic write: write .tmp then rename"""
    path = _pipeline_path(project_dir)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def get_project_root(project_name: str) -> Path:
    p = Path(project_name)
    if p.is_absolute():
        return p
    return Path.home() / "Desktop" / "narration_studio" / project_name
