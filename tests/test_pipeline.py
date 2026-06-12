import json
import tempfile
from pathlib import Path
from api.pipeline import write_pipeline, read_pipeline


def test_write_and_read_pipeline():
    data = {
        "project": "test_project",
        "narration_style": "third_person",
        "source_text": "test story",
        "assets": {"characters": {}, "scenes": {}, "props": {}},
        "narration_segments": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        write_pipeline(project_dir, data)
        result = read_pipeline(project_dir)
        assert result["project"] == "test_project"
        assert result["narration_style"] == "third_person"


def test_write_pipeline_atomic():
    """Verify write_pipeline uses atomic write (tmp + rename)"""
    data = {"project": "atomic_test", "narration_style": "first_person",
            "source_text": "", "assets": {"characters": {}, "scenes": {}, "props": {}},
            "narration_segments": {}}
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        write_pipeline(project_dir, data)
        assert not (project_dir / "pipeline.json.tmp").exists()
        assert (project_dir / "pipeline.json").exists()
