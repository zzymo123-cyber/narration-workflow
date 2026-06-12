# Narration-Driven Comic Workstation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a narration-driven comic narration workstation for Douyin, cloned from manhua-workflow but with a completely new data model centered on voice_timeline + shot_timeline dual-timeline board_pages.

**Architecture:** FastAPI backend with pipeline.json file storage. Story input → LLM decomposition → narration segments with board_pages → asset production → storyboard image → Seedance video. Reuses Vidu/Wetoken API clients from manhua-workflow. No TTS, no video_parts, no prompt_seed.

**Tech Stack:** Python 3.10+, FastAPI, httpx, anthropic/openai SDK (via ideaLAB), Vidu API (image gen), Wetoken/Seedance API (video gen)

---

## File Structure

```
C:\Users\boomer\Desktop\narration-workflow\
  main.py                     # FastAPI app
  api/
    __init__.py
    pipeline.py                # pipeline.json read/write + helpers
    duration.py                # voice/shot duration calculation
    validation.py              # board_page validation rules
    prompts.py                  # prompt assembly (storyboard + video)
    decomposition.py           # LLM decomposition system prompt + logic
    llm.py                     # LLM client (copied from manhua-workflow)
    vidu.py                    # Vidu client (copied from manhua-workflow)
    wetoken.py                 # Wetoken client (copied from manhua-workflow)
    poller.py                  # poller adapted for new pipeline
    routes/
      __init__.py
      project.py               # /api/project/* (create, status, decompose)
      assets.py                # /api/asset-file (file serving)
      tasks.py                 # /api/tasks/* (submit storyboard/video)
      settings.py              # /api/settings (copied from manhua-workflow)
  static/
    index.html                  # frontend (minimal stub for now)
  tests/
    __init__.py
    test_duration.py
    test_validation.py
    test_prompts.py
    test_pipeline.py
  requirements.txt
  settings.example.json
  .gitignore
```

---

### Task 1: Project scaffold

**Files:**
- Create: `C:\Users\boomer\Desktop\narration-workflow\` (entire directory)

- [ ] **Step 1: Clone manhua-workflow and strip**

```bash
cp -r "C:\Users\boomer\Desktop\manhua-workflow" "C:\Users\boomer\Desktop\narration-workflow"
cd "C:\Users\boomer\Desktop\narration-workflow"
rm -rf .git
rm -f board_s01_01.jpg board_s01_03_v2_p01.jpg char_*.jpg scene_*.jpg
rm -f api/routes/chat.py api/routes/prompts.py api/parser.py api/poller.py
rm -f api/routes/project.py api/routes/tasks.py
rm -rf tests
rm -f .impeccable/design.json .impeccable/live/config.json
rm -rf .impeccable
rm -f .pytest_cache
rm -rf __pycache__ api/__pycache__ api/routes/__pycache__
```

- [ ] **Step 2: Initialize new git repo**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git init
git add -A
git commit -m "chore: clone from manhua-workflow as starting point"
```

- [ ] **Step 3: Create new directory structure**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: Copy reusable files and verify they exist**

Verify these files exist and are untouched:
- `api/vidu.py` — Vidu image generation client
- `api/wetoken.py` — Wetoken video generation client
- `api/llm.py` — LLM API client
- `api/routes/settings.py` — settings management
- `api/routes/assets.py` — asset file serving
- `requirements.txt`
- `settings.example.json`

Run: `ls "C:\Users\boomer\Desktop\narration-workflow\api\vidu.py" "C:\Users\boomer\Desktop\narration-workflow\api\wetoken.py" "C:\Users\boomer\Desktop\narration-workflow\api\llm.py"`
Expected: all three files listed

- [ ] **Step 5: Commit scaffold**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git add -A
git commit -m "chore: scaffold narration-workflow project"
```

---

### Task 2: Pipeline data model (pipeline.py)

**Files:**
- Create: `api/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for pipeline read/write**

```python
# tests/test_pipeline.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.pipeline'`

- [ ] **Step 3: Write api/pipeline.py**

```python
# api/pipeline.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git add api/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline.json read/write with atomic writes"
```

---

### Task 3: Duration calculation (duration.py)

**Files:**
- Create: `api/duration.py`
- Create: `tests/test_duration.py`

- [ ] **Step 1: Write failing tests for duration calculation**

```python
# tests/test_duration.py
import math
from api.duration import narration_duration, dialogue_duration, voice_timeline_duration


def test_narration_duration_basic():
    # 18 chars / 4.5 = 4.0 -> ceil = 4
    assert narration_duration("她独自走在雨夜小巷里，手中握着一张泛黄的信纸。") == 4


def test_narration_duration_short():
    # "她抬头" = 3 chars / 4.5 = 0.67 -> ceil = 1, but min is 1
    assert narration_duration("她抬头") == 1


def test_narration_duration_single_char():
    assert narration_duration("她") == 1


def test_dialogue_duration_basic():
    # "你怎么来了？" = 6 chars / 3 = 2.0 -> ceil = 2
    assert dialogue_duration("你怎么来了？") == 2


def test_dialogue_duration_short():
    # "来" = 1 char / 3 = 0.33 -> ceil = 1, min 1
    assert dialogue_duration("来") == 1


def test_voice_timeline_duration():
    beats = [
        {"type": "narration", "text": "她独自走在雨夜小巷里，手中握着一张泛黄的信纸。"},
        {"type": "dialogue", "text": "你怎么来了？"},
        {"type": "narration", "text": "她抬起头，看见了他。"},
    ]
    total = voice_timeline_duration(beats)
    assert total == 9  # 4 + 2 + 3


def test_chinese_char_count_excludes_punctuation():
    """Verify we count Chinese characters, not bytes"""
    assert narration_duration("你好世界") == 1  # 4 chars / 4.5 < 1, ceil = 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_duration.py -v`
Expected: FAIL

- [ ] **Step 3: Write api/duration.py**

```python
# api/duration.py
import math
import re


def _chinese_char_count(text: str) -> int:
    """Count Chinese characters in text (CJK Unified Ideographs range)."""
    return sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf')


def narration_duration(text: str) -> int:
    """Calculate narration duration in integer seconds.
    ceil(chinese_chars / 4.5), minimum 1 second."""
    count = _chinese_char_count(text)
    if count == 0:
        return 1
    return max(1, math.ceil(count / 4.5))


def dialogue_duration(text: str) -> int:
    """Calculate dialogue duration in integer seconds.
    ceil(chinese_chars / 3), minimum 1 second."""
    count = _chinese_char_count(text)
    if count == 0:
        return 1
    return max(1, math.ceil(count / 3))


def voice_timeline_duration(beats: list[dict]) -> int:
    """Calculate total voice timeline duration."""
    total = 0
    for beat in beats:
        if beat["type"] == "narration":
            total += narration_duration(beat["text"])
        elif beat["type"] == "dialogue":
            total += dialogue_duration(beat["text"])
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_duration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git add api/duration.py tests/test_duration.py
git commit -m "feat: add narration/dialogue duration calculation with integer seconds"
```

---

### Task 4: Board page validation (validation.py)

**Files:**
- Create: `api/validation.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing tests for validation**

```python
# tests/test_validation.py
import pytest
from api.validation import validate_board_page


def _valid_board_page():
    """Return a valid board_page fixture matching the spec example."""
    return {
        "board_id": "seg_01_01_p01",
        "page": 1,
        "total_pages": 1,
        "compact_page": False,
        "voice_duration": 9,
        "visual_duration": 10,
        "board_duration": 10,
        "video_goal": "表现雨夜小巷中，林雪独自进入危险空间，阿明突然出现，关系带有悬念。",
        "voice_timeline": [
            {"beat_id": "v01", "type": "narration", "text": "她独自走在雨夜小巷里，手中握着一张泛黄的信纸。", "speaker": "旁白", "start": 0, "end": 4, "duration": 4},
            {"beat_id": "v02", "type": "dialogue", "text": "你怎么来了？", "speaker": "阿明", "start": 4, "end": 6, "duration": 2},
            {"beat_id": "v03", "type": "narration", "text": "她抬起头，看见了他。", "speaker": "旁白", "start": 6, "end": 9, "duration": 3},
        ],
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 2, "duration": 2, "voice_refs": ["v01"], "visual": "雨夜小巷全景", "camera": "wide_establishing", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "sync", "purpose": "建立氛围", "audio_behavior": "narration_sync", "continuity_from_previous": None, "transition_type": None},
            {"shot_id": "s02", "start": 2, "end": 4, "duration": 2, "voice_refs": ["v01"], "visual": "林雪低头", "camera": "medium_close", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "sync", "purpose": "展示角色", "audio_behavior": "narration_over", "continuity_from_previous": "林雪仍在小巷", "transition_type": "continuous"},
            {"shot_id": "s03", "start": 4, "end": 5, "duration": 1, "voice_refs": [], "visual": "巷口人影", "camera": "medium", "characters": ["阿明"], "scene": "雨夜小巷", "match_strategy": "foreshadow", "purpose": "制造悬念", "audio_behavior": "sound_lead_in", "continuity_from_previous": "切向巷口", "transition_type": "cut"},
            {"shot_id": "s04", "start": 5, "end": 6, "duration": 1, "voice_refs": ["v02"], "visual": "林雪僵住", "camera": "medium", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "reaction_first", "purpose": "先拍反应", "audio_behavior": "dialogue_offscreen", "continuity_from_previous": "脚步声停", "transition_type": "continuous"},
            {"shot_id": "s05", "start": 6, "end": 9, "duration": 3, "voice_refs": ["v03"], "visual": "林雪抬头", "camera": "close_up", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "reaction_first", "purpose": "情绪转变", "audio_behavior": "narration_over", "continuity_from_previous": "从僵住到抬头", "transition_type": "continuous"},
            {"shot_id": "s06", "start": 9, "end": 10, "duration": 1, "voice_refs": [], "visual": "两人对视", "camera": "two_shot", "characters": ["林雪", "阿明"], "scene": "雨夜小巷", "match_strategy": "emotional_landing", "purpose": "情感落点", "audio_behavior": "ambient_only", "continuity_from_previous": "同框延续", "transition_type": "continuous"},
        ],
        "storyboard_image": {"status": "needed", "prompt": "", "task_id": None, "url": None, "local_path": None},
        "video": {"status": "needed", "duration": 10, "prompt": "", "task_id": None, "url": None, "local_path": None},
        "asset_refs": {"characters": ["林雪", "阿明"], "scene": "雨夜小巷", "props": []},
    }


def test_valid_board_page_passes():
    errors = validate_board_page(_valid_board_page())
    assert errors == []


def test_board_duration_exceeds_15():
    page = _valid_board_page()
    page["board_duration"] = 16
    page["shot_timeline"][-1]["end"] = 16
    page["shot_timeline"][-1]["duration"] = 7
    page["visual_duration"] = 16
    errors = validate_board_page(page)
    assert any("board_duration" in e and "15" in e for e in errors)


def test_voice_duration_exceeds_board_duration():
    page = _valid_board_page()
    page["voice_duration"] = 11
    errors = validate_board_page(page)
    assert any("voice_duration" in e for e in errors)


def test_shot_timeline_gap():
    page = _valid_board_page()
    page["shot_timeline"][2]["start"] = 5  # gap: s02 ends at 4, s03 starts at 5
    page["shot_timeline"][2]["duration"] = 0
    errors = validate_board_page(page)
    assert any("gap" in e.lower() or "间隙" in e for e in errors)


def test_shot_timeline_overlap():
    page = _valid_board_page()
    page["shot_timeline"][1]["end"] = 3  # s01 ends at 2 but s02 ends at 3, then s03 starts at 3
    page["shot_timeline"][2]["start"] = 3
    page["shot_timeline"][2]["duration"] = 2
    errors = validate_board_page(page)
    assert len(errors) > 0


def test_insufficient_shots():
    page = _valid_board_page()
    # Remove shots to get below 5
    page["shot_timeline"] = page["shot_timeline"][:4]
    errors = validate_board_page(page)
    assert any("shot" in e.lower() or "镜头" in e for e in errors)


def test_compact_page_allows_3_4_shots():
    page = _valid_board_page()
    page["compact_page"] = True
    page["shot_timeline"] = page["shot_timeline"][:4]
    page["board_duration"] = 6
    page["visual_duration"] = 6
    page["shot_timeline"][-1]["end"] = 6
    page["shot_timeline"][-1]["duration"] = 2
    page["video"]["duration"] = 6
    errors = validate_board_page(page)
    shot_errors = [e for e in errors if "shot" in e.lower() or "镜头" in e]
    assert shot_errors == []


def test_orphan_beat():
    page = _valid_board_page()
    page["voice_timeline"].append({"beat_id": "v04", "type": "narration", "text": "无引用", "speaker": "旁白", "start": 9, "end": 11, "duration": 2})
    errors = validate_board_page(page)
    assert any("v04" in e or "孤立" in e for e in errors)


def test_invalid_voice_ref():
    page = _valid_board_page()
    page["shot_timeline"][0]["voice_refs"] = ["v99"]
    errors = validate_board_page(page)
    assert any("v99" in e or "voice_refs" in e for e in errors)


def test_non_integer_time():
    page = _valid_board_page()
    page["voice_timeline"][0]["duration"] = 4.5
    errors = validate_board_page(page)
    assert any("integer" in e.lower() or "整数" in e for e in errors)


def test_empty_video_goal():
    page = _valid_board_page()
    page["video_goal"] = ""
    errors = validate_board_page(page)
    assert any("video_goal" in e for e in errors)


def test_invalid_audio_behavior():
    page = _valid_board_page()
    page["shot_timeline"][0]["audio_behavior"] = "bgm"
    errors = validate_board_page(page)
    assert any("audio_behavior" in e for e in errors)


def test_missing_shot_field():
    page = _valid_board_page()
    del page["shot_timeline"][0]["purpose"]
    errors = validate_board_page(page)
    assert any("purpose" in e for e in errors)


def test_first_shot_continuity_not_null():
    page = _valid_board_page()
    page["shot_timeline"][0]["continuity_from_previous"] = "should be null"
    errors = validate_board_page(page)
    assert any("continuity_from_previous" in e for e in errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_validation.py -v`
Expected: FAIL

- [ ] **Step 3: Write api/validation.py**

```python
# api/validation.py
from typing import Optional

VALID_AUDIO_BEHAVIORS = {
    "narration_sync", "narration_over", "dialogue_sync", "dialogue_offscreen",
    "ambient_only", "sound_lead_in", "dramatic_silence", "ambient_transition",
}

VALID_MATCH_STRATEGIES = {
    "sync", "supplement", "contrast", "foreshadow", "reaction_first",
    "reveal", "emotional_landing", "transition",
}

REQUIRED_SHOT_FIELDS = {
    "shot_id", "start", "end", "duration", "voice_refs", "visual",
    "camera", "characters", "scene", "match_strategy", "purpose",
    "audio_behavior", "continuity_from_previous", "transition_type",
}


def validate_board_page(page: dict) -> list[str]:
    errors = []

    # Integer seconds
    for i, beat in enumerate(page.get("voice_timeline", [])):
        for field in ("start", "end", "duration"):
            if not isinstance(beat.get(field), int):
                errors.append(f"voice_timeline[{i}].{field} must be integer, got {beat.get(field)}")
    for i, shot in enumerate(page.get("shot_timeline", [])):
        for field in ("start", "end", "duration"):
            if not isinstance(shot.get(field), int):
                errors.append(f"shot_timeline[{i}].{field} must be integer, got {shot.get(field)}")
    for field in ("voice_duration", "visual_duration", "board_duration"):
        if not isinstance(page.get(field), int):
            errors.append(f"{field} must be integer, got {page.get(field)}")

    # Boundary
    if page.get("board_duration", 0) > 15:
        errors.append(f"board_duration {page.get('board_duration')} exceeds 15")
    if page.get("voice_duration", 0) > page.get("board_duration", 0):
        errors.append(f"voice_duration {page.get('voice_duration')} exceeds board_duration {page.get('board_duration')}")

    # video_goal
    if not page.get("video_goal"):
        errors.append("video_goal must not be empty")

    # Shot count
    shot_count = len(page.get("shot_timeline", []))
    compact = page.get("compact_page", False)
    if not compact and shot_count < 5:
        errors.append(f"Regular page must have 5-6 shots, got {shot_count}")
    if compact and shot_count < 3:
        errors.append(f"Compact page must have 3-4 shots, got {shot_count}")
    if shot_count > 6:
        errors.append(f"Page must not exceed 6 shots, got {shot_count}")

    # Shot coverage
    shots = page.get("shot_timeline", [])
    if shots:
        if shots[0].get("start") != 0:
            errors.append(f"First shot start must be 0, got {shots[0].get('start')}")
        if shots[-1].get("end") != page.get("board_duration"):
            errors.append(f"Last shot end must equal board_duration {page.get('board_duration')}, got {shots[-1].get('end')}")
        for i in range(len(shots) - 1):
            if shots[i].get("end") != shots[i + 1].get("start"):
                errors.append(f"Shot gap/overlap: shot_timeline[{i}].end={shots[i].get('end')} != shot_timeline[{i+1}].start={shots[i+1].get('start')}")
        for i, shot in enumerate(shots):
            if shot.get("duration") is not None and shot.get("start") is not None and shot.get("end") is not None:
                if shot["duration"] != shot["end"] - shot["start"]:
                    errors.append(f"shot_timeline[{i}].duration {shot['duration']} != end-start {shot['end']-shot['start']}")

    # voice_refs references
    beat_ids = {b.get("beat_id") for b in page.get("voice_timeline", [])}
    referenced_beats = set()
    for i, shot in enumerate(shots):
        for ref in shot.get("voice_refs", []):
            if ref not in beat_ids:
                errors.append(f"shot_timeline[{i}].voice_refs contains unknown beat_id '{ref}'")
            else:
                referenced_beats.add(ref)

    # Orphan beats
    for beat in page.get("voice_timeline", []):
        if beat.get("beat_id") not in referenced_beats:
            errors.append(f"Orphan beat: beat_id '{beat.get('beat_id')}' not referenced by any shot")

    # Shot field completeness
    for i, shot in enumerate(shots):
        missing = REQUIRED_SHOT_FIELDS - set(shot.keys())
        if missing:
            errors.append(f"shot_timeline[{i}] missing fields: {missing}")
        if shot.get("audio_behavior") not in VALID_AUDIO_BEHAVIORS:
            errors.append(f"shot_timeline[{i}].audio_behavior '{shot.get('audio_behavior')}' is not valid")
        if shot.get("match_strategy") not in VALID_MATCH_STRATEGIES:
            errors.append(f"shot_timeline[{i}].match_strategy '{shot.get('match_strategy')}' is not valid")

    # First shot constraints
    if shots:
        if shots[0].get("continuity_from_previous") is not None:
            errors.append("First shot continuity_from_previous must be null")
        if shots[0].get("transition_type") is not None:
            errors.append("First shot transition_type must be null")

    return errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_validation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git add api/validation.py tests/test_validation.py
git commit -m "feat: add board_page validation with all spec rules"
```

---

### Task 5: Prompt assembly (prompts.py)

**Files:**
- Create: `api/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests for prompt assembly**

```python
# tests/test_prompts.py
from api.prompts import (
    audio_behavior_text,
    match_strategy_text,
    assemble_storyboard_prompt,
    assemble_video_prompt,
)


def test_audio_behavior_text_narration_sync():
    assert "旁白与当前画面同步" in audio_behavior_text("narration_sync", [])


def test_audio_behavior_text_dialogue_offscreen_with_speaker():
    result = audio_behavior_text("dialogue_offscreen", ["v01"],
                                  voice_timeline=[{"beat_id": "v01", "speaker": "阿明", "text": "你怎么来了？"}])
    assert "阿明" in result
    assert "画外响起" in result


def test_match_strategy_text_sync():
    assert "直接呈现" in match_strategy_text("sync")


def test_match_strategy_text_foreshadow():
    assert "暗示尚未发生" in match_strategy_text("foreshadow")


def test_assemble_storyboard_prompt():
    board_page = {
        "board_id": "seg_01_01_p01",
        "board_duration": 10,
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 2, "camera": "wide", "visual": "雨夜小巷全景"},
        ],
    }
    prompt = assemble_storyboard_prompt(board_page)
    assert "导演分镜板" in prompt
    assert "总时长：10s" in prompt
    assert "镜头s01" in prompt
    assert "0-2s" in prompt
    assert "竖屏9:16" in prompt
    assert "不要完整旁白" in prompt


def test_assemble_video_prompt():
    board_page = {
        "board_id": "seg_01_01_p01",
        "board_duration": 10,
        "video_goal": "表现雨夜相遇的悬念氛围",
        "voice_timeline": [
            {"beat_id": "v01", "type": "narration", "speaker": "旁白", "text": "她走在小巷里。", "start": 0, "end": 2},
        ],
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 2, "camera": "wide", "visual": "雨夜小巷全景",
             "match_strategy": "sync", "purpose": "建立氛围",
             "audio_behavior": "narration_sync", "voice_refs": ["v01"],
             "continuity_from_previous": None},
        ],
        "asset_refs": {"characters": ["林雪"], "scene": "雨夜小巷", "props": []},
    }
    prompt = assemble_video_prompt(board_page, asset_urls={"characters": {"林雪": "/chars/lin.png"}},
                                    storyboard_image_path="/boards/seg_01_01_p01.png")
    assert "表现雨夜相遇的悬念氛围" in prompt
    assert "旁白：她走在小巷里。" in prompt
    assert "0-2s" in prompt
    assert "角色设定板" in prompt
    assert "不要生成字幕" in prompt
    assert "不要生成 BGM" in prompt


def test_video_prompt_excludes_missing_asset_urls():
    board_page = {
        "board_id": "seg_01_01_p01",
        "board_duration": 5,
        "video_goal": "测试目标",
        "voice_timeline": [],
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 5, "camera": "wide", "visual": "画面",
             "match_strategy": "sync", "purpose": "测试",
             "audio_behavior": "ambient_only", "voice_refs": [],
             "continuity_from_previous": None},
        ],
        "asset_refs": {"characters": ["林雪"], "scene": "雨夜小巷", "props": []},
    }
    prompt = assemble_video_prompt(board_page, asset_urls={}, storyboard_image_path="/board.png")
    # Should not mention character or scene refs if no URLs
    assert "角色设定板 林雪" not in prompt
    assert "场景参考板" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_prompts.py -v`
Expected: FAIL

- [ ] **Step 3: Write api/prompts.py**

```python
# api/prompts.py
from typing import Optional

AUDIO_BEHAVIOR_MAP = {
    "narration_sync": "旁白与当前画面同步，画面直接承载旁白内容",
    "narration_over": "旁白覆盖在画面上，画面可以补充、反差或伏笔，不要机械复述旁白",
    "dialogue_sync": "角色台词与当前画面同步，可以看到说话角色或其明确动作",
    "dialogue_offscreen": "角色台词画外响起，说话者不立即露面，先拍听到台词后的反应",
    "ambient_only": "只有环境声，没有旁白和台词",
    "sound_lead_in": "非语言声音先行，用来引出下一镜头或人物",
    "dramatic_silence": "戏剧性静默，用于震惊反应、冲突后停顿、压迫感",
    "ambient_transition": "环境声作为转场桥",
}

MATCH_STRATEGY_MAP = {
    "sync": "画面直接呈现旁白或台词描述的内容",
    "supplement": "画面补充旁白或台词未说出的信息",
    "contrast": "画面与旁白或台词形成反差",
    "foreshadow": "画面暗示尚未发生的事件",
    "reaction_first": "先拍角色反应，再揭示说话者或信息来源",
    "reveal": "画面揭示之前铺垫的信息",
    "emotional_landing": "画面提供情感落点，让观众消化情绪",
    "transition": "画面作为场景或情绪的过渡桥",
}

PROHIBITIONS = """\
- 不要生成字幕
- 不要生成额外文字
- 不要生成 BGM、背景音乐、配乐、氛围音乐
- 只允许旁白、角色台词、环境声和必要动作音效
- 无旁白无台词的镜头，只允许环境声或指定音效，不要用音乐填充
- 不要改变角色长相、发型、服装
- 不要让人物、道具、空间位置在镜头之间突然跳变
- 分镜板参考图只作为镜头顺序、构图关系和人物位置参考，不要把分镜板上的编号、时间码、说明文字生成进视频画面"""


def audio_behavior_text(behavior: str, voice_refs: list[str],
                         voice_timeline: Optional[list[dict]] = None) -> str:
    text = AUDIO_BEHAVIOR_MAP.get(behavior, behavior)
    if behavior == "dialogue_offscreen" and voice_refs and voice_timeline:
        beat_map = {b["beat_id"]: b for b in voice_timeline}
        speakers = []
        for ref in voice_refs:
            beat = beat_map.get(ref)
            if beat and beat.get("speaker"):
                speakers.append(beat["speaker"])
        if speakers:
            return f"{','.join(speakers)}台词画外响起，说话者不立即露面，先拍听到台词后的反应"
    return text


def match_strategy_text(strategy: str) -> str:
    return MATCH_STRATEGY_MAP.get(strategy, strategy)


def assemble_storyboard_prompt(board_page: dict) -> str:
    lines = ["导演分镜板", f"总时长：{board_page['board_duration']}s", ""]
    for shot in board_page.get("shot_timeline", []):
        lines.append(f"镜头{shot['shot_id']} | {shot['start']}-{shot['end']}s | {shot['camera']}")
        lines.append(shot["visual"])
        lines.append("")
    lines.extend([
        "风格：黑白铅笔线稿/导演分镜稿",
        f"版面：竖屏9:16，{len(board_page.get('shot_timeline', []))}格连续分镜，竖屏规整排版",
        "每格包含镜头编号、整数秒时间码、简短画面说明",
        "不要完整旁白",
        "不要完整台词",
        "不要大量中文小字",
        "保持角色、场景、道具一致",
        "每格画面要清楚表达镜头内容、角色位置和情绪变化",
    ])
    return "\n".join(lines)


def assemble_video_prompt(board_page: dict, asset_urls: Optional[dict] = None,
                           storyboard_image_path: Optional[str] = None) -> str:
    asset_urls = asset_urls or {}
    sections = []

    # Video goal
    sections.append(f"【视频目标】\n{board_page.get('video_goal', '')}")

    # Voice timeline
    voice_lines = []
    for beat in board_page.get("voice_timeline", []):
        voice_lines.append(f"  {beat['start']}-{beat['end']}s {beat['speaker']}：{beat['text']}")
    if voice_lines:
        sections.append("【声音时间轴】\n" + "\n".join(voice_lines))

    # Shot timeline
    shot_lines = []
    for shot in board_page.get("shot_timeline", []):
        shot_lines.append(f"  {shot['start']}-{shot['end']}s 镜头{shot['shot_id']} {shot['camera']}")
        shot_lines.append(f"  画面：{shot['visual']}")
        shot_lines.append(f"  声音设计：{audio_behavior_text(shot['audio_behavior'], shot.get('voice_refs', []), board_page.get('voice_timeline'))}")
        shot_lines.append(f"  画面匹配：{match_strategy_text(shot['match_strategy'])}")
        shot_lines.append(f"  意图：{shot['purpose']}")
        if shot.get("voice_refs"):
            beat_map = {b["beat_id"]: b for b in board_page.get("voice_timeline", [])}
            refs = []
            for ref in shot["voice_refs"]:
                beat = beat_map.get(ref)
                if beat:
                    refs.append(f"{beat['speaker']}：{beat['text']}")
            if refs:
                shot_lines.append(f"  引用声音：{'; '.join(refs)}")
        if shot.get("continuity_from_previous"):
            shot_lines.append(f"  视觉延续：{shot['continuity_from_previous']}")
    if shot_lines:
        sections.append("【镜头执行】\n" + "\n".join(shot_lines))

    # Reference images
    ref_lines = []
    char_urls = asset_urls.get("characters", {})
    asset_refs = board_page.get("asset_refs", {})
    for char in asset_refs.get("characters", []):
        url = char_urls.get(char)
        if url:
            ref_lines.append(f"  角色设定板 {char}：{url}")
    if storyboard_image_path:
        ref_lines.append(f"  分镜板参考图：{storyboard_image_path}")
    scene_url = asset_urls.get("scene")
    if scene_url:
        ref_lines.append(f"  场景参考板：{scene_url}")
    prop_urls = asset_urls.get("props", {})
    for prop in asset_refs.get("props", []):
        url = prop_urls.get(prop)
        if url:
            ref_lines.append(f"  道具参考 {prop}：{url}")
    if ref_lines:
        sections.append("【参考图使用】\n" + "\n".join(ref_lines))

    # Prohibitions
    sections.append(f"【禁止项】\n{PROHIBITIONS}")

    return "\n\n".join(sections)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git add api/prompts.py tests/test_prompts.py
git commit -m "feat: add prompt assembly with audio_behavior and match_strategy text mapping"
```

---

### Task 6: LLM decomposition (decomposition.py)

**Files:**
- Create: `api/decomposition.py`
- Create: `tests/test_decomposition.py`

- [ ] **Step 1: Write failing test for decomposition system prompt**

```python
# tests/test_decomposition.py
import json
from api.decomposition import build_decomposition_system_prompt, parse_decomposition_response


def test_build_decomposition_system_prompt_contains_key_elements():
    prompt = build_decomposition_system_prompt("third_person")
    assert "旁白" in prompt
    assert "voice_timeline" in prompt
    assert "shot_timeline" in prompt
    assert "4.5" in prompt  # narration speed
    assert "3" in prompt  # dialogue speed
    assert "整数" in prompt  # integer seconds
    assert "video_goal" in prompt
    assert "match_strategy" in prompt
    assert "audio_behavior" in prompt


def test_parse_decomposition_response_valid():
    response_json = {
        "narration_style": "third_person",
        "assets": {
            "characters": {"林雪": {"seed": "年轻女性"}},
            "scenes": {"雨夜小巷": {"seed": "昏暗小巷"}},
            "props": {},
        },
        "narration_segments": {
            "seg_01_01": {
                "episode": 1,
                "segment_index": 1,
                "characters_in_segment": ["林雪"],
                "scene_location": "雨夜小巷",
                "boards": [{
                    "board_id": "seg_01_01_p01",
                    "page": 1,
                    "total_pages": 1,
                    "compact_page": False,
                    "voice_duration": 4,
                    "visual_duration": 5,
                    "board_duration": 5,
                    "video_goal": "建立雨夜氛围",
                    "voice_timeline": [
                        {"beat_id": "v01", "type": "narration", "text": "她走在小巷里。", "speaker": "旁白", "start": 0, "end": 4, "duration": 4},
                    ],
                    "shot_timeline": [
                        {"shot_id": "s01", "start": 0, "end": 4, "duration": 4, "voice_refs": ["v01"], "visual": "雨夜小巷", "camera": "wide", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "sync", "purpose": "建立氛围", "audio_behavior": "narration_sync", "continuity_from_previous": None, "transition_type": None},
                        {"shot_id": "s02", "start": 4, "end": 5, "duration": 1, "voice_refs": [], "visual": "林雪独行", "camera": "medium", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "emotional_landing", "purpose": "落点", "audio_behavior": "ambient_only", "continuity_from_previous": "林雪仍在走", "transition_type": "continuous"},
                    ],
                    "storyboard_image": {"status": "needed", "prompt": "", "task_id": None, "url": None, "local_path": None},
                    "video": {"status": "needed", "duration": 5, "prompt": "", "task_id": None, "url": None, "local_path": None},
                    "asset_refs": {"characters": ["林雪"], "scene": "雨夜小巷", "props": []},
                }],
            },
        },
    }
    result = parse_decomposition_response(response_json)
    assert "seg_01_01" in result["narration_segments"]
    assert result["narration_segments"]["seg_01_01"]["boards"][0]["voice_timeline"][0]["speaker"] == "旁白"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_decomposition.py -v`
Expected: FAIL

- [ ] **Step 3: Write api/decomposition.py**

```python
# api/decomposition.py
import json

DECOMPOSITION_SYSTEM_PROMPT = """你是解说漫剧本拆解专家。你的任务是将故事/大纲/想法拆解为旁白分镜数据结构。

## 输入
用户提供的自由文本（故事、大纲、或一个想法）。

## 旁白风格
{style_instruction}

## 输出格式
严格输出 JSON，结构如下：

```json
{{
  "narration_style": "third_person 或 first_person",
  "assets": {{
    "characters": {{ "角色名": {{ "seed": "外貌描述" }} }},
    "scenes": {{ "场景名": {{ "seed": "场景描述" }} }},
    "props": {{ "道具名": {{ "seed": "道具描述" }} }}
  }},
  "narration_segments": {{
    "seg_EP_SEG": {{
      "episode": 1,
      "segment_index": 1,
      "characters_in_segment": ["角色名"],
      "scene_location": "场景名",
      "boards": [BOARD_PAGE]
    }}
  }}
}}
```

## BOARD_PAGE 格式

每个 board_page 是一个 Seedance 视频生成单位，最长 15 秒：

```json
{{
  "board_id": "seg_EP_SEG_p01",
  "page": 1,
  "total_pages": 1,
  "compact_page": false,
  "voice_duration": INTEGER,
  "visual_duration": INTEGER,
  "board_duration": INTEGER,
  "video_goal": "本页视频的戏剧/情绪/叙事目标",
  "voice_timeline": [BEAT],
  "shot_timeline": [SHOT],
  "storyboard_image": {{ "status": "needed", "prompt": "", "task_id": null, "url": null, "local_path": null }},
  "video": {{ "status": "needed", "duration": INTEGER, "prompt": "", "task_id": null, "url": null, "local_path": null }},
  "asset_refs": {{ "characters": [], "scene": "", "props": [] }}
}}
```

## BEAT 格式

```json
{{
  "beat_id": "v01",
  "type": "narration 或 dialogue",
  "text": "旁白或台词文本",
  "speaker": "旁白（narration）或角色名（dialogue）",
  "start": INTEGER,
  "end": INTEGER,
  "duration": INTEGER
}}
```

时长计算（整数秒）：
- 旁白：ceil(中文字数 / 4.5)，下限 1 秒
- 对白：ceil(中文字数 / 3)，下限 1 秒
- 超过 8 秒的 beat 建议拆分

## SHOT 格式

```json
{{
  "shot_id": "s01",
  "start": INTEGER,
  "end": INTEGER,
  "duration": INTEGER,
  "voice_refs": ["beat_id"],
  "visual": "画面描述",
  "camera": "镜头类型",
  "characters": ["角色名"],
  "scene": "场景名",
  "match_strategy": "见下方枚举",
  "purpose": "中文自然语言描述镜头意图",
  "audio_behavior": "见下方枚举",
  "continuity_from_previous": "中文描述或 null（首个镜头必须 null）",
  "transition_type": "cut/match_cut/dissolve/continuous 或 null（首个镜头必须 null）"
}}
```

## match_strategy 枚举
- sync: 画面直接呈现旁白或台词内容
- supplement: 画面补充未说出的信息
- contrast: 画面与旁白/台词形成反差
- foreshadow: 画面暗示尚未发生的事件
- reaction_first: 先拍角色反应，再揭示说话者
- reveal: 画面揭示之前铺垫的信息
- emotional_landing: 画面提供情感落点
- transition: 画面作为场景/情绪过渡

## audio_behavior 枚举
- narration_sync: 旁白与画面同步
- narration_over: 旁白覆盖画面，画面可补充/反差
- dialogue_sync: 台词与画面同步
- dialogue_offscreen: 台词画外响起，说话者不露面
- ambient_only: 只有环境声
- sound_lead_in: 非语言声音先行
- dramatic_silence: 戏剧性静默
- ambient_transition: 环境声转场

## 硬规则
1. 所有时间字段必须是整数，不允许小数
2. board_duration <= 15
3. 常规页 5-6 个 shot，短尾页（compact_page=true）3-4 个
4. shot_timeline 必须完整覆盖 0 到 board_duration，无间隙无重叠
5. voice_timeline 和 shot_timeline 不是一一绑定
6. 允许 voice_refs 为空的镜头（伏笔、反应、情感落点、转场）
7. 第一个 shot 的 continuity_from_previous 和 transition_type 必须为 null
8. 禁止生成 BGM/背景音乐，只允许旁白、台词、环境声、动作音效
9. video_goal 不能为空
10. purpose 使用中文自然语言，不要用英文枚举"""


STYLE_INSTRUCTIONS = {
    "third_person": "使用第三人称旁白，旁白者speaker固定为"旁白"。角色台词保持原话，speaker为角色名。",
    "first_person": "使用第一人称旁白，旁白者为主角内心独白，speaker为角色名。其他角色台词speaker为角色名。",
}


def build_decomposition_system_prompt(style: str = "third_person") -> str:
    style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["third_person"])
    return DECOMPOSITION_SYSTEM_PROMPT.format(style_instruction=style_instruction)


def parse_decomposition_response(response: dict) -> dict:
    """Parse and return the decomposition response as pipeline.json structure.
    Adds project-level fields that the LLM doesn't generate."""
    return {
        "project": "",
        "narration_style": response.get("narration_style", "third_person"),
        "source_text": "",
        "assets": response.get("assets", {"characters": {}, "scenes": {}, "props": {}}),
        "narration_segments": response.get("narration_segments", {}),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/test_decomposition.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git add api/decomposition.py tests/test_decomposition.py
git commit -m "feat: add LLM decomposition system prompt and response parser"
```

---

### Task 7: API routes (project + tasks + poller + main.py)

**Files:**
- Create: `api/routes/project.py`
- Create: `api/routes/tasks.py`
- Modify: `main.py`
- Create: `api/poller.py`

This task wires everything together into a running FastAPI app. Key routes:
- `POST /api/project/create` — create project with source text
- `POST /api/project/decompose` — LLM decomposition
- `GET /api/project/status` — read pipeline status
- `POST /api/tasks/submit-storyboard` — submit storyboard image to Vidu
- `POST /api/tasks/submit-video` — submit video to Wetoken/Seedance
- Poller: adapted from manhua-workflow for new pipeline structure

- [ ] **Step 1: Write api/routes/project.py**

Create project, decompose, and status routes. Use `decomposition.py` for LLM call, `pipeline.py` for read/write, `validation.py` for checking, `prompts.py` for prompt assembly.

- [ ] **Step 2: Write api/routes/tasks.py**

Storyboard image submission (Vidu) and video submission (Wetoken/Seedance). Use `prompts.py` for assembling prompts, `pipeline.py` for state management.

- [ ] **Step 3: Write api/poller.py**

Adapted from manhua-workflow poller. Scans all projects, polls Vidu/Wetoken task status, updates pipeline.json on completion, downloads images/videos.

- [ ] **Step 4: Rewrite main.py**

FastAPI app with all routes mounted, poller started in lifespan, static file serving.

- [ ] **Step 5: Run all tests**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Start server and verify health**

Run: `cd "C:\Users\boomer\Desktop\narration-workflow" && python main.py`
Expected: Server starts on port 8002, `/api/health` returns `{"ok": true}`

- [ ] **Step 7: Commit**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git add -A
git commit -m "feat: add API routes for project, tasks, and poller"
```

---

### Task 8: End-to-end integration verification

- [ ] **Step 1: Create a test project via API**

Use curl or httpie to POST to `/api/project/create` with a story text.

- [ ] **Step 2: Run LLM decomposition**

POST to `/api/project/decompose` and verify the response contains valid narration_segments with voice_timeline + shot_timeline.

- [ ] **Step 3: Validate decomposition output**

Run validation on the decomposition output. All board_pages should pass.

- [ ] **Step 4: Verify prompt assembly**

Check that storyboard_image prompt and video prompt are correctly assembled from the decomposition data.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\boomer\Desktop\narration-workflow"
git add -A
git commit -m "feat: verify end-to-end integration"
```
