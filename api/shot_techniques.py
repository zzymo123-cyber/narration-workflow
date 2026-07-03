import json
import re
from functools import lru_cache
from pathlib import Path


TECHNIQUE_LIBRARY_PATH = Path(__file__).parent.parent / "config" / "shot_techniques.json"


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


@lru_cache(maxsize=1)
def load_technique_library() -> dict[str, dict]:
    if not TECHNIQUE_LIBRARY_PATH.exists():
        return {}
    with open(TECHNIQUE_LIBRARY_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    return {
        row["id"]: row
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }


def get_technique(technique_id: str | None) -> dict | None:
    if not technique_id:
        return None
    return load_technique_library().get(str(technique_id).strip())


def shot_technique_image_hint(shot: dict) -> str:
    technique = get_technique(shot.get("technique_id"))
    if not technique:
        return ""
    return _text(technique.get("image_prompt_hint"))


def shot_technique_storyboard_visual(shot: dict) -> str:
    technique = get_technique(shot.get("technique_id"))
    if not technique:
        return _text(shot.get("visual"))
    template = _text(technique.get("image_visual_template"))
    if template:
        return _render_template(template, shot)
    visual = _text(shot.get("visual"))
    hint = _text(technique.get("image_prompt_hint"))
    if hint:
        return f"{visual}。{hint}" if visual else hint
    return visual


def shot_technique_video_hint(shot: dict) -> str:
    technique = get_technique(shot.get("technique_id"))
    if not technique:
        return ""
    template = _text(technique.get("video_prompt_template"))
    if not template:
        return ""
    return _render_template(template, shot)


def shot_technique_video_line(shot: dict) -> str:
    technique = get_technique(shot.get("technique_id"))
    if not technique:
        return ""
    template = _text(technique.get("video_line_template"))
    if not template:
        return ""
    return _render_template(template, shot)


def technique_options_for_prompt() -> str:
    options = []
    for item in load_technique_library().values():
        options.append(f'{item["id"]}={item.get("name", item["id"])}')
    return "；".join(options)


def _template_values(shot: dict) -> dict[str, str]:
    visual = _text(shot.get("visual"))
    camera = _text(shot.get("camera"))
    subject = _text(shot.get("subject")) or _first(shot.get("characters")) or visual
    target = (
        _text(shot.get("target"))
        or _text(shot.get("technique_target"))
        or _target_from_visual(f"{visual}，{camera}")
        or subject
    )
    return {
        "duration": _duration_text(shot),
        "start": _text(shot.get("start")),
        "end": _text(shot.get("end")),
        "camera": camera,
        "visual": visual,
        "subject": subject,
        "target": target,
        "action": _text(shot.get("action")) or visual,
        "foreground": _text(shot.get("foreground")) or subject,
        "background": _text(shot.get("background")) or target,
        "frame_object": _text(shot.get("frame_object")) or "门框、窗框或环境边缘",
    }


def _render_template(template: str, shot: dict) -> str:
    values = _template_values(shot)
    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", lambda m: values.get(m.group(1), ""), template)


def _duration_text(shot: dict) -> str:
    duration = shot.get("duration")
    if duration in (None, ""):
        start = shot.get("start")
        end = shot.get("end")
        if isinstance(start, int) and isinstance(end, int):
            duration = end - start
    return f"{duration}秒" if duration not in (None, "") else ""


def _first(value) -> str:
    if isinstance(value, list) and value:
        return _text(value[0])
    return ""


def _target_from_visual(visual: str) -> str:
    patterns = (
        r"的([\u4e00-\u9fff]{1,8}?)(?:突然|缓缓|慢慢|一闪|出现|晃动|发出|靠近|落下|打开|关闭)",
        r"(?:看到|看见|对准|聚焦|注视|盯着)([\u4e00-\u9fff]{1,8})",
    )
    for pattern in patterns:
        match = re.search(pattern, visual)
        if match:
            return match.group(1)
    return ""
