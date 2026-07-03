import math
import re

from api.duration import _chinese_char_count, dialogue_duration, narration_duration


SENTENCE_END = set("。！？!?")
QUOTE_OPEN = "“\""
QUOTE_CLOSE = "”\""
PLANNER_VERSION = 3
NON_SPEAKER_WORDS = {
    "恐惧语气", "前所未有", "语气", "声音", "电话那头", "几秒后", "下一秒",
}


def _clean_text(text: str) -> str:
    return re.sub(r"[\s“”\"]+", "", text or "")


def _speaker_from_prefix(prefix: str) -> str:
    cleaned = re.sub(r"\s+", "", prefix or "")
    match = re.search(r"([\u4e00-\u9fff]{1,4})(?:低声|沉声|吼道|说道|问道|喊道|叫道|答道|追问|提醒|骂道|说|问|喊|吼)[:：]?$", cleaned)
    if match:
        speaker = match.group(1)
        if speaker in NON_SPEAKER_WORDS or "语气" in speaker:
            return "角色"
        return speaker
    return "角色"


def _is_inline_narration_quote(source: str, sentence_start: int, quote_start: int, close: int) -> bool:
    prefix = source[sentence_start:quote_start]
    content = source[quote_start + 1:close]
    after_index = close + 1
    while after_index < len(source) and source[after_index].isspace():
        after_index += 1
    after = source[after_index] if after_index < len(source) else ""
    cleaned_prefix = re.sub(r"\s+", "", prefix or "")
    if not cleaned_prefix:
        return False
    if re.search(r"(低声|沉声|吼道|说道|问道|喊道|叫道|答道|追问|提醒|骂道|说|问|喊|吼)[:：，,]?$", cleaned_prefix):
        return False
    if any(action in cleaned_prefix[-12:] for action in ("摆了摆手", "开口", "回答", "追问", "提醒")):
        return False
    if after in "，,、；;":
        return True
    return _chinese_char_count(content) <= 4 and after not in SENTENCE_END


def _slice_id(index: int) -> str:
    return f"s{index:04d}"


def _beat_id(index: int) -> str:
    return f"v{index:04d}"


def _board_id(index: int) -> str:
    return f"b{index:04d}"


def _append_slice(items: list[dict], source: str, start: int, end: int, kind: str, speaker: str | None = None) -> None:
    raw = source[start:end]
    stripped = raw.strip()
    if not stripped:
        return
    left_trim = len(raw) - len(raw.lstrip())
    right_trim = len(raw.rstrip())
    real_start = start + left_trim
    real_end = start + right_trim
    item = {
        "slice_id": _slice_id(len(items) + 1),
        "source_start": real_start,
        "source_end": real_end,
        "text": source[real_start:real_end].strip(),
        "kind": kind,
        "speaker": speaker or ("旁白" if kind == "narration" else "角色"),
    }
    item["char_count"] = _chinese_char_count(item["text"])
    item["estimated_seconds"] = estimate_slice_seconds(item)
    items.append(item)


def slice_source_text(source: str) -> list[dict]:
    """Split source text into ordered narration/dialogue slices with source spans."""
    items: list[dict] = []
    sentence_start = 0
    i = 0
    while i < len(source):
        char = source[i]
        if char == "“":
            close = source.find("”", i + 1)
            if close == -1:
                close = i
            if _is_inline_narration_quote(source, sentence_start, i, close):
                i += 1
                continue
            prefix_start = sentence_start
            prefix_end = i
            speaker = _speaker_from_prefix(source[max(0, prefix_start):prefix_end])
            _append_slice(items, source, prefix_start, prefix_end, "narration")
            if close == i:
                pass
            else:
                _append_slice(items, source, i + 1, close, "dialogue", speaker)
            sentence_start = close + 1
            i = close + 1
            continue
        if char in SENTENCE_END:
            end = i + 1
            while end < len(source) and source[end] in QUOTE_CLOSE:
                end += 1
            _append_slice(items, source, sentence_start, end, "narration")
            sentence_start = end
            i = end
            continue
        i += 1
    _append_slice(items, source, sentence_start, len(source), "narration")
    return items


def estimate_slice_seconds(item: dict) -> int:
    if item.get("kind") == "dialogue":
        return dialogue_duration(item.get("text", ""))
    return narration_duration(item.get("text", ""))


def build_draft_voice_beats(slices: list[dict]) -> list[dict]:
    beats = []
    for item in slices:
        beat_type = "dialogue" if item.get("kind") == "dialogue" else "narration"
        for text in split_voice_text(item.get("text", ""), beat_type):
            duration = dialogue_duration(text) if beat_type == "dialogue" else narration_duration(text)
            beats.append({
                "beat_id": _beat_id(len(beats) + 1),
                "source_slice_ids": [item["slice_id"]],
                "type": beat_type,
                "speaker": item.get("speaker") or ("旁白" if beat_type == "narration" else "角色"),
                "text": text,
                "duration": duration,
                "char_count": _chinese_char_count(text),
            })
    return beats


def split_voice_text(text: str, beat_type: str, max_seconds: int = 15) -> list[str]:
    """Split a long narration/dialogue beat so every generated board can stay <= max_seconds."""
    text = (text or "").strip()
    if not text:
        return []
    max_chars = int((3 if beat_type == "dialogue" else 4.5) * max_seconds)
    if _chinese_char_count(text) <= max_chars:
        return [text]

    parts = []
    remaining = text
    split_marks = "，,；;、。！？!?"
    while _chinese_char_count(remaining) > max_chars:
        char_count = 0
        split_at = 0
        fallback_at = 0
        best_mark = 0
        for idx, ch in enumerate(remaining):
            if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
                char_count += 1
            if char_count <= max_chars:
                fallback_at = idx + 1
                if ch in split_marks and char_count >= max_chars * 0.55:
                    best_mark = idx + 1
            if char_count >= max_chars:
                split_at = best_mark or fallback_at
                break
        if split_at <= 0:
            break
        while split_at < len(remaining) and remaining[split_at] in "，,、；;":
            split_at += 1
        part = remaining[:split_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def pack_voice_beats(
    beats: list[dict],
    target_seconds: int = 13,
    max_seconds: int = 15,
) -> list[dict]:
    boards: list[dict] = []
    current: list[dict] = []
    current_duration = 0

    def flush() -> None:
        nonlocal current, current_duration
        if not current:
            return
        boards.append({
            "board_id": _board_id(len(boards) + 1),
            "voice_beat_ids": [beat["beat_id"] for beat in current],
            "source_slice_ids": [
                slice_id
                for beat in current
                for slice_id in beat.get("source_slice_ids", [])
            ],
            "duration": current_duration,
            "target_seconds": target_seconds,
            "max_seconds": max_seconds,
        })
        current = []
        current_duration = 0

    for beat in beats:
        duration = int(beat.get("duration") or 0)
        would_exceed_max = current_duration + duration > max_seconds
        target_already_met = current_duration >= target_seconds
        if current and (would_exceed_max or target_already_met):
            flush()
        if duration > max_seconds:
            beat["duration_warning"] = f"beat duration {duration}s exceeds board max {max_seconds}s"
        current.append(beat)
        current_duration += duration
        if current_duration >= max_seconds:
            flush()
    flush()
    return boards


def validate_plan_coverage(source: str, slices: list[dict]) -> list[str]:
    errors = []
    previous_end = 0
    rebuilt = []
    for item in slices:
        start = item.get("source_start")
        end = item.get("source_end")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            errors.append(f"{item.get('slice_id')} has invalid source span")
            continue
        if start < previous_end:
            errors.append(f"{item.get('slice_id')} overlaps previous slice")
        previous_end = end
        rebuilt.append(source[start:end])
    if _clean_text("".join(rebuilt)) != _clean_text(source):
        errors.append("slices do not cover source text")
    return errors


def plan_script_locally(
    source: str,
    target_seconds: int = 13,
    max_seconds: int = 15,
) -> dict:
    slices = slice_source_text(source)
    beats = build_draft_voice_beats(slices)
    boards = pack_voice_beats(beats, target_seconds=target_seconds, max_seconds=max_seconds)
    coverage_errors = validate_plan_coverage(source, slices)
    return {
        "planner_version": PLANNER_VERSION,
        "script_slices": slices,
        "voice_beats": beats,
        "board_plan": boards,
        "coverage_errors": coverage_errors,
        "stats": {
            "source_chars": _chinese_char_count(source),
            "slices": len(slices),
            "voice_beats": len(beats),
            "boards": len(boards),
            "estimated_seconds": sum(beat.get("duration", 0) for beat in beats),
            "estimated_minutes": math.ceil(sum(beat.get("duration", 0) for beat in beats) / 60),
        },
    }
