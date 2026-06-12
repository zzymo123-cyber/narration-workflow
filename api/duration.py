import math


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
