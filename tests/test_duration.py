import math
from api.duration import narration_duration, dialogue_duration, voice_timeline_duration


def test_narration_duration_basic():
    # 21 chars / 4.5 = 4.67 -> ceil = 5
    assert narration_duration("她独自走在雨夜小巷里，手中握着一张泛黄的信纸。") == 5


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
