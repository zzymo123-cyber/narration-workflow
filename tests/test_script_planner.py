from api.script_planner import (
    build_draft_voice_beats,
    estimate_slice_seconds,
    pack_voice_beats,
    plan_script_locally,
    split_voice_text,
    slice_source_text,
    validate_plan_coverage,
)


def test_slice_source_text_preserves_order_and_coverage():
    source = "沈砚看着玄墨。\n秦越吼道：“立刻装箱送来！”他挂断电话。"

    slices = slice_source_text(source)
    errors = validate_plan_coverage(source, slices)

    assert errors == []
    assert [s["text"] for s in slices] == [
        "沈砚看着玄墨。",
        "秦越吼道：",
        "立刻装箱送来！",
        "他挂断电话。",
    ]
    assert [s["kind"] for s in slices] == ["narration", "narration", "dialogue", "narration"]
    assert slices[2]["speaker"] == "秦越"


def test_slice_source_text_keeps_inline_quoted_terms_in_narration():
    source = "谁也没料到，这条“长虫”，最近染上了一个让我无法理解的新怪癖。"

    slices = slice_source_text(source)
    errors = validate_plan_coverage(source, slices)

    assert errors == []
    assert [s["text"] for s in slices] == [source]
    assert [s["kind"] for s in slices] == ["narration"]


def test_slice_source_text_keeps_inline_sound_effect_with_narration():
    source = "地板上传来极其细微的摩擦声，“嘶啦，嘶啦”，如同春蚕啃食桑叶。"

    slices = slice_source_text(source)
    errors = validate_plan_coverage(source, slices)

    assert errors == []
    assert [s["text"] for s in slices] == [source]
    assert [s["kind"] for s in slices] == ["narration"]


def test_split_voice_text_does_not_leave_leading_punctuation():
    text = "这是很长的一段旁白，" * 12 + "最后完整收束。"

    parts = split_voice_text(text, "narration", max_seconds=4)

    assert len(parts) > 1
    assert all(not part.startswith(("，", ",", "、", "；", ";")) for part in parts)


def test_estimate_slice_seconds_uses_dialogue_and_narration_rates():
    narration = {"kind": "narration", "text": "这是九个中文字"}
    dialogue = {"kind": "dialogue", "text": "这是九个中文字"}

    assert estimate_slice_seconds(narration) == 2
    assert estimate_slice_seconds(dialogue) == 3


def test_pack_voice_beats_respects_target_and_max_seconds():
    beats = [
        {"beat_id": "v001", "text": "一二三四五六七八九", "type": "narration", "duration": 2, "source_slice_ids": ["s001"]},
        {"beat_id": "v002", "text": "一二三四五六七八九", "type": "narration", "duration": 2, "source_slice_ids": ["s002"]},
        {"beat_id": "v003", "text": "这是一个较长的关键对白", "type": "dialogue", "duration": 5, "source_slice_ids": ["s003"]},
        {"beat_id": "v004", "text": "一二三四五六七八九", "type": "narration", "duration": 2, "source_slice_ids": ["s004"]},
    ]

    boards = pack_voice_beats(beats, target_seconds=6, max_seconds=8)

    assert [b["duration"] for b in boards] == [4, 7]
    assert boards[0]["voice_beat_ids"] == ["v001", "v002"]
    assert boards[1]["voice_beat_ids"] == ["v003", "v004"]
    assert all(b["duration"] <= 8 for b in boards)


def test_plan_script_locally_creates_traceable_boards():
    source = "沈砚看着玄墨。秦越吼道：“立刻装箱送来！”他挂断电话。"

    plan = plan_script_locally(source, target_seconds=6, max_seconds=8)

    assert plan["coverage_errors"] == []
    assert plan["stats"]["slices"] == 4
    assert plan["stats"]["voice_beats"] == 4
    assert plan["stats"]["boards"] >= 1
    assigned = [
        beat_id
        for board in plan["board_plan"]
        for beat_id in board["voice_beat_ids"]
    ]
    assert assigned == [beat["beat_id"] for beat in plan["voice_beats"]]


def test_plan_script_splits_long_beats_before_packing():
    source = "这是一段很长的旁白" * 12 + "。"

    plan = plan_script_locally(source)

    assert plan["planner_version"] >= 2
    assert all(beat["duration"] <= 15 for beat in plan["voice_beats"])
    assert all(board["duration"] <= 15 for board in plan["board_plan"])
    assert len(split_voice_text(source, "narration")) > 1
