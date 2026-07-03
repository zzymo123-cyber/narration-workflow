import pytest
from api.validation import validate_board_page


def _valid_board_page():
    """Return a valid board_page fixture matching the spec example."""
    return {
        "board_id": "seg_01_01_p01",
        "page": 1,
        "total_pages": 1,
        "compact_page": False,
        "voice_duration": 10,
        "visual_duration": 10,
        "board_duration": 10,
        "video_goal": "表现雨夜小巷中，林雪独自进入危险空间，阿明突然出现，关系带有悬念。",
        "voice_timeline": [
            {"beat_id": "v01", "type": "narration", "text": "她独自走在雨夜小巷里，手中握着一张泛黄的信纸。", "speaker": "旁白", "start": 0, "end": 5, "duration": 5},
            {"beat_id": "v02", "type": "dialogue", "text": "你怎么来了？", "speaker": "阿明", "start": 5, "end": 7, "duration": 2},
            {"beat_id": "v03", "type": "narration", "text": "她抬起头，看见了他。", "speaker": "旁白", "start": 7, "end": 10, "duration": 3},
        ],
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 2, "duration": 2, "voice_refs": ["v01"], "visual": "雨夜小巷全景", "camera": "wide_establishing", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "sync", "purpose": "建立氛围", "audio_behavior": "narration_sync", "continuity_from_previous": None, "transition_type": None},
            {"shot_id": "s02", "start": 2, "end": 5, "duration": 3, "voice_refs": ["v01"], "visual": "林雪低头", "camera": "medium_close", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "sync", "purpose": "展示角色", "audio_behavior": "narration_over", "continuity_from_previous": "林雪仍在小巷", "transition_type": "continuous"},
            {"shot_id": "s03", "start": 5, "end": 6, "duration": 1, "voice_refs": [], "visual": "巷口人影", "camera": "medium", "characters": ["阿明"], "scene": "雨夜小巷", "match_strategy": "foreshadow", "purpose": "制造悬念", "audio_behavior": "sound_lead_in", "continuity_from_previous": "切向巷口", "transition_type": "cut"},
            {"shot_id": "s04", "start": 6, "end": 7, "duration": 1, "voice_refs": ["v02"], "visual": "林雪僵住", "camera": "medium", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "reaction_first", "purpose": "先拍反应", "audio_behavior": "dialogue_offscreen", "continuity_from_previous": "脚步声停", "transition_type": "continuous"},
            {"shot_id": "s05", "start": 7, "end": 9, "duration": 2, "voice_refs": ["v03"], "visual": "林雪抬头", "camera": "close_up", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "reaction_first", "purpose": "情绪转变", "audio_behavior": "narration_over", "continuity_from_previous": "从僵住到抬头", "transition_type": "continuous"},
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
    page["shot_timeline"][2]["start"] = 6  # gap: s02 ends at 5, s03 starts at 6
    page["shot_timeline"][2]["duration"] = 0
    errors = validate_board_page(page)
    assert any("gap" in e.lower() or "间隙" in e or "overlap" in e.lower() for e in errors)


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
    page["board_duration"] = 7
    page["visual_duration"] = 7
    page["voice_duration"] = 7
    page["video"]["duration"] = 7
    # Remove v03 which is outside compact board_duration
    page["voice_timeline"] = page["voice_timeline"][:2]
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


def test_voice_beat_requires_valid_type_speaker_and_text():
    page = _valid_board_page()
    page["voice_timeline"][1]["speaker"] = "电话那头"
    page["voice_timeline"][2]["text"] = ""
    page["voice_timeline"].append({
        "beat_id": "v04",
        "type": "aside",
        "text": "无效声音类型",
        "speaker": "旁白",
        "start": 10,
        "end": 11,
        "duration": 1,
    })
    page["shot_timeline"].append({
        "shot_id": "s07",
        "start": 10,
        "end": 11,
        "duration": 1,
        "voice_refs": ["v04"],
        "visual": "补充镜头",
        "camera": "close_up",
        "characters": [],
        "scene": "雨夜小巷",
        "match_strategy": "supplement",
        "purpose": "补充声音",
        "audio_behavior": "narration_over",
        "continuity_from_previous": "延续",
        "transition_type": "cut",
    })
    page["board_duration"] = 11
    page["visual_duration"] = 11
    page["voice_duration"] = 11

    errors = validate_board_page(page)

    assert any("voice_timeline[1].speaker" in e for e in errors)
    assert any("voice_timeline[2].text" in e for e in errors)
    assert any("voice_timeline[3].type" in e for e in errors)


def test_voice_beat_duration_must_fit_text_length():
    page = _valid_board_page()
    page["voice_timeline"][0]["text"] = "这是一段明显超过两秒钟无法正常念完的长旁白内容"
    page["voice_timeline"][0]["end"] = 2
    page["voice_timeline"][0]["duration"] = 2
    errors = validate_board_page(page)
    assert any("too short for text" in e for e in errors)


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


def test_phone_dialogue_audio_behavior_is_valid():
    page = _valid_board_page()
    page["shot_timeline"][3]["audio_behavior"] = "phone_dialogue"
    errors = validate_board_page(page)
    assert not any("audio_behavior" in e for e in errors)


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
