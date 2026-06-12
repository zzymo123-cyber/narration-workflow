import json
from api.decomposition import build_step1_prompt, build_step2_prompt, build_step3_prompt, parse_decomposition_response


def test_build_step1_prompt_contains_key_elements():
    prompt = build_step1_prompt("third_person")
    assert "旁白" in prompt
    assert "characters" in prompt
    assert "scenes" in prompt
    assert "props" in prompt
    assert "seed" in prompt


def test_build_step2_prompt_contains_key_elements():
    assets = {"characters": {"林雪": {"seed": "年轻女性"}}, "scenes": {}, "props": {}}
    prompt = build_step2_prompt("third_person", assets)
    assert "旁白" in prompt
    assert "segments" in prompt
    assert "scene_location" in prompt
    assert "characters_in_segment" in prompt
    assert "num_boards" in prompt
    assert "林雪" in prompt  # assets injected


def test_build_step3_prompt_contains_key_elements():
    assets = {"characters": {"林雪": {"seed": "年轻女性"}}, "scenes": {"雨夜小巷": {"seed": "昏暗"}}, "props": {}}
    prompt = build_step3_prompt("third_person", "seg_1_1", "雨夜小巷", ["林雪"], 2, assets)
    assert "旁白" in prompt
    assert "voice_timeline" in prompt
    assert "shot_timeline" in prompt
    assert "4.5" in prompt  # narration speed
    assert "3" in prompt  # dialogue speed
    assert "整数" in prompt  # integer seconds
    assert "video_goal" in prompt
    assert "match_strategy" in prompt
    assert "audio_behavior" in prompt
    assert "board_duration" in prompt
    assert "seg_1_1" in prompt  # seg_key injected
    assert "雨夜小巷" in prompt  # scene injected


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
