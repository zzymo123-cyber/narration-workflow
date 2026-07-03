"""End-to-end integration test: create project, simulate decomposition, validate, assemble prompts."""
import json
import tempfile
from pathlib import Path

from api.pipeline import write_pipeline, read_pipeline
from api.decomposition import build_step3_prompt, parse_decomposition_response
from api.validation import validate_board_page
from api.prompts import assemble_storyboard_prompt, assemble_video_prompt
from api.duration import narration_duration, dialogue_duration


def test_e2e_create_project():
    """Step 1: Create a project with pipeline.json"""
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        data = {
            "project": "e2e_test",
            "narration_style": "third_person",
            "source_text": "林雪独自走在雨夜小巷里，手中握着一张泛黄的信纸。阿明突然出现在巷口。",
            "assets": {"characters": {}, "scenes": {}, "props": {}},
            "narration_segments": {},
        }
        write_pipeline(project_dir, data)
        result = read_pipeline(project_dir)
        assert result["project"] == "e2e_test"
        assert result["narration_style"] == "third_person"
        assert (project_dir / "pipeline.json").exists()


def test_e2e_decomposition_prompt():
    """Step 2: Verify step3 prompt contains all required elements for board generation"""
    assets = {
        "characters": {"林雪": {"seed": "年轻女性"}, "阿明": {"seed": "中年男性"}},
        "scenes": {"雨夜小巷": {"seed": "昏暗小巷"}},
        "props": {},
    }
    prompt = build_step3_prompt("third_person", "seg_1_1", "雨夜小巷", ["林雪", "阿明"], 2, assets)
    # Must contain key spec elements
    assert "旁白" in prompt
    assert "voice_timeline" in prompt
    assert "shot_timeline" in prompt
    assert "4.5" in prompt
    assert "3" in prompt
    assert "整数" in prompt
    assert "video_goal" in prompt
    assert "match_strategy" in prompt
    assert "audio_behavior" in prompt
    assert "board_duration" in prompt


def test_e2e_parse_and_validate():
    """Step 3: Simulate LLM decomposition output, parse it, validate all board_pages"""
    simulated_llm_response = {
        "narration_style": "third_person",
        "assets": {
            "characters": {
                "林雪": {"seed": "年轻女性，短发，穿黑色风衣"},
                "阿明": {"seed": "中年男性，戴眼镜，穿灰色西装"},
            },
            "scenes": {
                "雨夜小巷": {"seed": "昏暗的城市小巷，雨夜，路灯昏黄"},
            },
            "props": {
                "泛黄信纸": {"seed": "一封旧信纸，边缘泛黄"},
            },
        },
        "narration_segments": {
            "seg_01_01": {
                "episode": 1,
                "segment_index": 1,
                "characters_in_segment": ["林雪", "阿明"],
                "scene_location": "雨夜小巷",
                "boards": [{
                    "board_id": "seg_01_01_p01",
                    "page": 1,
                    "total_pages": 1,
                    "compact_page": False,
                    "voice_duration": 9,
                    "visual_duration": 10,
                    "board_duration": 10,
                    "video_goal": "表现雨夜小巷中，林雪独自进入危险空间，阿明突然出现，关系带有悬念。",
                    "voice_timeline": [
                        {"beat_id": "v01", "type": "narration", "text": "她独自走在雨夜小巷里，手中握着一张泛黄的信纸。", "speaker": "旁白", "start": 0, "end": 5, "duration": 5},
                        {"beat_id": "v02", "type": "dialogue", "text": "你怎么来了？", "speaker": "阿明", "start": 5, "end": 7, "duration": 2},
                        {"beat_id": "v03", "type": "narration", "text": "她抬起头，看见了他。", "speaker": "旁白", "start": 7, "end": 9, "duration": 2},
                    ],
                    "shot_timeline": [
                        {"shot_id": "s01", "start": 0, "end": 2, "duration": 2, "voice_refs": ["v01"], "visual": "雨夜小巷全景，雨水沿着屋檐滴落", "camera": "wide_establishing", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "sync", "purpose": "建立雨夜氛围和场景", "audio_behavior": "narration_sync", "continuity_from_previous": None, "transition_type": None},
                        {"shot_id": "s02", "start": 2, "end": 4, "duration": 2, "voice_refs": ["v01"], "visual": "林雪低头看手中的信纸", "camera": "medium_close", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "sync", "purpose": "展示角色和关键道具", "audio_behavior": "narration_over", "continuity_from_previous": "林雪仍在小巷", "transition_type": "continuous"},
                        {"shot_id": "s03", "start": 4, "end": 5, "duration": 1, "voice_refs": [], "visual": "巷口出现一个人影", "camera": "medium", "characters": ["阿明"], "scene": "雨夜小巷", "match_strategy": "foreshadow", "purpose": "制造悬念", "audio_behavior": "sound_lead_in", "continuity_from_previous": "切向巷口", "transition_type": "cut"},
                        {"shot_id": "s04", "start": 5, "end": 7, "duration": 2, "voice_refs": ["v02"], "visual": "林雪僵住，镜头对准她的脸", "camera": "close_up", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "reaction_first", "purpose": "先拍反应再揭示说话者", "audio_behavior": "dialogue_offscreen", "continuity_from_previous": "脚步声停", "transition_type": "continuous"},
                        {"shot_id": "s05", "start": 7, "end": 9, "duration": 2, "voice_refs": ["v03"], "visual": "林雪缓缓抬头", "camera": "close_up", "characters": ["林雪"], "scene": "雨夜小巷", "match_strategy": "reaction_first", "purpose": "情绪转变", "audio_behavior": "narration_over", "continuity_from_previous": "从僵住到抬头", "transition_type": "continuous"},
                        {"shot_id": "s06", "start": 9, "end": 10, "duration": 1, "voice_refs": [], "visual": "两人对视，雨声渐大", "camera": "two_shot", "characters": ["林雪", "阿明"], "scene": "雨夜小巷", "match_strategy": "emotional_landing", "purpose": "情感落点", "audio_behavior": "ambient_only", "continuity_from_previous": "同框延续", "transition_type": "continuous"},
                    ],
                    "storyboard_image": {"status": "needed", "prompt": "", "task_id": None, "url": None, "local_path": None},
                    "video": {"status": "needed", "duration": 10, "prompt": "", "task_id": None, "url": None, "local_path": None},
                    "asset_refs": {"characters": ["林雪", "阿明"], "scene": "雨夜小巷", "props": []},
                }],
            },
        },
    }

    # Parse
    result = parse_decomposition_response(simulated_llm_response)
    assert "seg_01_01" in result["narration_segments"]

    # Validate all board_pages
    for seg_key, seg in result["narration_segments"].items():
        for i, board in enumerate(seg.get("boards", [])):
            errors = validate_board_page(board)
            assert errors == [], f"Board {seg_key}[{i}] validation errors: {errors}"

    # Save to pipeline.json
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        write_pipeline(project_dir, result)
        loaded = read_pipeline(project_dir)
        assert "seg_01_01" in loaded["narration_segments"]


def test_e2e_prompt_assembly():
    """Step 4: Verify storyboard and video prompt assembly from decomposition data"""
    board = {
        "board_id": "seg_01_01_p01",
        "board_duration": 10,
        "video_goal": "表现雨夜相遇的悬念氛围",
        "voice_timeline": [
            {"beat_id": "v01", "type": "narration", "speaker": "旁白", "text": "她走在小巷里。", "start": 0, "end": 2},
            {"beat_id": "v02", "type": "dialogue", "speaker": "阿明", "text": "你怎么来了？", "start": 4, "end": 6},
        ],
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 2, "camera": "wide", "visual": "雨夜小巷全景",
             "match_strategy": "sync", "purpose": "建立氛围",
             "audio_behavior": "narration_sync", "voice_refs": ["v01"],
             "continuity_from_previous": None},
            {"shot_id": "s02", "start": 2, "end": 4, "camera": "medium", "visual": "林雪独行",
             "match_strategy": "supplement", "purpose": "补充画面",
             "audio_behavior": "ambient_only", "voice_refs": [],
             "continuity_from_previous": "延续"},
            {"shot_id": "s03", "start": 4, "end": 6, "camera": "close_up", "visual": "林雪抬头",
             "match_strategy": "reaction_first", "purpose": "反应镜头",
             "audio_behavior": "dialogue_offscreen", "voice_refs": ["v02"],
             "continuity_from_previous": "从僵住到抬头"},
            {"shot_id": "s04", "start": 6, "end": 8, "camera": "medium", "visual": "阿明走入画面",
             "match_strategy": "reveal", "purpose": "揭示说话者",
             "audio_behavior": "ambient_only", "voice_refs": [],
             "continuity_from_previous": "阿明走近"},
            {"shot_id": "s05", "start": 8, "end": 10, "camera": "two_shot", "visual": "两人对视",
             "match_strategy": "emotional_landing", "purpose": "情感落点",
             "audio_behavior": "ambient_only", "voice_refs": [],
             "continuity_from_previous": "同框"},
        ],
        "asset_refs": {"characters": ["林雪", "阿明"], "scene": "雨夜小巷", "props": []},
    }

    # Storyboard prompt
    sb_prompt = assemble_storyboard_prompt(board)
    assert "导演分镜板" in sb_prompt
    assert "总时长：10s" in sb_prompt
    assert "竖屏9:16" in sb_prompt
    assert "镜头s01" in sb_prompt
    assert "0-2s" in sb_prompt
    assert "不要完整旁白" in sb_prompt

    # Video prompt
    video_prompt = assemble_video_prompt(
        board,
        asset_urls={"characters": {"林雪": "/chars/lin.png"}},
        storyboard_image_path="/boards/seg_01_01_p01.png"
    )
    assert "表现雨夜相遇的悬念氛围" in video_prompt
    assert "真人实拍电影短片" in video_prompt
    assert "“她走在小巷里。”" in video_prompt
    assert "“你怎么来了？”" in video_prompt
    assert "说明：阿明画外对白" in video_prompt
    assert "画外对白" in video_prompt
    assert "角色参考图：林雪" in video_prompt
    assert "分镜板参考图" in video_prompt
    assert "约束" in video_prompt
    assert "运动设计" not in video_prompt
    assert "旁白A（0-2秒）" in video_prompt
    assert "“她走在小巷里。”" in video_prompt
    assert "声音为旁白A" in video_prompt
    assert "不要生成字幕" in video_prompt
    assert "不要生成 BGM" in video_prompt
    # Missing asset URLs should not appear
    assert "角色参考图：阿明" not in video_prompt


def test_e2e_duration_calculation():
    """Step 5: Verify duration calculations match spec"""
    # Narration: 4.5 字/秒
    assert narration_duration("她独自走在雨夜小巷里，手中握着一张泛黄的信纸。") == 5  # 21 chars
    assert narration_duration("你好世界") == 1  # 4 chars / 4.5 < 1

    # Dialogue: 3 字/秒
    assert dialogue_duration("你怎么来了？") == 2  # 6 chars / 3 = 2
    assert dialogue_duration("来") == 1  # 1 char / 3 < 1

    # All durations must be integers
    for text in ["测", "测试", "测试文本内容", "这是一段很长的旁白文本用于测试整数秒的计算结果是否正确"]:
        assert isinstance(narration_duration(text), int)
        assert isinstance(dialogue_duration(text), int)
