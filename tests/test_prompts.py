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
