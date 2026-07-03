from api.prompts import (
    audio_behavior_text,
    match_strategy_text,
    assemble_storyboard_prompt,
    assemble_video_prompt,
)
from api.shot_techniques import load_technique_library


def test_all_shot_techniques_use_fused_templates():
    library = load_technique_library()

    assert library
    for technique_id, technique in library.items():
        assert technique.get("image_visual_template"), technique_id
        assert technique.get("video_line_template"), technique_id


def test_audio_behavior_text_narration_sync():
    assert "旁白与当前画面同步" in audio_behavior_text("narration_sync", [])


def test_audio_behavior_text_dialogue_offscreen_with_speaker():
    result = audio_behavior_text("dialogue_offscreen", ["v01"],
                                  voice_timeline=[{"beat_id": "v01", "speaker": "阿明", "text": "你怎么来了？"}])
    assert "阿明" in result
    assert "画外响起" in result


def test_audio_behavior_text_phone_dialogue_with_speaker():
    result = audio_behavior_text("phone_dialogue", ["v01"],
                                  voice_timeline=[{"beat_id": "v01", "speaker": "秦越", "text": "立刻送来！"}])
    assert "秦越" in result
    assert "电话中传来" in result


def test_match_strategy_text_sync():
    assert "直接呈现" in match_strategy_text("sync")


def test_match_strategy_text_foreshadow():
    assert "暗示尚未发生" in match_strategy_text("foreshadow")


def test_assemble_storyboard_prompt():
    board_page = {
        "board_id": "seg_01_01_p01",
        "board_duration": 10,
        "story_continuity": {
            "just_happened": "林雪刚走进小巷。",
            "previous_final_panel": "林雪停在巷口，回头看向身后的脚步声。",
            "now_happening": "她发现墙角有黑影靠近。",
            "current_stage": "雨夜小巷",
            "visible_identity_refs": ["林雪：年轻女性主角", "雨夜小巷：当前场景参考"],
        },
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 2, "camera": "wide", "visual": "雨夜小巷全景",
             "audio_behavior": "narration_sync", "voice_refs": ["v01"]},
        ],
        "voice_timeline": [
            {"beat_id": "v01", "type": "narration", "speaker": "旁白", "text": "她走在小巷里。"},
        ],
    }
    prompt = assemble_storyboard_prompt(board_page)
    assert "导演分镜板" in prompt
    assert "总时长：10s" in prompt
    assert "连续性承接" in prompt
    assert "刚发生了什么：林雪刚走进小巷。" in prompt
    assert "上一张故事板最后一格：林雪停在巷口" in prompt
    assert "只用于承接角色位置、动作和情绪" in prompt
    assert "现在发生什么：" in prompt
    assert "她发现墙角有黑影靠近。" in prompt
    assert "接下来会发生什么" not in prompt
    assert "参考图身份" in prompt
    assert "只画现在发生的事件" in prompt
    assert "当前板1格" in prompt
    assert "镜头s01" in prompt
    assert "0-2s" in prompt
    assert "声音关系" in prompt
    assert "竖屏9:16" in prompt
    assert "不要完整旁白" in prompt


def test_storyboard_prompt_expands_static_shot_technique_hint():
    board_page = {
        "board_id": "tech_p01",
        "board_duration": 2,
        "shot_timeline": [
            {
                "shot_id": "s01",
                "start": 0,
                "end": 2,
                "duration": 2,
                "camera": "主观视角，快速聚焦铃铛",
                "visual": "舞台深处的铃铛突然晃动",
                "audio_behavior": "sound_lead_in",
                "voice_refs": [],
                "technique_id": "crash_zoom",
            },
        ],
        "voice_timeline": [],
    }

    prompt = assemble_storyboard_prompt(board_page)

    assert "舞台深处的铃铛突然晃动，铃铛压在画面视觉中心，形成清晰的特写焦点。" in prompt
    assert "。关键表情或物件处于画面视觉中心" not in prompt
    assert "镜头快速拉近" not in prompt


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
    assert "真人实拍电影短片" in prompt
    assert "说明：旁白，画外音" in prompt
    assert "她走在小巷里。" in prompt
    assert "0-2秒" in prompt
    assert "参考素材" in prompt
    assert "按从上到下的顺序演绎" in prompt
    assert "每一格是不同时刻，不是同时出现在画面里的多个场景" in prompt
    assert "角色参考图：林雪" in prompt
    assert "不是新增角色、复制体、镜像体" in prompt
    assert "不要生成重复人物、重复动物或第二个相似主体" in prompt
    assert "画面脚本" in prompt
    assert "约束：" in prompt
    assert "运动设计" not in prompt
    assert "不要生成字幕" in prompt
    assert "不要生成 BGM" in prompt
    assert "不要漫画" in prompt
    assert prompt.index("参考素材：") < prompt.index("生成要求：") < prompt.index("声音脚本：")
    assert prompt.index("声音脚本：") < prompt.index("画面脚本：") < prompt.index("约束：")
    assert "旁白A（0-2秒）" in prompt
    assert "“她走在小巷里。”" in prompt
    assert "说明：旁白，画外音" in prompt
    assert "声音为旁白A" in prompt


def test_video_prompt_expands_dynamic_shot_technique_template():
    board_page = {
        "board_id": "tech_p01",
        "board_duration": 2,
        "video_goal": "揭示铃声来源",
        "palette_id": "suspense_cold_blue",
        "voice_timeline": [],
        "shot_timeline": [
            {
                "shot_id": "s01",
                "start": 0,
                "end": 2,
                "duration": 2,
                "camera": "主观视角，快速聚焦铃铛",
                "visual": "舞台深处的铃铛突然晃动",
                "characters": ["林澈"],
                "scene": "老剧院",
                "match_strategy": "reveal",
                "purpose": "揭示异常线索",
                "audio_behavior": "sound_lead_in",
                "voice_refs": [],
                "continuity_from_previous": None,
                "technique_id": "crash_zoom",
            },
        ],
        "asset_refs": {"characters": ["林澈"], "scene": "老剧院", "props": ["铃铛"]},
    }

    prompt = assemble_video_prompt(board_page)

    assert "0-2秒：主观视角，快速聚焦铃铛，舞台深处的铃铛突然晃动；镜头快速压向铃铛特写，短暂停住，压住观众注意力。" in prompt
    assert "风格控制：" in prompt
    assert "色板：悬疑冷青灰（suspense_cold_blue）" in prompt
    assert "镜头技法：s01：快速拉近（crash_zoom）。" in prompt
    assert "时长2秒。镜头快速拉近" not in prompt
    assert "关键表情或物件处于画面视觉中心" not in prompt


def test_video_prompt_uses_camera_text_for_technique_target_fallback():
    board_page = {
        "board_id": "tech_p01",
        "board_duration": 2,
        "video_goal": "揭示铃声来源",
        "voice_timeline": [],
        "shot_timeline": [
            {
                "shot_id": "s01",
                "start": 0,
                "end": 2,
                "duration": 2,
                "camera": "主观视角模拟林澈视线，快速聚焦铃铛",
                "visual": "舞台深处一个金色铃铛悬空晃动，发出声响",
                "characters": ["林澈"],
                "audio_behavior": "sound_lead_in",
                "voice_refs": [],
                "technique_id": "crash_zoom",
            },
        ],
        "asset_refs": {"characters": ["林澈"], "scene": "老剧院", "props": ["铃铛"]},
    }

    prompt = assemble_video_prompt(board_page)

    assert "镜头快速压向铃铛特写" in prompt
    assert "镜头快速压向林澈特写" not in prompt


def test_video_prompt_phone_dialogue_uses_role_dialogue_not_live_dialogue():
    board_page = {
        "board_id": "seg_1_1_p03",
        "board_duration": 8,
        "video_goal": "朋友通过电话提醒沈砚",
        "voice_timeline": [
            {"beat_id": "v01", "type": "dialogue", "speaker": "秦越", "text": "立刻把它装箱送来！", "start": 0, "end": 4},
        ],
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 4, "camera": "近景", "visual": "沈砚听电话后脸色发白",
             "match_strategy": "reaction_first", "purpose": "表现朋友警告带来的恐惧",
             "audio_behavior": "phone_dialogue", "voice_refs": ["v01"],
             "continuity_from_previous": None},
            {"shot_id": "s02", "start": 4, "end": 8, "camera": "特写", "visual": "手机屏幕显示秦越来电",
             "match_strategy": "supplement", "purpose": "明确声音来自电话",
             "audio_behavior": "phone_dialogue", "voice_refs": ["v01"],
             "continuity_from_previous": "延续电话声"},
        ],
        "asset_refs": {"characters": ["沈砚", "秦越"], "scene": "工作室", "props": ["手机"]},
    }
    prompt = assemble_video_prompt(board_page, audio_reference_name="男声")
    assert "“立刻把它装箱送来！”" in prompt
    assert "秦越现场对白" not in prompt
    assert "音频1：旁白音色参考：男声" in prompt
    assert "不用于角色对白" in prompt
    assert "对白A（0-4秒）" in prompt
    assert "说明：秦越电话对白" in prompt
    assert "声音为对白A，从电话中传来" in prompt


def test_video_prompt_identity_rule_applies_to_all_referenced_subjects():
    board_page = {
        "video_goal": "测试主体一致性",
        "voice_timeline": [],
        "shot_timeline": [
            {"shot_id": "s01", "start": 0, "end": 2, "camera": "近景", "visual": "林雪和阿明对视",
             "audio_behavior": "ambient_only", "voice_refs": []},
        ],
        "asset_refs": {"characters": ["林雪", "阿明"], "scene": "", "props": []},
    }
    prompt = assemble_video_prompt(
        board_page,
        asset_urls={"characters": {"林雪": "/chars/lin.png", "阿明": "/chars/aming.png"}},
        storyboard_image_path="/boards/seg.png",
    )
    assert "林雪、阿明" in prompt
    assert "不是新增角色、复制体、镜像体" in prompt
    assert "不要生成重复人物、重复动物或第二个相似主体" in prompt


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
    assert "角色参考图：林雪" not in prompt
    assert "场景参考图" not in prompt
