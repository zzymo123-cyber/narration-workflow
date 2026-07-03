from typing import Optional

from api.shot_techniques import (
    get_technique,
    shot_technique_storyboard_visual,
    shot_technique_video_hint,
    shot_technique_video_line,
)
from api.color_palettes import board_color_palette, color_palette_video_note

AUDIO_BEHAVIOR_MAP = {
    "narration_sync": "旁白与当前画面同步，画面直接承载旁白内容",
    "narration_over": "旁白覆盖在画面上，画面可以补充、反差或伏笔，不要机械复述旁白",
    "dialogue_sync": "角色台词与当前画面同步，可以看到说话角色或其明确动作",
    "dialogue_offscreen": "角色台词画外响起，说话者不立即露面，先拍听到台词后的反应",
    "phone_dialogue": "角色台词从电话中传来，画面可以拍听电话者反应、手机或电话那头说话者",
    "ambient_only": "只有环境声，没有旁白和台词",
    "sound_lead_in": "非语言声音先行，用来引出下一镜头或人物",
    "dramatic_silence": "戏剧性静默，用于震惊反应、冲突后停顿、压迫感",
    "ambient_transition": "环境声作为转场桥",
}

MATCH_STRATEGY_MAP = {
    "sync": "画面直接呈现旁白或台词描述的内容",
    "supplement": "画面补充旁白或台词未说出的信息",
    "contrast": "画面与旁白或台词形成反差",
    "foreshadow": "画面暗示尚未发生的事件",
    "reaction_first": "先拍角色反应，再揭示说话者或信息来源",
    "reveal": "画面揭示之前铺垫的信息",
    "emotional_landing": "画面提供情感落点，让观众消化情绪",
    "transition": "画面作为场景或情绪的过渡桥",
}

PROHIBITIONS = """\
- 不要生成字幕
- 不要生成额外文字
- 不要生成 BGM、背景音乐、配乐、氛围音乐
- 只允许旁白、角色台词、环境声和必要动作音效
- 无旁白无台词的镜头，只允许环境声或指定音效，不要用音乐填充
- 不要漫画、不要线稿、不要插画、不要分镜稿质感
- 不要改变角色长相、发型、服装
- 不要让人物、道具、空间位置在镜头之间突然跳变
- 分镜板参考图只作为镜头顺序、构图关系、人物位置和动作节点参考，不要把分镜板上的编号、时间码、说明文字生成进视频画面"""


def _text(value) -> str:
    if value is None:
        return ""
    return str(value)


def audio_behavior_text(behavior: str, voice_refs: list[str],
                         voice_timeline: Optional[list[dict]] = None) -> str:
    text = AUDIO_BEHAVIOR_MAP.get(behavior, behavior)
    if behavior in {"dialogue_offscreen", "phone_dialogue"} and voice_refs and voice_timeline:
        beat_map = {b["beat_id"]: b for b in voice_timeline}
        speakers = []
        for ref in voice_refs:
            beat = beat_map.get(ref)
            if beat and beat.get("speaker"):
                speakers.append(beat["speaker"])
        if speakers:
            if behavior == "phone_dialogue":
                return f"{','.join(speakers)}台词从电话中传来，画面可以拍听电话者反应、手机或电话那头说话者"
            return f"{','.join(speakers)}台词画外响起，说话者不立即露面，先拍听到台词后的反应"
    return text


def match_strategy_text(strategy: str) -> str:
    return MATCH_STRATEGY_MAP.get(strategy, strategy)


def _is_snake_character(name: str, url: str = "") -> bool:
    return any(word in name for word in ("玄墨", "蟒", "蛇"))


def _character_sort_key(item: tuple[str, str]) -> tuple[int, str]:
    name, url = item
    return (0 if _is_snake_character(name, url) else 1, name)


def _voice_role(beat: dict) -> str:
    beat_type = _text(beat.get("type"))
    speaker = _text(beat.get("speaker")) or ("旁白" if beat_type == "narration" else "角色")
    if beat_type == "narration":
        if speaker and speaker != "旁白":
            return f"第一人称男声旁白（{speaker}画外音 voice-over，不是现场对白）"
        return "旁白画外音 voice-over，不是现场对白"
    return f"{speaker}角色对白"


def _narration_speakers(voice_timeline: list[dict]) -> list[str]:
    speakers = []
    for beat in voice_timeline:
        if _text(beat.get("type") or "narration") != "narration":
            continue
        speaker = _text(beat.get("speaker")) or "旁白"
        if speaker not in speakers:
            speakers.append(speaker)
    return speakers


def _style_control_section(board_page: dict) -> str:
    lines = []
    palette = board_color_palette(board_page)
    if palette:
        palette_name = _text(palette.get("name")) or _text(palette.get("id"))
        palette_id = _text(palette.get("id"))
        lines.append(f"色板：{palette_name}（{palette_id}），以最后一张参考图和下方色板参考说明为准。")

    technique_lines = []
    for shot in board_page.get("shot_timeline", []) or []:
        technique = get_technique(shot.get("technique_id"))
        if not technique:
            continue
        technique_lines.append(
            f"{_text(shot.get('shot_id')) or 'shot'}："
            f"{_text(technique.get('name'))}（{_text(technique.get('id'))}）"
        )
    if technique_lines:
        lines.append("镜头技法：" + "；".join(technique_lines) + "。")

    return "\n".join(lines)


def _compact_text(value, limit: int = 120) -> str:
    text = " ".join(_text(value).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _story_continuity_section(board_page: dict) -> list[str]:
    continuity = board_page.get("story_continuity") or board_page.get("visual_story_context") or {}
    just_happened = _compact_text(continuity.get("just_happened"), 120)
    now_happening = _compact_text(
        continuity.get("now_happening") or board_page.get("video_goal") or continuity.get("current_stage"),
        140,
    )
    previous_final_panel = _compact_text(continuity.get("previous_final_panel"), 160)
    current_stage = _compact_text(continuity.get("current_stage"), 80)
    visible_refs = continuity.get("visible_identity_refs") or []
    spatial_rules = [
        _compact_text(item, 120)
        for item in continuity.get("spatial_rules", []) or []
        if _compact_text(item, 120)
    ]
    if not any([just_happened, previous_final_panel, now_happening, current_stage, visible_refs, spatial_rules]):
        return []

    lines = [
        "任务：",
        f"生成一张竖屏{len(board_page.get('shot_timeline', []))}格连续导演分镜板，按从上到下阅读。",
        "",
        "连续性承接：",
        f"刚发生了什么：{just_happened or '无'}",
    ]
    if previous_final_panel:
        lines.extend([
            f"上一张故事板最后一格：{previous_final_panel}",
            "说明：上一张故事板最后一格只用于承接角色位置、动作和情绪，不要重复画成当前板的全部内容。",
        ])
    lines.extend(["", "现在发生什么：", now_happening or "按当前板推进"])
    if current_stage:
        lines.append(f"当前场景：{current_stage}")
    if spatial_rules:
        lines.extend(["", "空间方位："])
        lines.extend(spatial_rules)
    if visible_refs:
        refs = "；".join(_compact_text(item, 70) for item in visible_refs if _compact_text(item, 70))
        if refs:
            lines.extend(["", "参考图身份：", refs])
    lines.extend([
        "",
        "限制：",
        "只画现在发生的事件。上一张故事板最后一格只作为承接，不要重复上一板，不要提前画后续事件。",
    ])
    if visible_refs:
        lines.append("只出现参考图对应的当前角色，不出现第三个人或未列出的角色。")
    return lines


def _audio_section(voice_timeline: list[dict], audio_reference_name: Optional[str] = None) -> str:
    narration = [beat for beat in voice_timeline if _text(beat.get("type") or "narration") == "narration"]
    dialogue = [beat for beat in voice_timeline if _text(beat.get("type")) == "dialogue"]
    lines = []
    if narration:
        speakers = _narration_speakers(voice_timeline)
        if speakers == ["旁白"]:
            lines.append("旁白：画外音 voice-over，不是角色现场对白。")
        else:
            lines.append(f"旁白：{'、'.join(speakers)}第一人称男声画外音 voice-over，不是现场对白。")
    if dialogue:
        lines.append("对白：角色对白，可能是现场、画外或电话声；以每个镜头的声音标注为准。")
    if audio_reference_name:
        lines.append(
            f"旁白音色：使用音频1（{audio_reference_name}）的声线、年龄感和语气质地；对白不要模仿音频1。"
        )
    lines.extend([
        "不要字幕、不要背景音乐、不要配乐；只保留旁白、角色台词、环境声和必要动作音效。",
    ])
    return "\n".join(lines)


def _beat_behavior_map(shots: list[dict]) -> dict[str, str]:
    result = {}
    for shot in shots:
        behavior = _text(shot.get("audio_behavior"))
        for ref in shot.get("voice_refs", []) or []:
            result.setdefault(ref, behavior)
    return result


def _voice_line(beat: dict, behavior: str = "", audio_reference_name: Optional[str] = None) -> str:
    beat_type = _text(beat.get("type") or "narration")
    speaker = _text(beat.get("speaker")) or ("旁白" if beat_type == "narration" else "角色")
    text = _text(beat.get("text"))
    if beat_type == "narration":
        label = "旁白" if speaker == "旁白" else f"{speaker}旁白"
        tone = "，使用音频1音色" if audio_reference_name else ""
        return f"{label}（画外音{tone}）：{text}"
    if behavior == "phone_dialogue":
        return f"{speaker}电话对白：{text}"
    if behavior == "dialogue_offscreen":
        return f"{speaker}画外对白：{text}"
    return f"{speaker}对白：{text}"


def _voice_description(beat: dict, behavior: str = "", audio_reference_name: Optional[str] = None) -> str:
    beat_type = _text(beat.get("type") or "narration")
    speaker = _text(beat.get("speaker")) or ("旁白" if beat_type == "narration" else "角色")
    if beat_type == "narration":
        label = "旁白" if speaker == "旁白" else f"{speaker}旁白"
        tone = "，使用音频1音色" if audio_reference_name else ""
        return f"{label}，画外音{tone}，不是现场对白"
    if behavior == "phone_dialogue":
        return f"{speaker}电话对白"
    if behavior == "dialogue_offscreen":
        return f"{speaker}画外对白"
    return f"{speaker}对白"


def _beat_label(index: int, beat: dict) -> str:
    letter = chr(ord("A") + index)
    beat_type = _text(beat.get("type") or "narration")
    if beat_type == "narration":
        return f"旁白{letter}"
    return f"对白{letter}"


def _beat_labels(voice_timeline: list[dict]) -> dict[str, str]:
    labels = {}
    for index, beat in enumerate(voice_timeline):
        beat_id = beat.get("beat_id")
        if beat_id:
            labels[beat_id] = _beat_label(index, beat)
    return labels


def _shot_sound_text(shot: dict, beat_map: dict, beat_labels: dict[str, str]) -> str:
    refs = []
    behavior = _text(shot.get("audio_behavior"))
    for ref in shot.get("voice_refs", []) or []:
        beat = beat_map.get(ref)
        if beat:
            label = beat_labels.get(ref, "对应声音")
            if _text(beat.get("type") or "narration") == "narration":
                refs.append(f"{label}继续，画外音")
            elif behavior == "phone_dialogue":
                refs.append(f"{label}继续，从电话中传来")
            elif behavior == "dialogue_offscreen":
                refs.append(f"{label}继续，画外传来")
            else:
                refs.append(f"{label}继续，角色对白")
    if refs:
        return "；".join(refs)
    if behavior == "ambient_only":
        return "只有环境声和必要动作音效，不要配乐。"
    if behavior == "dramatic_silence":
        return "戏剧性静默，只保留轻微环境声。"
    return audio_behavior_text(behavior, [], [])


def _shot_sound_coverage(shot: dict, beat_map: dict, beat_labels: dict[str, str]) -> str:
    refs = []
    behavior = _text(shot.get("audio_behavior"))
    for ref in shot.get("voice_refs", []) or []:
        beat = beat_map.get(ref)
        if beat:
            label = beat_labels.get(ref, "对应声音")
            if _text(beat.get("type") or "narration") == "narration":
                refs.append(f"声音为{label}。")
            elif behavior == "phone_dialogue":
                refs.append(f"声音为{label}，从电话中传来。")
            elif behavior == "dialogue_offscreen":
                refs.append(f"声音为{label}，画外传来。")
            else:
                refs.append(f"声音为{label}。")
    if refs:
        return "".join(refs)
    if behavior == "ambient_only":
        return "只有环境声和必要动作音效，不要配乐。"
    if behavior == "dramatic_silence":
        return "戏剧性静默，只保留轻微环境声。"
    return audio_behavior_text(behavior, [], [])


def _reference_section(asset_urls: dict, asset_refs: dict, storyboard_image_path: Optional[str],
                       audio_reference_name: Optional[str] = None,
                       voice_timeline: Optional[list[dict]] = None) -> str:
    lines = []
    ref_index = 1
    if storyboard_image_path:
        lines.append(
            f"  图片{ref_index}：竖向分镜板参考图。按从上到下的顺序演绎成一段连续视频；"
            "每一格是不同时刻，不是同时出现在画面里的多个场景。只参考镜头顺序、构图关系、人物位置和动作节点；"
            "不要生成图上的编号、时间码、中文说明。"
        )
        ref_index += 1

    char_urls = asset_urls.get("characters", {})
    referenced_chars = []
    for char, url in sorted(char_urls.items(), key=_character_sort_key):
        if char not in asset_refs.get("characters", []):
            continue
        referenced_chars.append(char)
        note = f"  图片{ref_index}：角色参考图：{char}。保持脸型、发型、服装和角色身份一致。"
        if _is_snake_character(char, url):
            note = (
                f"  图片{ref_index}：角色参考图：{char}。{char}是蛇类/蟒蛇角色，必须无四肢、无爪、无外耳、无脚掌；"
                "禁止画成蜥蜴、龙、四脚爬行动物或长脚怪物。"
            )
        lines.append(note)
        ref_index += 1

    if storyboard_image_path and referenced_chars:
        names = "、".join(referenced_chars)
        lines.append(
            f"  主体一致性：图片1分镜板里反复出现的{names}，和后续角色参考图里的{names}是同一批角色在不同时间点的连续动作，"
            "不是新增角色、复制体、镜像体或同时出现的多个相同主体。每个镜头只保留剧情要求的主体数量，不要生成重复人物、重复动物或第二个相似主体。"
        )

    scene_url = asset_urls.get("scene")
    if scene_url:
        lines.append(f"  图片{ref_index}：场景参考图。保持空间结构、光线方向和主要陈设一致。")
        ref_index += 1

    prop_urls = asset_urls.get("props", {})
    for prop in asset_refs.get("props", []):
        url = prop_urls.get(prop)
        if url:
            lines.append(f"  图片{ref_index}：道具参考图：{prop}。保持外形、材质和位置逻辑一致。")
            ref_index += 1

    if audio_reference_name:
        speakers = _narration_speakers(voice_timeline or [])
        target = "旁白" if not speakers or speakers == ["旁白"] else "、".join(speakers) + "旁白"
        lines.append(
            f"  音频1：旁白音色参考：{audio_reference_name}。只用于{target}/narration 的声线和语气，不用于角色对白。"
        )

    return "\n".join(lines)


def assemble_storyboard_prompt(board_page: dict) -> str:
    lines = ["导演分镜板", f"总时长：{_text(board_page.get('board_duration'))}s", ""]
    story_lines = _story_continuity_section(board_page)
    if story_lines:
        lines.extend(story_lines)
        lines.append("")
        lines.append(f"当前板{len(board_page.get('shot_timeline', []))}格：")
    beat_map = {b.get("beat_id"): b for b in board_page.get("voice_timeline", [])}
    for shot in board_page.get("shot_timeline", []):
        lines.append(f"镜头{_text(shot.get('shot_id'))} | {_text(shot.get('start'))}-{_text(shot.get('end'))}s | {_text(shot.get('camera'))}")
        lines.append(shot_technique_storyboard_visual(shot))
        voice_notes = []
        for ref in shot.get("voice_refs", []) or []:
            beat = beat_map.get(ref)
            if beat:
                voice_notes.append(_voice_role(beat))
        behavior = audio_behavior_text(_text(shot.get("audio_behavior")), shot.get("voice_refs", []), board_page.get("voice_timeline"))
        if behavior or voice_notes:
            lines.append(f"声音关系：{behavior}" + (f"；说话者：{'、'.join(voice_notes)}" if voice_notes else ""))
        lines.append("")
    lines.extend([
        "风格：黑白铅笔线稿/导演分镜稿",
        f"版面：竖屏9:16，{len(board_page.get('shot_timeline', []))}格连续分镜，竖屏规整排版",
        "每格包含镜头编号、整数秒时间码、简短画面说明",
        "不要完整旁白",
        "不要完整台词",
        "不要大量中文小字",
        "保持角色、场景、道具一致",
        "每格画面要清楚表达镜头内容、角色位置和情绪变化",
    ])
    return "\n".join(lines)


def assemble_video_prompt(board_page: dict, asset_urls: Optional[dict] = None,
                           storyboard_image_path: Optional[str] = None,
                           audio_reference_name: Optional[str] = None) -> str:
    asset_urls = asset_urls or {}
    sections = []
    asset_refs = board_page.get("asset_refs", {})
    voice_timeline = board_page.get("voice_timeline", [])

    reference_text = _reference_section(
        asset_urls,
        asset_refs,
        storyboard_image_path,
        audio_reference_name,
        voice_timeline,
    )
    if reference_text:
        sections.append("参考素材：\n" + reference_text)

    spatial_rules = [
        _compact_text(item, 140)
        for item in (board_page.get("story_continuity") or {}).get("spatial_rules", []) or []
        if _compact_text(item, 140)
    ]
    if spatial_rules:
        sections.append("空间方位：\n" + "\n".join(spatial_rules))

    sections.append(
        "导演手法：\n"
        "按每个镜头的 camera、purpose 和 visual 执行景别、机位、运镜、主体调度和构图焦点；"
        "镜头之间要有节奏变化，避免全程同一景别或静态平移。"
    )
    style_control = _style_control_section(board_page)
    if style_control:
        sections.append("风格控制：\n" + style_control)

    sections.append(
        "生成要求：\n"
        "生成真人实拍电影短片，竖屏9:16，镜头构图、人物位置和动作节点以分镜板为准。\n"
        "成片不能是分镜稿、漫画、线稿或插画质感。\n"
        f"本段目标：{board_page.get('video_goal', '')}"
    )

    if voice_timeline:
        behavior_map = _beat_behavior_map(board_page.get("shot_timeline", []))
        labels = _beat_labels(voice_timeline)
        voice_lines = [_audio_section(voice_timeline, audio_reference_name), ""]
        for beat in voice_timeline:
            behavior = behavior_map.get(beat.get("beat_id"), "")
            voice_lines.append(
                f"{labels.get(beat.get('beat_id'), '声音')}（{_text(beat.get('start'))}-{_text(beat.get('end'))}秒）：\n"
                f"“{_text(beat.get('text'))}”\n"
                f"说明：{_voice_description(beat, behavior, audio_reference_name)}。"
            )
        sections.append("声音脚本：\n" + "\n".join(line for line in voice_lines if line))

    shot_lines = []
    beat_map = {b.get("beat_id"): b for b in voice_timeline}
    beat_labels = _beat_labels(voice_timeline)
    for shot in board_page.get("shot_timeline", []):
        line = shot_technique_video_line(shot)
        if not line:
            line = f"{_text(shot.get('start'))}-{_text(shot.get('end'))}秒：{_text(shot.get('camera'))}，{_text(shot.get('visual'))}。"
            technique_hint = shot_technique_video_hint(shot)
            if technique_hint:
                line += technique_hint
        sound = _shot_sound_coverage(shot, beat_map, beat_labels)
        if sound:
            line += sound
        if shot.get("continuity_from_previous"):
            line += f"衔接：{_text(shot.get('continuity_from_previous'))}。"
        shot_lines.append(line)
    if shot_lines:
        sections.append("画面脚本：\n" + "\n".join(shot_lines).strip())

    palette_note = color_palette_video_note(board_page)
    if palette_note:
        sections.append("色板参考：\n" + palette_note)

    sections.append(
        "约束：\n"
        "按每个镜头的画面描述完成真实动作、反应或焦点变化；不要额外发明与分镜无关的运动；不要静态图片平移。\n"
        f"{PROHIBITIONS}"
    )

    return "\n\n".join(sections)
