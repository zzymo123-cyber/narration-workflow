from typing import Optional

AUDIO_BEHAVIOR_MAP = {
    "narration_sync": "旁白与当前画面同步，画面直接承载旁白内容",
    "narration_over": "旁白覆盖在画面上，画面可以补充、反差或伏笔，不要机械复述旁白",
    "dialogue_sync": "角色台词与当前画面同步，可以看到说话角色或其明确动作",
    "dialogue_offscreen": "角色台词画外响起，说话者不立即露面，先拍听到台词后的反应",
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
- 不要改变角色长相、发型、服装
- 不要让人物、道具、空间位置在镜头之间突然跳变
- 分镜板参考图只作为镜头顺序、构图关系和人物位置参考，不要把分镜板上的编号、时间码、说明文字生成进视频画面"""


def audio_behavior_text(behavior: str, voice_refs: list[str],
                         voice_timeline: Optional[list[dict]] = None) -> str:
    text = AUDIO_BEHAVIOR_MAP.get(behavior, behavior)
    if behavior == "dialogue_offscreen" and voice_refs and voice_timeline:
        beat_map = {b["beat_id"]: b for b in voice_timeline}
        speakers = []
        for ref in voice_refs:
            beat = beat_map.get(ref)
            if beat and beat.get("speaker"):
                speakers.append(beat["speaker"])
        if speakers:
            return f"{','.join(speakers)}台词画外响起，说话者不立即露面，先拍听到台词后的反应"
    return text


def match_strategy_text(strategy: str) -> str:
    return MATCH_STRATEGY_MAP.get(strategy, strategy)


def assemble_storyboard_prompt(board_page: dict) -> str:
    lines = ["导演分镜板", f"总时长：{board_page['board_duration']}s", ""]
    for shot in board_page.get("shot_timeline", []):
        lines.append(f"镜头{shot['shot_id']} | {shot['start']}-{shot['end']}s | {shot['camera']}")
        lines.append(shot["visual"])
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
                           storyboard_image_path: Optional[str] = None) -> str:
    asset_urls = asset_urls or {}
    sections = []

    # Video goal
    sections.append(f"【视频目标】\n{board_page.get('video_goal', '')}")

    # Voice timeline
    voice_lines = []
    for beat in board_page.get("voice_timeline", []):
        voice_lines.append(f"  {beat['start']}-{beat['end']}s {beat['speaker']}：{beat['text']}")
    if voice_lines:
        sections.append("【声音时间轴】\n" + "\n".join(voice_lines))

    # Shot timeline
    shot_lines = []
    for shot in board_page.get("shot_timeline", []):
        shot_lines.append(f"  {shot['start']}-{shot['end']}s 镜头{shot['shot_id']} {shot['camera']}")
        shot_lines.append(f"  画面：{shot['visual']}")
        shot_lines.append(f"  声音设计：{audio_behavior_text(shot['audio_behavior'], shot.get('voice_refs', []), board_page.get('voice_timeline'))}")
        shot_lines.append(f"  画面匹配：{match_strategy_text(shot['match_strategy'])}")
        shot_lines.append(f"  意图：{shot['purpose']}")
        if shot.get("voice_refs"):
            beat_map = {b["beat_id"]: b for b in board_page.get("voice_timeline", [])}
            refs = []
            for ref in shot["voice_refs"]:
                beat = beat_map.get(ref)
                if beat:
                    refs.append(f"{beat['speaker']}：{beat['text']}")
            if refs:
                shot_lines.append(f"  引用声音：{'; '.join(refs)}")
        if shot.get("continuity_from_previous"):
            shot_lines.append(f"  视觉延续：{shot['continuity_from_previous']}")
    if shot_lines:
        sections.append("【镜头执行】\n" + "\n".join(shot_lines))

    # Reference images
    ref_lines = []
    char_urls = asset_urls.get("characters", {})
    asset_refs = board_page.get("asset_refs", {})
    for char in asset_refs.get("characters", []):
        url = char_urls.get(char)
        if url:
            ref_lines.append(f"  角色设定板 {char}：{url}")
    if storyboard_image_path:
        ref_lines.append(f"  分镜板参考图：{storyboard_image_path}")
    scene_url = asset_urls.get("scene")
    if scene_url:
        ref_lines.append(f"  场景参考板：{scene_url}")
    prop_urls = asset_urls.get("props", {})
    for prop in asset_refs.get("props", []):
        url = prop_urls.get(prop)
        if url:
            ref_lines.append(f"  道具参考 {prop}：{url}")
    if ref_lines:
        sections.append("【参考图使用】\n" + "\n".join(ref_lines))

    # Prohibitions
    sections.append(f"【禁止项】\n{PROHIBITIONS}")

    return "\n\n".join(sections)
