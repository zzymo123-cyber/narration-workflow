# Narration-Driven Comic Narration Workstation Design

## Problem Statement

Create a new independent project (cloned from manhua-workflow) for producing Douyin-style comic narration videos. The core paradigm shift: from script-driven storyboard pipeline to narration-driven video production.

Key differences from manhua-workflow:
- **Driving unit**: narration segment (not script scene)
- **Audio**: narration/dialogue text embedded in Seedance video prompt (no separate TTS)
- **Storyboard**: director's storyboard board_page with dual timeline (voice + shot), not 9-panel grid
- **Target**: Douyin vertical 9:16, 1-3 minute videos, 5-10 seconds per segment
- **Output**: Seedance 2.0 video segments with narration audio embedded

## Project Relationship

- Original manhua-workflow: preserved, no changes
- New project: independent clone, reuses only Vidu/Wetoken API client layer from original
- Data model, parser, routes, frontend: all redesigned

## Core Workflow

```
Story / Outline / Idea
  | LLM decomposition (mandatory step)
  v
Narration Segments (narration text + dialogue + visual description)
  |
  v
Asset Production (character sheets, scene refs, prop refs)
  |
  v
Storyboard Board Pages (per narration segment: voice_timeline + shot_timeline)
  | Vidu generates storyboard image
  v
Video Production (Seedance 2.0: storyboard image ref + narration prompt + asset refs)
  |
  v
(P2) Auto-composition into final video
```

Phase dependencies:
- LLM decomposition can run independently (no assets needed)
- Asset production and storyboard production can partially overlap
- Storyboard image -> video is serial (image must exist before video generation)
- Auto-composition is P2

## Data Model: pipeline.json

### Top-level structure

```json
{
  "project": "project name",
  "narration_style": "third_person",
  "source_text": "original story/outline text",

  "assets": {
    "characters": {
      "character_name": {
        "seed": "appearance description",
        "status": "needed|drafted|submitted|completed|failed",
        "task_id": null,
        "result_url": null
      }
    },
    "scenes": { "...": "same structure" },
    "props": { "...": "same structure" }
  },

  "narration_segments": {
    "seg_01_01": {
      "episode": 1,
      "segment_index": 1,
      "characters_in_segment": ["character names"],
      "scene_location": "scene name",
      "boards": [ "see board_page below" ]
    }
  }
}
```

### board_page structure

board_page is the Seedance video generation unit. Max 15 seconds. Contains dual timelines.

```json
{
  "board_id": "seg_01_01_p01",
  "page": 1,
  "total_pages": 1,
  "compact_page": false,
  "voice_duration": 9,
  "visual_duration": 10,
  "board_duration": 10,

  "video_goal": "dramatic/emotional/narrative goal of this board page in natural Chinese",

  "voice_timeline": [
    {
      "beat_id": "v01",
      "type": "narration",
      "text": "narration or dialogue text",
      "speaker": "旁白 or character name",
      "start": 0,
      "end": 4,
      "duration": 4
    }
  ],

  "shot_timeline": [
    {
      "shot_id": "s01",
      "start": 0,
      "end": 2,
      "duration": 2,
      "voice_refs": ["v01"],
      "visual": "visual description of this shot",
      "camera": "camera type",
      "characters": ["character names in this shot"],
      "scene": "scene name",
      "match_strategy": "sync|supplement|contrast|foreshadow|reaction_first|reveal|emotional_landing|transition",
      "purpose": "natural Chinese description of shot intent",
      "audio_behavior": "narration_sync|narration_over|dialogue_sync|dialogue_offscreen|ambient_only|sound_lead_in|dramatic_silence|ambient_transition",
      "continuity_from_previous": "natural Chinese description or null for first shot",
      "transition_type": "cut|match_cut|dissolve|continuous or null for first shot"
    }
  ],

  "storyboard_image": {
    "status": "needed",
    "prompt": "",
    "task_id": null,
    "url": null,
    "local_path": null
  },

  "video": {
    "status": "needed",
    "duration": 10,
    "prompt": "",
    "task_id": null,
    "url": null,
    "local_path": null
  },

  "asset_refs": {
    "characters": ["character names"],
    "scene": "scene name",
    "props": ["prop names"]
  }
}
```

### Key design decisions

1. **voice_timeline + shot_timeline are independent timelines** - voice beats and visual shots are NOT 1:1 bound. One voice beat can span multiple shots. Multiple short beats can share one shot. Shots can have no voice_refs at all (pure visual moments for reaction, foreshadowing, emotional landing, transition).
2. **speaker field** - narration uses speaker="旁白", dialogue uses speaker=character name. No separate "character" field.
3. **No video_parts** - board_page is the video generation unit. Single `video` field replaces video_parts.
4. **No prompt_seed / draft_prompt / plan** - storyboard_image.prompt and video.prompt are the only prompt fields. No legacy prompt fields.
5. **No TTS/audio task** - narration text is embedded in Seedance video prompt. No separate audio generation step.
6. **All time fields are integers** - no decimal seconds anywhere.

### Concrete example

Story input: "A man encounters the woman he thought had left, on a rainy night in an alley."

**voice_timeline:**

| beat_id | type | text | speaker | start | end | duration |
|---------|------|------|---------|-------|-----|----------|
| v01 | narration | 她独自走在雨夜小巷里，手中握着一张泛黄的信纸。 | 旁白 | 0 | 4 | 4 |
| v02 | dialogue | 你怎么来了？ | 阿明 | 4 | 6 | 2 |
| v03 | narration | 她抬起头，看见了他。 | 旁白 | 6 | 9 | 3 |

**shot_timeline:**

| shot_id | start | end | voice_refs | visual | camera | match_strategy | purpose | audio_behavior |
|---------|-------|-----|------------|--------|--------|----------------|---------|----------------|
| s01 | 0 | 2 | [v01] | 雨夜小巷全景，昏暗路灯映着积水 | wide_establishing | sync | 建立雨夜小巷的整体氛围和空间感 | narration_sync |
| s02 | 2 | 4 | [v01] | 林雪低头看手中泛黄信纸，表情迷茫 | medium_close | sync | 展示角色状态和关键道具，建立观众共情 | narration_over |
| s03 | 4 | 5 | [] | 巷口出现人影，脚步声响起 | medium | foreshadow | 用脚步声暗示新角色即将出现，制造悬念 | sound_lead_in |
| s04 | 5 | 6 | [v02] | 阿明的声音响起时，林雪突然停住，肩膀僵住，不露阿明正脸 | medium | reaction_first | 先拍听到台词后的身体反应，制造悬念和压迫感 | dialogue_offscreen |
| s05 | 6 | 9 | [v03] | 林雪缓缓抬头，目光从迷茫转为惊讶 | close_up | reaction_first | 捕捉角色核心情绪转变，让观众代入她的震惊与意外 | narration_over |
| s06 | 9 | 10 | [] | 两人对视，雨声渐大 | two_shot | emotional_landing | 给这场相遇一个情感落点，让观众消化情绪 | ambient_only |

Key observations:
- v01 (4s narration) splits into 2 shots (s01+s02), not 1:1 bound
- s03 has no voice_refs (foreshadow - visual appears before voice)
- s04: dialogue_offscreen with reaction_first - we see Lin Xue's reaction, not A Ming's face
- s06 has no voice_refs (emotional_landing - 1s pure visual closure after narration ends)
- voice_duration = 9s, board_duration = 10s, 1s is pure visual time

## Validation Rules

### Integer seconds (hard)
1. All start, end, duration in voice_timeline must be integer
2. All start, end, duration in shot_timeline must be integer
3. voice_duration, visual_duration, board_duration must be integer

### Duration calculation
4. Narration duration = ceil(Chinese char count / 4.5), min 1s
5. Dialogue duration = ceil(Chinese char count / 3), min 1s
6. Voice beats exceeding 8s should be split
7. board_duration = max(voice_duration, visual_duration)

### Coverage
8. shot_timeline must fully cover [0, board_duration], no gaps, no overlaps
9. First shot: start = 0
10. Last shot: end = board_duration
11. Adjacent shots: shot[i].end == shot[i+1].start
12. Each shot.duration = shot.end - shot.start

### Shot count
13. Regular board_page: 5-6 shots
14. Short tail page (compact_page=true): 3-4 shots
15. Regular pages must not have fewer than 5 shots

### Boundary
16. board_duration <= 15
17. voice_duration <= board_duration

### References
18. shot.voice_refs values must exist as beat_ids in voice_timeline
19. Every beat_id in voice_timeline must be referenced by at least one shot (no orphan beats)

### match_strategy soft validation
20. sync: typically needs non-empty voice_refs
21. supplement: typically needs non-empty voice_refs
22. contrast: typically needs non-empty voice_refs
23. foreshadow: voice_refs optional
24. reaction_first: voice_refs optional
25. reveal: voice_refs optional
26. emotional_landing: typically no voice_refs, but may reference previous beat
27. transition: typically no voice_refs

### audio_behavior rules
28. audio_behavior must be one of: narration_sync, narration_over, dialogue_sync, dialogue_offscreen, ambient_only, sound_lead_in, dramatic_silence, ambient_transition
29. No BGM/background music/score
30. Only narration, dialogue, ambient sound, and necessary action SFX
31. Shots without narration or dialogue: only ambient sound or specified SFX

### Shot field completeness
32. Every shot must contain all fields: shot_id, start, end, duration, voice_refs, visual, camera, characters, scene, match_strategy, purpose, audio_behavior, continuity_from_previous, transition_type
33. First shot: continuity_from_previous = null, transition_type = null

### purpose
34. purpose must be natural Chinese text describing shot intent, not English enum

### video_goal
35. video_goal must be natural Chinese text describing dramatic/emotional/narrative goal
36. video_goal must not be empty

## audio_behavior_text Mapping

| audio_behavior | Prompt text |
|---|---|
| narration_sync | 旁白与当前画面同步，画面直接承载旁白内容 |
| narration_over | 旁白覆盖在画面上，画面可以补充、反差或伏笔，不要机械复述旁白 |
| dialogue_sync | 角色台词与当前画面同步，可以看到说话角色或其明确动作 |
| dialogue_offscreen | 角色台词画外响起，说话者不立即露面，先拍听到台词后的反应 |
| ambient_only | 只有环境声，没有旁白和台词 |
| sound_lead_in | 非语言声音先行，用来引出下一镜头或人物 |
| dramatic_silence | 戏剧性静默，用于震惊反应、冲突后停顿、压迫感 |
| ambient_transition | 环境声作为转场桥 |

Assembly rule: When dialogue_offscreen and voice_refs references a character's dialogue, inject the character name. Example: "阿明台词画外响起，说话者不立即露面，先拍听到台词后的反应"。

## match_strategy_text Mapping

| match_strategy | Prompt text |
|---|---|
| sync | 画面直接呈现旁白或台词描述的内容 |
| supplement | 画面补充旁白或台词未说出的信息 |
| contrast | 画面与旁白或台词形成反差 |
| foreshadow | 画面暗示尚未发生的事件 |
| reaction_first | 先拍角色反应，再揭示说话者或信息来源 |
| reveal | 画面揭示之前铺垫的信息 |
| emotional_landing | 画面提供情感落点，让观众消化情绪 |
| transition | 画面作为场景或情绪的过渡桥 |

## Seedance Prompt Template

```
【视频目标】
{video_goal}

【声音时间轴】
{for beat in voice_timeline}:
  {beat.start}-{beat.end}s {beat.speaker}：{beat.text}

【镜头执行】
{for shot in shot_timeline}:
  {shot.start}-{shot.end}s 镜头{shot.shot_id} {shot.camera}
  画面：{shot.visual}
  声音设计：{audio_behavior_text(shot.audio_behavior, shot.voice_refs)}
  画面匹配：{match_strategy_text(shot.match_strategy)}
  意图：{shot.purpose}
  {if shot.voice_refs non-empty}引用声音：{referenced beat speaker + text}{endif}
  {if shot.continuity_from_previous}视觉延续：{shot.continuity_from_previous}{endif}

【参考图使用】
{for char in asset_refs.characters where result_url exists}:
  角色设定板 {char}：{result_url}
分镜板参考图：{storyboard_image local_path or url}
{if scene result_url exists}场景参考板：{scene result_url}{endif}
{for prop in asset_refs.props where result_url exists}:
  道具参考 {prop}：{result_url}{endfor}

【禁止项】
- 不要生成字幕
- 不要生成额外文字
- 不要生成 BGM、背景音乐、配乐、氛围音乐
- 只允许旁白、角色台词、环境声和必要动作音效
- 无旁白无台词的镜头，只允许环境声或指定音效，不要用音乐填充
- 不要改变角色长相、发型、服装
- 不要让人物、道具、空间位置在镜头之间突然跳变
- 分镜板参考图只作为镜头顺序、构图关系和人物位置参考，不要把分镜板上的编号、时间码、说明文字生成进视频画面
```

Reference image rules (MVP):
- Default: character sheets + storyboard image only
- Scene reference: optional, only include if result_url exists
- Prop reference: optional, only include if result_url exists
- Never declare non-existent references in prompt

## Storyboard Image Prompt Template

```
导演分镜板
总时长：{board_duration}s

{for shot in shot_timeline}:
  镜头{shot.shot_id} | {shot.start}-{shot.end}s | {shot.camera}
  {shot.visual}

风格：黑白铅笔线稿/导演分镜稿
版面：竖屏9:16，{len(shot_timeline)}格连续分镜，竖屏规整排版
每格包含镜头编号、整数秒时间码、简短画面说明
不要完整旁白
不要完整台词
不要大量中文小字
保持角色、场景、道具一致
每格画面要清楚表达镜头内容、角色位置和情绪变化
```

## What This Project Does NOT Include

- No separate TTS/audio task generation
- No video_parts (replaced by single video field)
- No prompt_seed / draft_prompt / plan (replaced by storyboard_image.prompt and video.prompt)
- No BGM / background music / score generation
- No storyboard v1 nine-grid layout (only director_sheet vertical format)
- No auto-composition of final video (P2)

## Narration Styles (MVP)

- third_person: Third-person narration (旁白 as observer)
- first_person: First-person narration (protagonist as speaker)

Extensible: additional styles can be added later without changing data model.

## Asset Production Flow

Character/scene/prop reference boards are generated before storyboard and video production. This ensures visual consistency across segments.

Flow:
1. LLM decomposition identifies characters, scenes, props from the story
2. Generate character sheets (Vidu) → result used as Seedance reference images
3. Generate scene reference boards (Vidu, optional) → used as Seedance reference
4. Generate prop references (Vidu, optional) → used as Seedance reference
5. Storyboard image generation uses asset_refs but does not require all assets to be completed
6. Video generation requires: storyboard image + at minimum character sheets

## LLM Decomposition

The LLM decomposition step is mandatory. It takes a story/outline/idea and produces:
- narration_segments with voice_timeline and shot_timeline
- Character/scene/prop identification for assets
- Duration calculations (narration 4.5 chars/sec, dialogue 3 chars/sec)
- Shot-to-voice mapping with match_strategy and audio_behavior
- video_goal for each board_page

Input: free-form text (story, outline, or just an idea)
Output: fully structured pipeline.json with narration_segments, assets, and board_pages

The user can adjust each segment's content before entering production.
