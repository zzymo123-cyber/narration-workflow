"""Multi-step decomposition prompts for narration-workflow.

Step 1: Extract assets (characters, scenes, props) from story
Step 2: Generate segment outline (scene breaks, character assignments, board counts)
Step 3: Generate board pages for one segment at a time
"""

STYLE_INSTRUCTIONS = {
    "third_person": '使用第三人称旁白，旁白者speaker固定为"旁白"。角色台词保持原话，speaker为角色名。',
    "first_person": '使用第一人称旁白，旁白者为主角内心独白，speaker为角色名。其他角色台词speaker为角色名。',
}

JSON_INSTRUCTION = "只输出纯 JSON，不要用 markdown 代码块，不要解释，不要注释。直接以 { 开头，以 } 结尾。"


# ═══════════════════════════════════════════════════════════════
# Step 1: Extract assets
# ═══════════════════════════════════════════════════════════════

STEP1_SYSTEM_PROMPT = """你是解说漫剧本拆解专家。你的任务是从故事中提取所有角色、场景和道具。

## 旁白风格
{style_instruction}

## 输出格式
{json_instruction}

输出结构：
{{
  "characters": {{ "角色名": {{ "seed": "外貌描述（用于AI生图的prompt片段）" }} }},
  "scenes": {{ "场景名": {{ "seed": "场景视觉描述（用于AI生图的prompt片段）" }} }},
  "props": {{ "道具名": {{ "seed": "道具视觉描述（用于AI生图的prompt片段）" }} }}
}}

## 硬规则
1. 必须提取故事中出现的所有角色，不得遗漏
2. 必须提取故事中出现的所有场景，不得遗漏
3. 必须提取故事中出现的所有关键道具，不得遗漏
4. seed 描述要具体、视觉化，能用于 AI 生图
5. 角色名用故事中的名字，不要改名"""


def build_step1_prompt(style: str = "third_person") -> str:
    style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["third_person"])
    return STEP1_SYSTEM_PROMPT.format(style_instruction=style_instruction, json_instruction=JSON_INSTRUCTION)


# ═══════════════════════════════════════════════════════════════
# Step 2: Generate segment outline
# ═══════════════════════════════════════════════════════════════

STEP2_SYSTEM_PROMPT = """你是解说漫剧本拆解专家。你的任务是将故事按场景转换切分为多个旁白段落，并规划每段需要多少个视频板(board)。

## 旁白风格
{style_instruction}

## 已提取的资产
{assets_json}

## 固定声音预算
{script_plan_json}

## 输出格式
{json_instruction}

输出结构：
{{
  "segments": {{
    "seg_EP_SEG": {{
      "episode": INTEGER,
      "segment_index": INTEGER,
      "scene_location": "场景名（必须来自上方资产库）",
      "characters_in_segment": ["角色名（必须来自上方资产库）"],
      "num_boards": INTEGER
    }}
  }}
}}

## 编号规则
- 格式: seg_EP_SEG，EP是集数，SEG是该集内的段落序号
- 例: 第一集第一个段落 = seg_1_1，第一集第二个段落 = seg_1_2

## 硬规则
1. 故事每转换一次场景，就开一个新段落
2. 同一场景内的连续情节归入同一段落
3. 每个 board 最长 15 秒，旁白按 4.5 个中文/秒、对白按 3 个中文/秒计算
4. 必须优先遵守上方固定声音预算；所有段落 num_boards 之和必须等于 fixed_total_boards
5. 不要自行压缩台词/旁白秒数；如果场景段落跨多个 fixed boards，就把这些 boards 分配给同一段
6. 每个 board 最多承载约 55 个中文旁白字；超过必须增加 num_boards，不允许硬塞
7. num_boards 必须按本段实际文本长度估算，允许 1-8 个 board，不要固定写 2-3
8. 必须覆盖故事的全部内容，不得跳过任何情节
9. 一般故事至少 3-5 个段落
10. characters_in_segment 中的角色名必须来自上方资产库
11. scene_location 必须来自上方资产库中的场景"""


def build_step2_prompt(style: str, assets: dict, script_plan: dict | None = None) -> str:
    import json as _json
    style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["third_person"])
    assets_json = _json.dumps(assets, ensure_ascii=False, indent=2)
    board_plan = (script_plan or {}).get("board_plan", [])
    script_plan_json = _json.dumps({
        "fixed_total_boards": len(board_plan),
        "board_plan": [
            {
                "board_id": item.get("board_id"),
                "duration": item.get("duration"),
                "voice_beat_ids": item.get("voice_beat_ids", []),
                "source_slice_ids": item.get("source_slice_ids", []),
            }
            for item in board_plan
        ],
    }, ensure_ascii=False, indent=2) if board_plan else "未提供。按硬规则自行估算，但仍要满足 15 秒可朗读限制。"
    return STEP2_SYSTEM_PROMPT.format(
        style_instruction=style_instruction,
        assets_json=assets_json,
        script_plan_json=script_plan_json,
        json_instruction=JSON_INSTRUCTION,
    )


# ═══════════════════════════════════════════════════════════════
# Step 3: Generate boards for one segment
# ═══════════════════════════════════════════════════════════════

STEP3_SYSTEM_PROMPT = """你是解说漫剧本拆解专家。你的任务是为一个旁白段落生成所有视频板(board_page)的详细内容。

## 旁白风格
{style_instruction}

## 本段信息
- 段落ID: {seg_key}
- 场景: {scene_location}
- 出场角色: {characters}
- 规划板数: {num_boards}

## 可用资产
{assets_json}

## 输出格式
{json_instruction}

输出结构：
{{
  "boards": [BOARD_PAGE, BOARD_PAGE, ...]
}}

## BOARD_PAGE 格式
每个 board_page 是一个 Seedance 视频生成单位，最长 15 秒：

{{
  "board_id": "{seg_key}_p01",
  "page": 1,
  "total_pages": {num_boards},
  "compact_page": false,
  "voice_duration": INTEGER,
  "visual_duration": INTEGER,
  "board_duration": INTEGER,
  "video_goal": "本页视频的戏剧/情绪/叙事目标",
  "palette_id": "suspense_cold_blue/family_warm_gray/hospital_cold_white/domestic_brown_gray 中选择一个",
  "voice_timeline": [BEAT],
  "shot_timeline": [SHOT],
  "storyboard_image": {{ "status": "needed", "prompt": "", "task_id": null, "url": null, "local_path": null }},
  "video": {{ "status": "needed", "duration": INTEGER, "prompt": "", "task_id": null, "url": null, "local_path": null }},
  "asset_refs": {{ "characters": [], "scene": "", "props": [] }}
}}

## BEAT 格式
{{
  "beat_id": "v01",
  "type": "narration 或 dialogue",
  "text": "旁白或台词文本",
  "speaker": "旁白（narration）或角色名（dialogue）",
  "start": INTEGER,
  "end": INTEGER,
  "duration": INTEGER
}}

时长计算（整数秒）：
- 旁白：ceil(中文字数 / 4.5)，下限 1 秒
- 对白：ceil(中文字数 / 3)，下限 1 秒
- 每个 beat 的 duration 必须等于 end-start，且不得小于上面的计算结果
- voice_duration 必须等于所有 beat.duration 之和
- 如果本页声音内容按规则会超过 15 秒，必须把内容移到下一页，不允许压缩秒数
- 超过 8 秒的 beat 必须拆分成多个 beat

## SHOT 格式（每个常规页必须 5-6 个 shot，不可少于 5 个！）

{{
  "shot_id": "s01",
  "start": INTEGER,
  "end": INTEGER,
  "duration": INTEGER,
  "voice_refs": ["beat_id"],
  "visual": "画面描述",
  "camera": "镜头类型",
  "characters": ["角色名"],
  "scene": "场景名",
  "match_strategy": "见下方枚举",
  "purpose": "中文自然语言描述镜头意图",
  "audio_behavior": "见下方枚举",
  "continuity_from_previous": "中文描述或 null（首个镜头必须 null）",
  "transition_type": "cut/match_cut/dissolve/continuous 或 null（首个镜头必须 null）",
  "technique_id": "可为空；只能从 slow_push_in/dolly_zoom/crash_zoom/rack_focus/slow_motion/handheld_follow/orbit_360/pov_shot/dutch_angle/frame_within_frame 中选择"
}}

## match_strategy 枚举
- sync: 画面直接呈现旁白或台词内容
- supplement: 画面补充未说出的信息
- contrast: 画面与旁白/台词形成反差
- foreshadow: 画面暗示尚未发生的事件
- reaction_first: 先拍角色反应，再揭示说话者
- reveal: 画面揭示之前铺垫的信息
- emotional_landing: 画面提供情感落点
- transition: 画面作为场景/情绪过渡

## audio_behavior 枚举
- narration_sync: 旁白与画面同步
- narration_over: 旁白覆盖画面，画面可补充/反差
- dialogue_sync: 台词与画面同步
- dialogue_offscreen: 台词画外响起，说话者不露面
- ambient_only: 只有环境声
- sound_lead_in: 非语言声音先行
- dramatic_silence: 戏剧性静默
- ambient_transition: 环境声转场

## 硬规则
1. 所有时间字段必须是整数，不允许小数
2. board_duration <= 15，且 voice_duration <= board_duration
3. 【必须】常规页（compact_page=false）必须生成 5-6 个 shot，不可少于 5 个！
4. shot_timeline 必须完整覆盖 0 到 board_duration，无间隙无重叠
5. voice_timeline 和 shot_timeline 不是一一绑定
6. 允许 voice_refs 为空的镜头（伏笔、反应、情感落点、转场）
7. 第一个 shot 的 continuity_from_previous 和 transition_type 必须为 null
8. 禁止生成 BGM/背景音乐，只允许旁白、台词、环境声、动作音效
9. video_goal 不能为空
10. purpose 使用中文自然语言，不要用英文枚举
11. 必须覆盖本段落的全部内容，不得跳过
12. 不允许出现“15 秒内念完明显超过 70 个中文”的 board；遇到长文本就增加页数或压缩旁白文本
13. palette_id 必填；同一场戏内优先保持统一
14. technique_id 是可选镜头技法宏；每个 board 最多 1-2 个非空，普通交代镜头留空"""


def build_step3_prompt(
    style: str,
    seg_key: str,
    scene_location: str,
    characters: list,
    num_boards: int,
    assets: dict,
) -> str:
    import json as _json
    style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["third_person"])
    assets_json = _json.dumps(assets, ensure_ascii=False, indent=2)
    return STEP3_SYSTEM_PROMPT.format(
        style_instruction=style_instruction,
        seg_key=seg_key,
        scene_location=scene_location,
        characters=", ".join(characters),
        num_boards=num_boards,
        assets_json=assets_json,
        json_instruction=JSON_INSTRUCTION,
    )


# ═══════════════════════════════════════════════════════════════
# Parsers (unchanged logic, just keep backward compat)
# ═══════════════════════════════════════════════════════════════

def parse_decomposition_response(response: dict) -> dict:
    """Parse and return the decomposition response as pipeline.json structure.
    Adds project-level fields that the LLM doesn't generate."""
    return {
        "project": "",
        "narration_style": response.get("narration_style", "third_person"),
        "source_text": "",
        "assets": response.get("assets", {"characters": {}, "scenes": {}, "props": {}}),
        "narration_segments": response.get("narration_segments", {}),
    }
