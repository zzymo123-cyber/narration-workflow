import re
from datetime import datetime
from pathlib import Path


ASSET_TYPES = {"characters", "scenes", "props"}
AUDIO_REF_EXTS = {".mp3", ".wav", ".m4a", ".aac"}
BGM_REF_EXTS = AUDIO_REF_EXTS
MAX_HISTORY_ITEMS = 30
DEFAULT_STYLE_PALETTES = {
    "suspense_cold_blue",
    "family_warm_gray",
    "hospital_cold_white",
    "domestic_brown_gray",
}

PALETTE_RULES = [
    (
        "hospital_cold_white",
        ("医院", "病房", "诊室", "手术", "护士", "医生", "办公室", "荧光灯", "冷白"),
    ),
    (
        "domestic_brown_gray",
        ("压抑", "饭桌", "旧屋", "争吵", "家庭矛盾", "沉闷", "中年", "旧旧", "暗黄"),
    ),
    (
        "suspense_cold_blue",
        ("悬疑", "惊悚", "犯罪", "夜", "深夜", "黑暗", "阴影", "异常", "惊醒", "危险", "恐惧", "逼近", "密闭"),
    ),
    (
        "family_warm_gray",
        ("家庭", "日常", "生活", "对白", "温柔", "饭菜", "客厅", "厨房"),
    ),
]

TECHNIQUE_RULES = [
    ("slow_push_in", ("缓慢", "靠近", "逼近", "推近", "推进", "探出", "爬行", "压迫")),
    ("pov_shot", ("第一人称", "主观", "视角", "眼前", "低头看", "看向")),
    ("dolly_zoom", ("纵深", "背景拉长", "空间压缩", "眩晕", "不真实")),
    ("frame_within_frame", ("门缝", "窗框", "框住", "框内", "玻璃", "开口", "饲养箱")),
    ("rack_focus", ("聚焦", "焦点", "虚化", "前景", "背景", "由模糊到清晰")),
    ("crash_zoom", ("猛然", "突然", "快速", "瞪大", "惊醒", "冲", "扑", "扫过")),
    ("handheld_follow", ("跟拍", "手持", "晃动", "追随", "奔跑", "跟随")),
    ("orbit_360", ("环绕", "绕行", "旋转", "一圈")),
    ("slow_motion", ("慢动作", "放慢", "凝固", "悬停")),
    ("dutch_angle", ("倾斜", "失衡", "错位", "歪斜")),
]


def _is_snake_character_asset(asset_name: str, seed: str = "") -> bool:
    if asset_name == "玄墨":
        return True
    text = seed or ""
    snake_body_patterns = (
        r"(纯黑|成年|幼年|巨型|巨大|体型|体长|鳞片|盘踞|蜷缩|爬行|无四肢|蛇类体态).{0,10}(蟒|蛇)",
        r"(蟒|蛇).{0,10}(纯黑|成年|幼年|巨型|巨大|体型|体长|鳞片|盘踞|蜷缩|爬行|无四肢|蛇类体态)",
        r"(一条|同一条).{0,8}(蟒|蛇)",
    )
    return any(re.search(pattern, text) for pattern in snake_body_patterns)


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", name.strip())
    return cleaned.strip("_") or "asset"


def asset_output_path(project_dir: Path, asset_type: str, asset_name: str) -> Path:
    if asset_type not in ASSET_TYPES:
        raise ValueError(f"Unsupported asset_type: {asset_type}")
    return project_dir / asset_type / f"{safe_filename(asset_name)}.jpg"


def storyboard_output_path(project_dir: Path, segment_key: str, board_index: int) -> Path:
    return project_dir / "storyboards" / f"{safe_filename(segment_key)}_p{board_index + 1:02d}.jpg"


def board_display_id(board: dict, fallback_index: int | None = None) -> str:
    try:
        index = int(board.get("global_board_index") or board.get("page") or fallback_index or 0)
    except (TypeError, ValueError):
        index = int(fallback_index or 0)
    if index > 0:
        return f"P{index:03d}"
    return "P---"


def board_output_stem(board: dict, segment_key: str, board_index: int) -> str:
    display = board_display_id(board, board_index + 1).lower()
    board_id = safe_filename(str(board.get("board_id") or display))
    return f"{display}_{safe_filename(segment_key)}_{board_id}"


def video_output_path(project_dir: Path, board: dict, segment_key: str, board_index: int) -> Path:
    return project_dir / "videos" / f"{board_output_stem(board, segment_key, board_index)}.mp4"


def normalize_video_output_paths(project_dir: Path, data: dict) -> bool:
    """Keep completed video local_path aligned with the stable output_path."""
    changed = False
    for seg_key, seg in data.get("narration_segments", {}).items():
        for board_index, board in enumerate(seg.get("boards", [])):
            video = board.get("video")
            if not isinstance(video, dict):
                continue

            output_path = video.get("output_path")
            if not output_path:
                output_path = str(video_output_path(project_dir, board, seg_key, board_index))
                video["output_path"] = output_path
                changed = True
            expected = Path(output_path)

            if expected.exists() and video.get("local_path") != str(expected):
                video["status"] = "completed"
                video["local_path"] = str(expected)
                changed = True
                continue

            local_path = video.get("local_path")
            if (
                video.get("status") == "completed"
                and local_path
                and not str(local_path).startswith(("http://", "https://"))
            ):
                actual = Path(local_path)
                if actual.exists() and actual != expected:
                    expected.parent.mkdir(parents=True, exist_ok=True)
                    if not expected.exists():
                        actual.replace(expected)
                    video["local_path"] = str(expected)
                    changed = True
    return changed


def append_generation_history(target: dict, event: str, **fields) -> dict:
    history = target.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        target["history"] = history
    entry = {
        "event": event,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    for key, value in fields.items():
        if value is not None:
            entry[key] = value
    history.append(entry)
    if len(history) > MAX_HISTORY_ITEMS:
        del history[:-MAX_HISTORY_ITEMS]
    return entry


def _segment_sort_key(seg_key: str) -> tuple:
    numbers = [int(part) for part in re.findall(r"\d+", seg_key)]
    return (*numbers, seg_key)


def sync_board_metadata(data: dict) -> dict:
    segments = data.get("narration_segments", {})
    total = sum(len((seg or {}).get("boards", [])) for seg in segments.values())
    global_index = 1
    for seg_key in sorted(segments.keys(), key=_segment_sort_key):
        seg = segments.get(seg_key) or {}
        boards = seg.get("boards", [])
        for board_index, board in enumerate(boards):
            page = board.get("page")
            try:
                page_index = int(page)
            except (TypeError, ValueError):
                page_index = global_index
                board["page"] = page_index
            board["total_pages"] = int(board.get("total_pages") or total or len(boards) or 1)
            board["global_board_index"] = page_index
            board["segment_key"] = seg_key
            board["segment_board_index"] = board_index + 1
            board["display_id"] = board_display_id(board, global_index)
            board["segment_display_id"] = f"{seg_key} P{board_index + 1:02d}"
            if not board.get("board_id"):
                board["board_id"] = f"b{page_index:04d}"
            if isinstance(board.get("video"), dict):
                board["video"].setdefault("duration", board.get("board_duration", 10))
            global_index += 1
    return data


def _board_style_text(board: dict) -> str:
    parts = [
        board.get("video_goal", ""),
        board.get("palette_hint", ""),
        board.get("scene", ""),
    ]
    refs = board.get("asset_refs") or {}
    parts.append(refs.get("scene", ""))
    for shot in board.get("shot_timeline", []) or []:
        parts.extend([
            shot.get("visual", ""),
            shot.get("camera", ""),
            shot.get("purpose", ""),
        ])
    return "\n".join(str(part) for part in parts if part)


def infer_board_palette_id(board: dict) -> str:
    existing = str(board.get("palette_id") or "").strip()
    if existing in DEFAULT_STYLE_PALETTES:
        return existing
    text = _board_style_text(board)
    for palette_id, keywords in PALETTE_RULES:
        if any(keyword in text for keyword in keywords):
            return palette_id
    return ""


def infer_shot_technique_id(shot: dict) -> str:
    existing = str(shot.get("technique_id") or "").strip()
    if existing:
        return existing
    text = "\n".join(
        str(shot.get(field) or "")
        for field in ("visual", "camera", "purpose")
    )
    for technique_id, keywords in TECHNIQUE_RULES:
        if any(keyword in text for keyword in keywords):
            return technique_id
    return ""


def apply_default_visual_macros(data: dict) -> bool:
    """Backfill optional video-only style macros for older board data."""
    changed = False
    for seg in data.get("narration_segments", {}).values():
        boards = seg.get("boards", []) or []
        segment_palette_id = ""
        for board in boards:
            palette_id = infer_board_palette_id(board)
            if palette_id:
                segment_palette_id = palette_id
                break

        for board in boards:
            if not str(board.get("palette_id") or "").strip() and segment_palette_id:
                board["palette_id"] = segment_palette_id
                changed = True

        for board in boards:
            if not str(board.get("palette_id") or "").strip():
                palette_id = infer_board_palette_id(board)
                if palette_id:
                    board["palette_id"] = palette_id
                    changed = True

            used = sum(
                1
                for shot in board.get("shot_timeline", []) or []
                if str(shot.get("technique_id") or "").strip()
            )
            for shot in board.get("shot_timeline", []) or []:
                if "technique_id" not in shot:
                    shot["technique_id"] = ""
                    changed = True
                if used >= 2 or str(shot.get("technique_id") or "").strip():
                    continue
                technique_id = infer_shot_technique_id(shot)
                if technique_id:
                    shot["technique_id"] = technique_id
                    used += 1
                    changed = True
    return changed


def asset_prompt(asset_type: str, asset_name: str, asset_info: dict) -> str:
    seed = asset_info.get("seed", "") if isinstance(asset_info, dict) else ""
    if asset_type == "characters":
        title = "角色设定板"
        if _is_snake_character_asset(asset_name, seed):
            guidance = (
                "四视图蛇类角色参考板：俯视全身、左侧全身、右侧全身、头部近景。"
                "四个视图必须是同一条玄墨，纯黑鳞片和暗紫金属光泽一致。"
                "全片只呈现一条蛇，禁止第二条蛇、复制蛇、蛇群或镜像蛇。"
                "必须保持真实蟒蛇体态：无四肢、无爪、无外耳、无角，禁止画成蜥蜴、龙或四脚爬行动物。"
                "不要人物手部、不要笼子、不要道具、不要环境干扰。"
            )
        else:
            guidance = (
                "四视图真人角色参考板：正面全身、左侧全身、右侧全身、背面全身，附一个自然头像细节。"
                "所有视图必须是同一角色，五官、发型、体型、服装和年龄一致。"
                "中性站姿，双手自然下垂或放在身体两侧；手里不要拿任何东西，不要道具、包、手机、食物、武器或宠物。"
                "设定中的剧情道具只用于理解身份，不要画出任何手持物或旁边道具。"
                "干净浅灰背景，均匀光线，便于后续视频模型识别。"
            )
    elif asset_type == "scenes":
        title = "场景参考板"
        guidance = "清楚呈现场景空间结构、光线、关键陈设、气氛和镜头可用角度。"
    elif asset_type == "props":
        title = "道具参考板"
        guidance = "单个主要道具清晰居中，展示材质、尺度、磨损、颜色和可识别特征。"
    else:
        raise ValueError(f"Unsupported asset_type: {asset_type}")

    return "\n".join([
        title,
        f"名称：{asset_name}",
        f"设定：{seed}",
        guidance,
        "风格：写实电影感概念设计，适合作为后续分镜和视频生成参考图。",
        "不要文字，不要水印，不要排版标题，不要多余角色。",
    ])


def normalize_assets(data: dict) -> dict:
    assets = data.setdefault("assets", {})
    for asset_type in ASSET_TYPES:
        bucket = assets.setdefault(asset_type, {})
        for name, info in list(bucket.items()):
            if not isinstance(info, dict):
                info = {"seed": str(info)}
                bucket[name] = info
            info.setdefault("status", "needed")
            info.setdefault("prompt", asset_prompt(asset_type, name, info))
            info.setdefault("task_id", None)
            info.setdefault("url", None)
            info.setdefault("local_path", None)
            info.setdefault("error", None)
    return data


def ensure_referenced_assets(data: dict) -> dict:
    assets = data.setdefault("assets", {})
    characters = assets.setdefault("characters", {})
    scenes = assets.setdefault("scenes", {})
    props = assets.setdefault("props", {})

    for seg in data.get("narration_segments", {}).values():
        for board in seg.get("boards", []):
            refs = board.get("asset_refs", {}) or {}
            for name in refs.get("characters", []) or []:
                characters.setdefault(name, {"seed": f"故事中出现的角色：{name}"})
            scene = refs.get("scene")
            if scene:
                scenes.setdefault(scene, {"seed": f"故事中出现的场景：{scene}"})
            for name in refs.get("props", []) or []:
                props.setdefault(name, {"seed": f"故事中出现的关键道具：{name}"})

    return normalize_assets(data)


def ensure_audio_refs(data: dict, project_dir: Path | None = None) -> dict:
    """Keep project-level narrator voice options in pipeline data."""
    refs = data.setdefault("audio_refs", {})
    options = refs.setdefault("options", {})
    refs.setdefault("selected", "")

    search_dirs = []
    if project_dir:
        search_dirs.append(project_dir / "audio_refs")
    desktop = Path.home() / "Desktop"
    search_dirs.extend([desktop / "未命名文件夹", desktop / "audio_refs"])

    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if not path.is_file() or path.suffix.lower() not in AUDIO_REF_EXTS:
                continue
            name = path.stem
            info = options.setdefault(name, {"name": name})
            info["name"] = name
            info["local_path"] = str(path)
            info.setdefault("asset_uri", None)
            info.setdefault("status", "local")
            info.setdefault("role", "narration")
    if not refs.get("selected") and len(options) == 1:
        refs["selected"] = next(iter(options.keys()))
    if refs.get("selected") and refs["selected"] not in options:
        refs["selected"] = ""
    return data


def ensure_bgm_refs(data: dict, project_dir: Path | None = None) -> dict:
    refs = data.setdefault("bgm_refs", {})
    options = refs.setdefault("options", {})
    refs.setdefault("selected", "")
    refs.setdefault("volume", 0.18)

    search_dirs = []
    if project_dir:
        search_dirs.append(project_dir / "bgm")
    desktop = Path.home() / "Desktop"
    search_dirs.append(desktop / "bgm")

    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if not path.is_file() or path.suffix.lower() not in BGM_REF_EXTS:
                continue
            name = path.stem
            info = options.setdefault(name, {"name": name})
            info["name"] = name
            info["local_path"] = str(path)
            info.setdefault("status", "local")
            info.setdefault("role", "bgm")
    if refs.get("selected") and refs["selected"] not in options:
        refs["selected"] = ""
    return data


def selected_audio_ref(data: dict) -> dict | None:
    refs = data.get("audio_refs", {})
    selected = refs.get("selected")
    if not selected:
        return None
    info = refs.get("options", {}).get(selected)
    if not isinstance(info, dict):
        return None
    path = info.get("asset_uri") or info.get("local_path") or info.get("url")
    if not path:
        return None
    return {"name": selected, **info, "path": path}


def selected_bgm_ref(data: dict) -> dict | None:
    refs = data.get("bgm_refs", {})
    selected = refs.get("selected")
    if not selected:
        return None
    info = refs.get("options", {}).get(selected)
    if not isinstance(info, dict):
        return None
    path = info.get("local_path") or info.get("url")
    if not path:
        return None
    volume = refs.get("volume", 0.18)
    try:
        volume = max(0.0, min(1.0, float(volume)))
    except (TypeError, ValueError):
        volume = 0.18
    return {"name": selected, **info, "path": path, "volume": volume}


def _preferred_ref(info: dict) -> str | None:
    if not isinstance(info, dict):
        return None
    return info.get("local_path") or info.get("url")


def _character_ref_priority(assets: dict, name: str) -> tuple[int, str]:
    info = assets.get("characters", {}).get(name, {})
    seed = info.get("seed", "") if isinstance(info, dict) else ""
    if _is_snake_character_asset(name, seed):
        return (0, name)
    return (1, name)


def _asset_ref_note(asset_type: str, name: str, info: dict) -> str:
    seed = info.get("seed", "") if isinstance(info, dict) else ""
    if asset_type == "characters":
        note = f"角色参考图：{name}。{seed}".strip()
        if _is_snake_character_asset(name, seed):
            note += " 这是蛇类/蟒蛇角色，必须无四肢、无爪、无外耳，禁止画成蜥蜴、龙或四脚爬行动物。"
        return note
    if asset_type == "scenes":
        return f"场景参考图：{name}。{seed}".strip()
    return f"道具参考图：{name}。{seed}".strip()


def collect_asset_reference_items(data: dict, board: dict) -> list[dict]:
    assets = data.get("assets", {})
    items = []
    asset_refs = board.get("asset_refs", {})

    characters = sorted(asset_refs.get("characters", []), key=lambda name: _character_ref_priority(assets, name))
    for char in characters:
        info = assets.get("characters", {}).get(char, {})
        ref = _preferred_ref(info)
        if ref:
            items.append({
                "asset_type": "characters",
                "asset_name": char,
                "path": ref,
                "note": _asset_ref_note("characters", char, info),
            })

    scene = asset_refs.get("scene")
    if scene:
        info = assets.get("scenes", {}).get(scene, {})
        ref = _preferred_ref(info)
        if ref:
            items.append({
                "asset_type": "scenes",
                "asset_name": scene,
                "path": ref,
                "note": _asset_ref_note("scenes", scene, info),
            })

    for prop in asset_refs.get("props", []):
        info = assets.get("props", {}).get(prop, {})
        ref = _preferred_ref(info)
        if ref:
            items.append({
                "asset_type": "props",
                "asset_name": prop,
                "path": ref,
                "note": _asset_ref_note("props", prop, info),
            })

    return items


def collect_asset_reference_images(data: dict, board: dict) -> list[str]:
    return [item["path"] for item in collect_asset_reference_items(data, board)]


def collect_character_reference_images(data: dict, board: dict) -> list[str]:
    assets = data.get("assets", {})
    refs = []
    asset_refs = board.get("asset_refs", {})

    characters = sorted(asset_refs.get("characters", []), key=lambda name: _character_ref_priority(assets, name))
    for char in characters:
        ref = _preferred_ref(assets.get("characters", {}).get(char, {}))
        if ref:
            refs.append(ref)

    return refs


def missing_asset_references(data: dict, board: dict) -> list[str]:
    assets = data.get("assets", {})
    asset_refs = board.get("asset_refs", {})
    missing = []

    for char in asset_refs.get("characters", []):
        if not _preferred_ref(assets.get("characters", {}).get(char, {})):
            missing.append(f"角色:{char}")

    scene = asset_refs.get("scene")
    if scene and not _preferred_ref(assets.get("scenes", {}).get(scene, {})):
        missing.append(f"场景:{scene}")

    for prop in asset_refs.get("props", []):
        if not _preferred_ref(assets.get("props", {}).get(prop, {})):
            missing.append(f"道具:{prop}")

    return missing


def missing_character_references(data: dict, board: dict) -> list[str]:
    assets = data.get("assets", {})
    asset_refs = board.get("asset_refs", {})
    missing = []

    for char in asset_refs.get("characters", []):
        if not _preferred_ref(assets.get("characters", {}).get(char, {})):
            missing.append(f"角色:{char}")

    return missing


def collect_video_reference_images(data: dict, board: dict) -> list[str]:
    refs = []
    storyboard_ref = _preferred_ref(board.get("storyboard_image", {}))
    if storyboard_ref:
        refs.append(storyboard_ref)
    refs.extend(collect_character_reference_images(data, board))
    return refs


def collect_asset_urls(data: dict, board: dict) -> dict:
    assets = data.get("assets", {})
    asset_refs = board.get("asset_refs", {})
    result = {"characters": {}, "props": {}}
    characters = sorted(asset_refs.get("characters", []), key=lambda name: _character_ref_priority(assets, name))
    for char in characters:
        ref = _preferred_ref(assets.get("characters", {}).get(char, {}))
        if ref:
            result["characters"][char] = ref
    scene = asset_refs.get("scene")
    if scene:
        ref = _preferred_ref(assets.get("scenes", {}).get(scene, {}))
        if ref:
            result["scene"] = ref
    for prop in asset_refs.get("props", []):
        ref = _preferred_ref(assets.get("props", {}).get(prop, {}))
        if ref:
            result["props"][prop] = ref
    return result


def collect_video_asset_urls(data: dict, board: dict) -> dict:
    assets = data.get("assets", {})
    asset_refs = board.get("asset_refs", {})
    result = {"characters": {}, "props": {}}
    characters = sorted(asset_refs.get("characters", []), key=lambda name: _character_ref_priority(assets, name))
    for char in characters:
        ref = _preferred_ref(assets.get("characters", {}).get(char, {}))
        if ref:
            result["characters"][char] = ref
    return result
