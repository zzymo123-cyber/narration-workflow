import json
import hashlib
import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import pipeline as pl
from api.decomposition import (
    build_step1_prompt, build_step2_prompt, build_step3_prompt,
    parse_decomposition_response,
)
from api.prompts import assemble_storyboard_prompt, assemble_video_prompt
from api.script_planner import PLANNER_VERSION, plan_script_locally
from api.shot_techniques import technique_options_for_prompt
from api.validation import validate_board_page
from api.llm import generate_prompt_async, get_llm_config, _resolve_api_key
from api.generation import (
    _is_snake_character_asset,
    apply_default_visual_macros,
    ensure_audio_refs,
    ensure_bgm_refs,
    ensure_referenced_assets,
    normalize_assets,
    normalize_video_output_paths,
    selected_bgm_ref,
    selected_audio_ref,
    sync_board_metadata,
)
from api.routes.settings import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateProjectRequest(BaseModel):
    project_name: str
    source_text: str
    narration_style: str = "third_person"


class DecomposeRequest(BaseModel):
    project_name: str
    narration_style: str | None = None


class SelectAudioRefRequest(BaseModel):
    project_name: str
    selected: str = ""


class SelectBgmRefRequest(BaseModel):
    project_name: str
    selected: str = ""


class ConcatEpisodeRequest(BaseModel):
    project_name: str
    episode: int
    copy_to_desktop: bool = True


def _project_stats(data: dict) -> dict:
    sync_board_metadata(data)
    assets = data.get("assets", {})
    segments = data.get("narration_segments", {})
    boards = [board for seg in segments.values() for board in seg.get("boards", [])]
    script_plan = data.get("script_plan", {})
    return {
        "source_len": len(data.get("source_text", "")),
        "script_boards": len(script_plan.get("board_plan", [])),
        "script_estimated_seconds": script_plan.get("stats", {}).get("estimated_seconds", 0),
        "characters": len(assets.get("characters", {})),
        "scenes": len(assets.get("scenes", {})),
        "props": len(assets.get("props", {})),
        "segments": len(segments),
        "boards": len(boards),
        "storyboards_completed": sum(1 for b in boards if b.get("storyboard_image", {}).get("status") == "completed"),
        "videos_completed": sum(1 for b in boards if b.get("video", {}).get("status") == "completed"),
    }


def _pipeline_integrity_issues(data: dict) -> list[str]:
    issues = []
    segments = data.get("narration_segments", {})
    if not segments:
        return issues
    for seg_key, seg in segments.items():
        boards = seg.get("boards", [])
        if not boards:
            issues.append(f"{seg_key} 缺失故事板")
            continue
        for index, board in enumerate(boards):
            label = board.get("display_id") or f"{seg_key} P{index + 1:02d}"
            refs = board.get("asset_refs", {}) or {}
            if not refs.get("characters"):
                issues.append(f"{label} 缺失角色引用")
            if not refs.get("scene"):
                issues.append(f"{label} 缺失场景引用")
            if not board.get("voice_timeline"):
                issues.append(f"{label} 缺失旁白/对白时间轴")
            if not board.get("shot_timeline"):
                issues.append(f"{label} 缺失镜头分解")
            if not isinstance(board.get("storyboard_image"), dict):
                issues.append(f"{label} 缺失故事板图片任务")
            if not isinstance(board.get("video"), dict):
                issues.append(f"{label} 缺失视频任务")
            for error in validate_board_page(board)[:3]:
                issues.append(f"{label} 校验失败：{error}")
    return issues[:50]


@router.get("/list")
async def list_projects():
    root = Path.home() / "Desktop" / "narration_studio"
    projects = []
    if root.exists():
        for project_dir in sorted(root.iterdir(), key=lambda p: p.name):
            pipeline_path = project_dir / "pipeline.json"
            if not project_dir.is_dir() or not pipeline_path.exists():
                continue
            try:
                data = pl.read_pipeline(project_dir)
            except Exception:
                continue
            projects.append({
                "name": data.get("project") or project_dir.name,
                "path": str(project_dir),
                **_project_stats(data),
            })
    return {"projects": projects}


@router.get("/preflight")
async def project_preflight(project_name: str):
    project_dir = _check_project(project_name)
    data = pl.read_pipeline(project_dir)
    ensure_referenced_assets(data)
    ensure_audio_refs(data, project_dir)
    sync_board_metadata(data)
    stats = _project_stats(data)
    llm = get_llm_config()
    integrity_issues = _pipeline_integrity_issues(data)
    checks = {
        "source": {"ok": bool(data.get("source_text", "").strip()), "label": "故事文本"},
        "llm": {"ok": bool(llm.get("has_api_key")), "label": "LLM Key"},
        "vidu": {"ok": bool(get_api_key("VIDU_API_KEY")), "label": "Vidu Key"},
        "wetoken": {"ok": bool(get_api_key("WETOKEN_API_KEY")), "label": "Wetoken Key"},
        "decomposed": {"ok": stats["segments"] > 0 and stats["boards"] > 0, "label": "故事拆解"},
        "audio_ref": {"ok": bool(selected_audio_ref(data)), "label": "旁白音色"},
        "pipeline_integrity": {"ok": not integrity_issues, "label": "流程完整性"},
    }
    blockers = []
    if not checks["source"]["ok"]:
        blockers.append("缺少故事文本")
    if not checks["llm"]["ok"]:
        provider_name = llm.get("provider", "LLM")
        provider = "DeepSeek" if provider_name == "deepseek" else provider_name
        blockers.append(f"缺少 {provider} 完整 API Key，无法拆解故事")
    if not checks["vidu"]["ok"]:
        blockers.append("缺少 Vidu 完整 API Key，无法生成资产图和分镜图")
    if not checks["wetoken"]["ok"]:
        blockers.append("缺少 Wetoken 完整 API Key，无法生成视频")
    if not checks["audio_ref"]["ok"]:
        blockers.append("缺少旁白音色，无法保证音色一致性")
    if not checks["pipeline_integrity"]["ok"]:
        blockers.append("项目存在角色、场景、故事板或声音结构缺失，请先修正")
    pl.write_pipeline(project_dir, data)
    return {
        "project": data.get("project") or project_name,
        "project_dir": str(project_dir),
        "stats": stats,
        "checks": checks,
        "blockers": blockers,
        "integrity_issues": integrity_issues,
        "can_decompose": checks["source"]["ok"] and checks["llm"]["ok"],
        "can_generate_images": checks["vidu"]["ok"] and checks["decomposed"]["ok"],
        "can_generate_videos": (
            checks["wetoken"]["ok"]
            and checks["audio_ref"]["ok"]
            and checks["pipeline_integrity"]["ok"]
            and stats["storyboards_completed"] > 0
        ),
    }


@router.post("/create")
async def create_project(req: CreateProjectRequest):
    """Create a new narration project with source text."""
    project_dir = pl.get_project_root(req.project_name)
    if project_dir.exists() and (project_dir / "pipeline.json").exists():
        raise HTTPException(status_code=409, detail=f"项目已存在: {req.project_name}")

    project_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "project": req.project_name,
        "narration_style": req.narration_style,
        "source_text": req.source_text,
        "assets": {"characters": {}, "scenes": {}, "props": {}},
        "audio_refs": {"selected": "", "options": {}},
        "bgm_refs": {"selected": "", "options": {}, "volume": 0.18},
        "narration_segments": {},
    }
    pl.write_pipeline(project_dir, data)
    return {"ok": True, "project_dir": str(project_dir)}


# ── Shared helpers ──

def _check_project(req_name: str):
    project_dir = pl.get_project_root(req_name)
    if not (project_dir / "pipeline.json").exists():
        raise HTTPException(status_code=404, detail=f"项目不存在: {req_name}")
    return project_dir


def _require_api_key():
    api_key = _resolve_api_key()
    if not api_key:
        config = get_llm_config()
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "no_api_key",
                "message": f"未配置 {config['provider']} 的 API Key",
                "config": config,
            },
        )
    return api_key


def _source_checksum(source_text: str) -> str:
    return hashlib.sha256((source_text or "").encode("utf-8")).hexdigest()


def _ensure_script_plan(data: dict, force: bool = False) -> tuple[dict, bool]:
    source_text = data.get("source_text", "")
    checksum = _source_checksum(source_text)
    existing = data.get("script_plan") or {}
    if not force and existing.get("source_checksum") == checksum and existing.get("planner_version") == PLANNER_VERSION:
        return existing, False
    plan = plan_script_locally(source_text)
    plan["source_checksum"] = checksum
    return plan, True


@router.post("/plan-script")
async def plan_project_script(req: DecomposeRequest):
    """Create the deterministic voice/script plan for a project."""
    project_dir = _check_project(req.project_name)
    data = pl.read_pipeline(project_dir)
    if not data.get("source_text", "").strip():
        raise HTTPException(status_code=400, detail="项目没有 source_text，无法规划声音")
    script_plan, _ = _ensure_script_plan(data, force=True)
    data["script_plan"] = script_plan
    pl.write_pipeline(project_dir, data)
    return {
        "ok": True,
        "coverage_errors": script_plan.get("coverage_errors", []),
        "stats": script_plan.get("stats", {}),
    }


@router.post("/reset-storyboards")
async def reset_project_storyboards(req: DecomposeRequest):
    """Clear generated storyboard structure and archive old storyboard images."""
    project_dir = _check_project(req.project_name)
    data = pl.read_pipeline(project_dir)
    data["narration_segments"] = {}
    data.pop("_decomposition_outline", None)

    archived = []
    archive_root = project_dir / "_archive"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for dirname in ("storyboards",):
        path = project_dir / dirname
        if path.exists():
            archive_root.mkdir(parents=True, exist_ok=True)
            dest = archive_root / f"{dirname}_{stamp}"
            path.rename(dest)
            archived.append(str(dest))

    pl.write_pipeline(project_dir, data)
    return {"ok": True, "archived": archived}


async def _call_llm(api_key: str, system_prompt: str, user_message: str, attempts: int = 1) -> dict:
    """Call LLM and return parsed JSON dict. Raises HTTPException on any failure."""
    # Call LLM
    raw_response = ""
    for attempt in range(max(1, attempts)):
        try:
            raw_response = await generate_prompt_async(api_key, system_prompt, user_message)
            if raw_response.strip() or attempt == attempts - 1:
                break
            logger.warning("LLM 返回空内容，重试 %s/%s", attempt + 1, attempts)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            config = get_llm_config()
            err_str = str(e)
            if "401" in err_str or "Authentication" in err_str or "Invalid" in err_str:
                error_type = "auth_failed"
            elif "403" in err_str or "Permission" in err_str:
                error_type = "permission_denied"
            elif "404" in err_str or "Not Found" in err_str:
                error_type = "not_found"
            elif "Connection" in err_str or "connect" in err_str.lower():
                error_type = "connection_error"
            else:
                error_type = "unknown"
            raise HTTPException(
                status_code=500,
                detail={"error_type": error_type, "message": err_str[:200], "config": config, "raw_error": err_str[:500]},
            )

    # Save debug file
    debug_dir = Path(__file__).parent.parent.parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    (debug_dir / "last_llm_response.txt").write_text(raw_response, encoding="utf-8")

    # Extract JSON
    text = raw_response.strip()
    if not text:
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "empty_llm_response",
                "message": "LLM 返回空内容，请重试当前步骤",
                "total_chars": 0,
                "debug_file": "debug/last_llm_response.txt",
            },
        )
    if "```" in text:
        lines = text.split("\n")
        cleaned = []
        in_fence = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence and stripped == "":
                continue
            cleaned.append(line)
        text = "\n".join(cleaned).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        extracted = text[first_brace:last_brace + 1]
    else:
        extracted = text

    # Parse JSON
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as e:
        logger.error(f"LLM JSON parse error: line {e.lineno} col {e.colno}")
        error_lines = extracted.split("\n")
        context_start = max(0, e.lineno - 6)
        context_end = min(len(error_lines), e.lineno + 5)
        context = "\n".join(
            f"{'>>>' if i == e.lineno - 1 else '   '} {context_start + i + 1:4d} | {line}"
            for i, line in enumerate(error_lines[context_start:context_end])
        )
        issues = []
        if not extracted:
            issues.insert(0, "LLM 输出为空")
        elif extracted[-1] != "}":
            issues.insert(0, "LLM 输出被截断，JSON 不完整")
        elif last_brace != -1 and last_brace < len(text) - 2:
            issues.insert(0, "LLM 输出被截断，JSON 不完整")
        if ",}" in extracted or ",]" in extracted:
            issues.append("包含多余逗号")
        if not issues:
            issues.append("可能是未转义字符或格式问题")

        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "json_parse_error",
                "message": "LLM 返回不是合法 JSON",
                "lineno": e.lineno, "colno": e.colno, "pos": e.pos, "msg": e.msg,
                "context": context, "detected_issues": issues,
                "total_chars": len(extracted), "debug_file": "debug/last_llm_response.txt",
            },
        )


# ── Multi-step decompose ──

def _segment_sort_key(seg_key: str) -> tuple:
    numbers = [int(part) for part in re.findall(r"\d+", seg_key)]
    return (*numbers, seg_key)


def _episode_number(seg_key: str, seg: dict) -> int:
    try:
        value = int(seg.get("episode") or 0)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    numbers = [int(part) for part in re.findall(r"\d+", seg_key)]
    return numbers[0] if numbers else 1


def _episode_groups(data: dict) -> list[dict]:
    exports = data.get("episode_exports", {})
    grouped: dict[int, dict] = {}
    segments = data.get("narration_segments", {})
    has_backend_episodes = any(
        _episode_number(key, seg or {}) > 1
        for key, seg in segments.items()
    )
    fallback_episode = 0
    fallback_seconds = 0
    max_seconds = _episode_max_seconds()
    for seg_key in sorted(segments.keys(), key=_segment_sort_key):
        seg = segments.get(seg_key) or {}
        seg_duration = sum(
            int(board.get("board_duration") or board.get("video", {}).get("duration") or 0)
            for board in seg.get("boards", [])
        )
        if has_backend_episodes:
            episode_no = _episode_number(seg_key, seg)
        else:
            if fallback_episode == 0 or (fallback_seconds > 0 and fallback_seconds + seg_duration > max_seconds):
                fallback_episode += 1
                fallback_seconds = 0
            episode_no = fallback_episode
            fallback_seconds += seg_duration
        group = grouped.setdefault(episode_no, {
            "episode": episode_no,
            "title": f"第{episode_no}集",
            "seconds": 0,
            "segments": [],
            "parts": [],
            "videos_total": 0,
            "videos_completed": 0,
            "missing_parts": [],
            "export": exports.get(str(episode_no), {}),
        })
        if seg_key not in group["segments"]:
            group["segments"].append(seg_key)
        for idx, board in enumerate(seg.get("boards", [])):
            video = board.get("video", {})
            duration = int(board.get("board_duration") or video.get("duration") or 0)
            status = video.get("status") or "needed"
            local_path = video.get("local_path")
            part = {
                "segment_key": seg_key,
                "board_index": idx,
                "label": f"{seg_key} P{idx + 1}",
                "duration": duration,
                "video_status": status,
                "local_path": local_path,
                "url": video.get("url"),
                "storyboard_status": board.get("storyboard_image", {}).get("status", "needed"),
                "video_goal": board.get("video_goal", ""),
            }
            group["parts"].append(part)
            group["seconds"] += duration
            group["videos_total"] += 1
            if status == "completed" and local_path:
                group["videos_completed"] += 1
            else:
                group["missing_parts"].append(part["label"])
    for group in grouped.values():
        group["can_concat"] = bool(group["parts"]) and not group["missing_parts"]
    return [grouped[key] for key in sorted(grouped)]


def _concat_video_files(
    inputs: list[Path],
    output: Path,
    subtitle_events: list[dict] | None = None,
    bgm_path: Path | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        durations = [_probe_media_duration(path) for path in inputs]
        audio_flags = [_probe_has_audio(path) for path in inputs]
        with tempfile.TemporaryDirectory(prefix="episode_subtitles_") as tmp:
            offsets = _part_start_offsets(durations)
            total_duration = (offsets[-1] + (durations[-1] or 0.0)) if offsets else 0.0
            subtitle_sequence = _render_subtitle_sequence(
                subtitle_events or [],
                offsets,
                Path(tmp),
                total_duration,
            )
            subprocess.run(
                _build_ffmpeg_concat_command(ffmpeg, inputs, output, durations, audio_flags, subtitle_sequence, bgm_path),
                check=True,
                capture_output=True,
                text=True,
            )
        return

    if bgm_path:
        raise RuntimeError("合成 BGM 需要安装 ffmpeg")

    swift = shutil.which("swift")
    if not swift:
        raise RuntimeError("未找到 ffmpeg 或 swift，无法在本机合成视频")
    inputs_json = json.dumps([str(path) for path in inputs], ensure_ascii=False)
    output_json = json.dumps(str(output), ensure_ascii=False)
    script = f"""
import Foundation
import AVFoundation

let inputPaths = {inputs_json}
let outputPath = {output_json}
let outputURL = URL(fileURLWithPath: outputPath)
try? FileManager.default.removeItem(at: outputURL)
let composition = AVMutableComposition()
guard let videoTrack = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {{
  fatalError("无法创建视频轨")
}}
let audioTrack = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
var cursor = CMTime.zero
var hasTransform = false
var insertedAudio = false
for path in inputPaths {{
  let asset = AVURLAsset(url: URL(fileURLWithPath: path))
  let range = CMTimeRange(start: .zero, duration: asset.duration)
  guard let sourceVideo = asset.tracks(withMediaType: .video).first else {{
    fatalError("缺少视频轨: \\(path)")
  }}
  if !hasTransform {{
    videoTrack.preferredTransform = sourceVideo.preferredTransform
    hasTransform = true
  }}
  try videoTrack.insertTimeRange(range, of: sourceVideo, at: cursor)
  if let sourceAudio = asset.tracks(withMediaType: .audio).first {{
    try audioTrack?.insertTimeRange(range, of: sourceAudio, at: cursor)
    insertedAudio = true
  }}
  cursor = cursor + asset.duration
}}
if !insertedAudio, let audioTrack = audioTrack {{
  composition.removeTrack(audioTrack)
}}
guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {{
  fatalError("无法创建导出器")
}}
exporter.outputURL = outputURL
exporter.outputFileType = .mp4
exporter.shouldOptimizeForNetworkUse = true
let sem = DispatchSemaphore(value: 0)
exporter.exportAsynchronously {{ sem.signal() }}
sem.wait()
if exporter.status != .completed {{
  fatalError(String(describing: exporter.error))
}}
"""
    subprocess.run([swift, "-"], input=script, check=True, capture_output=True, text=True)


def _probe_media_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError:
        return 0.0


def _probe_has_audio(path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return "audio" in result.stdout


def _part_start_offsets(durations: list[float]) -> list[float]:
    offsets = []
    cursor = 0.0
    for duration in durations:
        offsets.append(cursor)
        cursor += duration or 0.0
    return offsets


def _episode_subtitle_events(data: dict, parts: list[dict]) -> list[dict]:
    events = []
    for part_index, part in enumerate(parts):
        segment = data.get("narration_segments", {}).get(part["segment_key"], {})
        boards = segment.get("boards", [])
        board_index = part["board_index"]
        if board_index >= len(boards):
            continue
        board = boards[board_index]
        for beat in board.get("voice_timeline", []):
            raw_text = str(beat.get("text", "") or "")
            if not _subtitle_text(raw_text):
                continue
            try:
                start = float(beat.get("start", 0) or 0)
                end = float(beat.get("end", start + 1) or start + 1)
            except (TypeError, ValueError):
                continue
            chunks = _subtitle_chunks(raw_text)
            if not chunks:
                continue
            duration = max(0.6, end - start)
            chunk_duration = duration / len(chunks)
            for index, chunk in enumerate(chunks):
                events.append({
                    "part_index": part_index,
                    "start": start + chunk_duration * index,
                    "end": start + chunk_duration * (index + 1),
                    "text": chunk,
                })
    return events


def _subtitle_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^[《【\[]?第[一二三四五六七八九十百千万0-9]+[章节集幕段][》】\]]?[：:：、\s-]*", "", text)
    text = re.sub(r"^(旁白|标题|字幕|画外音|内心独白|沈砚|秦越|朋友|医生)[：:：]\s*", "", text)
    text = re.sub(r"[“”\"'‘’《》【】\[\]（）(){}<>]", "", text)
    text = re.sub(r"[，。！？；：、,.!?;:…·—~\-]", "", text)
    return re.sub(r"\s+", "", text)


def _subtitle_chunks(text: str, max_chars: int = 12, min_chars: int = 5) -> list[str]:
    raw_text = str(text or "").strip()
    raw_text = re.sub(r"^[《【\[]?第[一二三四五六七八九十百千万0-9]+[章节集幕段][》】\]]?[：:：、\s-]*", "", raw_text)
    raw_text = re.sub(r"^(旁白|标题|字幕|画外音|内心独白|沈砚|秦越|朋友|医生)[：:：]\s*", "", raw_text)
    pieces = [_subtitle_text(piece) for piece in re.split(r"[，。！？；：,.!?;:]+", raw_text)]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        cleaned = _subtitle_text(text)
        pieces = [cleaned] if cleaned else []
    if not pieces:
        return []

    def split_balanced(value: str) -> list[str]:
        if len(value) <= max_chars + 2:
            return [value]
        count = (len(value) + max_chars - 1) // max_chars
        while count > 1 and len(value) / count < min_chars:
            count -= 1
        chunk_size = (len(value) + count - 1) // count
        result = [value[index:index + chunk_size] for index in range(0, len(value), chunk_size)]
        if len(result) > 1 and len(result[-1]) < min_chars:
            tail = result.pop()
            combined = result.pop() + tail
            split_at = max(min_chars, len(combined) // 2)
            result.extend([combined[:split_at], combined[split_at:]])
        return result

    chunks = []
    current = ""
    for piece in pieces:
        if len(piece) > max_chars + 2:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_balanced(piece))
            continue
        if not current:
            current = piece
            continue
        if len(piece) < min_chars or len(current) < min_chars or len(current) + len(piece) <= max_chars:
            current += piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    balanced = []
    for chunk in chunks:
        balanced.extend(split_balanced(chunk))
    chunks = balanced
    return [chunk for chunk in chunks if chunk]


def _render_subtitle_sequence(
    events: list[dict],
    part_offsets: list[float],
    tmp_dir: Path,
    total_duration: float,
    fps: int = 6,
) -> dict | None:
    if not events:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        logger.warning("字幕图片渲染不可用: %s", exc)
        return None

    font_path = None
    try:
        font_path = _subtitle_font_path()
        font = ImageFont.truetype(str(font_path), 54)
    except Exception as exc:
        logger.warning("字幕字体加载失败: %s", exc)
        font = ImageFont.load_default()

    absolute_events = []
    for event in events:
        part_index = int(event.get("part_index", 0) or 0)
        if part_index >= len(part_offsets):
            continue
        start = part_offsets[part_index] + float(event.get("start", 0) or 0)
        end = part_offsets[part_index] + float(event.get("end", start + 1) or start + 1)
        if end <= start:
            continue
        text = str(event.get("text", "")).strip()
        if not text:
            continue
        absolute_events.append({"start": start, "end": end, "text": text})
    if not absolute_events or total_duration <= 0:
        return None

    tmp_dir.mkdir(parents=True, exist_ok=True)
    pattern = tmp_dir / "subtitle_%05d.png"
    image_cache = {}

    def split_subtitle_lines(text: str, line_count: int) -> list[str]:
        if line_count <= 1:
            return [text]
        size = (len(text) + line_count - 1) // line_count
        return [text[index:index + size] for index in range(0, len(text), size) if text[index:index + size]][:line_count]

    def choose_subtitle_layout(draw, text: str):
        if not font_path:
            return [text], font, 5
        for line_count in range(1, 4):
            lines = split_subtitle_lines(text, line_count)
            for size in range(54, 23, -2):
                candidate = ImageFont.truetype(str(font_path), size)
                candidate_stroke = max(2, round(size * 0.09))
                boxes = [
                    draw.textbbox((0, 0), line, font=candidate, stroke_width=candidate_stroke)
                    for line in lines
                ]
                widths = [box[2] - box[0] for box in boxes]
                heights = [box[3] - box[1] for box in boxes]
                total_height = sum(heights) + max(0, len(lines) - 1) * 6
                if max(widths or [0]) <= 684 and total_height <= 150:
                    return lines, candidate, candidate_stroke
        return split_subtitle_lines(text, 3), ImageFont.truetype(str(font_path), 24), 2

    def render_strip(text: str):
        if text in image_cache:
            return image_cache[text]
        image = Image.new("RGBA", (720, 220), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        if text:
            lines, active_font, stroke_width = choose_subtitle_layout(draw, text)
            boxes = [
                draw.textbbox((0, 0), line, font=active_font, stroke_width=stroke_width)
                for line in lines
            ]
            heights = [box[3] - box[1] for box in boxes]
            total_height = sum(heights) + max(0, len(lines) - 1) * 6
            cursor_y = max(12, (130 - total_height) // 2)
            for line, bbox, height in zip(lines, boxes, heights):
                width = bbox[2] - bbox[0]
                x = max(18, (720 - width) // 2) - bbox[0]
                y = cursor_y - bbox[1]
                draw.text(
                    (x, y),
                    line,
                    font=active_font,
                    fill=(248, 248, 248, 255),
                    stroke_width=stroke_width,
                    stroke_fill=(16, 16, 16, 245),
                )
                cursor_y += height + 6
        image_cache[text] = image
        return image

    frame_count = max(1, int(total_duration * fps) + 2)
    event_index = 0
    for frame_index in range(frame_count):
        timestamp = frame_index / fps
        while event_index < len(absolute_events) and timestamp >= absolute_events[event_index]["end"]:
            event_index += 1
        text = ""
        if event_index < len(absolute_events):
            event = absolute_events[event_index]
            if event["start"] <= timestamp < event["end"]:
                text = event["text"]
        render_strip(text).save(tmp_dir / f"subtitle_{frame_index:05d}.png")
    return {"pattern": pattern, "fps": fps, "y": 880}


def _subtitle_font_path() -> Path:
    candidates = [
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError("未找到可用的中文字幕字体")


def _build_ffmpeg_concat_command(
    ffmpeg: str,
    inputs: list[Path],
    output: Path,
    durations: list[float],
    audio_flags: list[bool] | None = None,
    subtitle_sequence: dict | None = None,
    bgm_path: Path | None = None,
    bgm_volume: float = 0.18,
) -> list[str]:
    if not inputs:
        raise ValueError("没有可合成的视频片段")
    if len(inputs) == 1 and not bgm_path:
        command = [ffmpeg, "-y", "-i", str(inputs[0])]
        if subtitle_sequence:
            command.extend([
                "-framerate",
                str(int(subtitle_sequence.get("fps", 6))),
                "-i",
                str(subtitle_sequence["pattern"]),
                "-filter_complex",
                "[0:v]fps=30,scale=720:1280:force_original_aspect_ratio=decrease,"
                "pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v0];"
                "[1:v]fps=30,format=rgba[subtitles];"
                f"[v0][subtitles]overlay=0:{int(subtitle_sequence.get('y', 880))}:shortest=1[vout]",
                "-map",
                "[vout]",
                "-map",
                "0:a?",
            ])
        else:
            command.extend([
                "-vf",
                "fps=30,scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
            ])
        command.extend([
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output),
        ])
        return command

    audio_flags = audio_flags or [True] * len(inputs)
    command = [ffmpeg, "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    if subtitle_sequence:
        command.extend([
            "-framerate",
            str(int(subtitle_sequence.get("fps", 6))),
            "-i",
            str(subtitle_sequence["pattern"]),
        ])
    bgm_input = None
    if bgm_path:
        bgm_input = len(inputs) + (1 if subtitle_sequence else 0)
        command.extend(["-stream_loop", "-1", "-i", str(bgm_path)])

    filter_parts = []
    for index, duration in enumerate(durations):
        filter_parts.append(
            f"[{index}:v]fps=30,scale=720:1280:force_original_aspect_ratio=decrease,"
            f"pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p,"
            f"settb=AVTB,setpts=PTS-STARTPTS[v{index}]"
        )
        if audio_flags[index]:
            filter_parts.append(
                f"[{index}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        else:
            fallback_duration = max(0.1, duration or 0.1)
            filter_parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate=44100,"
                f"atrim=duration={fallback_duration:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )

    concat_inputs = "".join(f"[v{index}][a{index}]" for index in range(len(inputs)))
    filter_parts.append(f"{concat_inputs}concat=n={len(inputs)}:v=1:a=1[vcat][acat]")
    current_video = "vcat"
    current_audio = "acat"

    if subtitle_sequence:
        subtitle_input = len(inputs)
        subtitle_y = int(subtitle_sequence.get("y", 880))
        filter_parts.append(
            f"[{subtitle_input}:v]fps=30,format=rgba[subtitles]"
        )
        filter_parts.append(
            f"[{current_video}][subtitles]overlay=0:{subtitle_y}:shortest=1[vsub]"
        )
        current_video = "vsub"

    if bgm_input is not None:
        safe_volume = max(0.0, min(1.0, float(bgm_volume)))
        filter_parts.append(
            f"[{bgm_input}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"volume={safe_volume:.2f},atrim=duration={sum(durations):.3f},asetpts=PTS-STARTPTS[bgm]"
        )
        filter_parts.append(f"[{current_audio}][bgm]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        current_audio = "aout"

    command.extend([
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        f"[{current_video}]",
        "-map",
        f"[{current_audio}]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ])
    return command


def _prepare_board_prompts(board: dict) -> None:
    board.setdefault("storyboard_image", {"status": "needed", "prompt": "", "task_id": None, "url": None, "local_path": None})
    board.setdefault("video", {"status": "needed", "prompt": "", "task_id": None, "url": None, "local_path": None})
    board["storyboard_image"]["prompt"] = assemble_storyboard_prompt(board)
    board["video"]["prompt"] = assemble_video_prompt(board)


def _write_decompose_progress(
    project_dir: Path,
    data: dict,
    *,
    status: str,
    stage: str,
    message: str,
    current: int = 0,
    total: int = 0,
    extra: dict | None = None,
) -> None:
    progress = {
        "status": status,
        "stage": stage,
        "message": message,
        "current": max(0, int(current or 0)),
        "total": max(0, int(total or 0)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        progress.update(extra)
    data["_decompose_progress"] = progress
    pl.write_pipeline(project_dir, data)


def _write_decompose_failure(project_dir: Path, data: dict, stage: str, exc: HTTPException) -> None:
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    previous = data.get("_decompose_progress", {})
    _write_decompose_progress(
        project_dir,
        data,
        status="failed",
        stage=stage or previous.get("stage") or "unknown",
        message=detail.get("message") or detail.get("detail") or "拆解失败",
        current=previous.get("current", 0),
        total=previous.get("total", 0),
        extra={
            "error_type": detail.get("error_type", "unknown"),
            "error_detail": detail,
        },
    )


def _planned_board_total(segments_outline: dict) -> int:
    total = 0
    for seg in segments_outline.values():
        try:
            total += max(0, int(seg.get("num_boards", 2) or 0))
        except (TypeError, ValueError):
            total += 2
    return total


def _completed_board_total(narration_segments: dict) -> int:
    return sum(len(seg.get("boards", [])) for seg in narration_segments.values())


def _decompose_concurrency() -> int:
    try:
        return max(1, min(64, int(os.environ.get("DECOMPOSE_BOARD_CONCURRENCY", "24"))))
    except ValueError:
        return 24


def _episode_min_seconds() -> int:
    try:
        return max(60, min(600, int(os.environ.get("EPISODE_MIN_SECONDS", "150"))))
    except ValueError:
        return 150


def _episode_target_seconds() -> int:
    try:
        return max(60, min(600, int(os.environ.get("EPISODE_TARGET_SECONDS", "180"))))
    except ValueError:
        return 180


def _episode_max_seconds() -> int:
    try:
        return max(60, min(600, int(os.environ.get("EPISODE_MAX_SECONDS", "210"))))
    except ValueError:
        return 210


def _segment_boards() -> int:
    try:
        return max(1, min(12, int(os.environ.get("SEGMENT_BOARD_COUNT", "5"))))
    except ValueError:
        return 5


def _build_segments_outline(
    board_plan: list[dict],
    script_plan: dict | None = None,
    episode_min_seconds: int | None = None,
    episode_target_seconds: int | None = None,
    episode_max_seconds: int | None = None,
    segment_boards: int | None = None,
) -> dict:
    """Group planned boards into reviewable segments inside semantic short episodes."""
    episode_min_seconds = episode_min_seconds or _episode_min_seconds()
    episode_target_seconds = episode_target_seconds or _episode_target_seconds()
    episode_max_seconds = episode_max_seconds or _episode_max_seconds()
    segment_boards = segment_boards or _segment_boards()
    beats_by_id = {
        beat.get("beat_id"): beat
        for beat in (script_plan or {}).get("voice_beats", [])
    }

    def board_duration(item: dict) -> int:
        return int(item.get("duration") or item.get("estimated_seconds") or 0)

    def board_text(item: dict) -> str:
        parts = []
        for beat_id in item.get("voice_beat_ids", []):
            text = beats_by_id.get(beat_id, {}).get("text")
            if text:
                parts.append(text)
        return "".join(parts)

    def safe_episode_boundary(end_index: int) -> bool:
        text = board_text(board_plan[end_index - 1]).rstrip()
        if end_index >= len(board_plan):
            return True
        next_text = board_text(board_plan[end_index]).lstrip()
        if next_text.startswith(("，", ",", "、", "；", ";")):
            return False
        return text.endswith(("。", "！", "？", "!", "?", "…", "”"))

    def boundary_reason(end_index: int, seconds: int) -> tuple[int, str]:
        item = board_plan[end_index - 1]
        text = board_text(item)
        score = 0
        reasons = []
        if text.rstrip().endswith(("。", "！", "？", "!", "?", "”")):
            score += 3
            reasons.append("句子收束")
        if any(word in text for word in ("警告", "突然", "发现", "意识到", "真相", "不是", "竟然", "没想到", "来不及")):
            score += 5
            reasons.append("语义边界")
        if any(mark in text for mark in ("！", "？", "!?")):
            score += 2
            reasons.append("情绪钩子")
        distance = abs(seconds - episode_target_seconds)
        score += max(0, 4 - distance // 15)
        if not reasons:
            reasons.append("软时长边界")
        return score, "、".join(dict.fromkeys(reasons))

    episodes = []
    start = 1
    seconds = 0
    candidates = []
    for index, item in enumerate(board_plan, start=1):
        seconds += board_duration(item)
        if seconds >= episode_min_seconds and safe_episode_boundary(index):
            score, reason = boundary_reason(index, seconds)
            candidates.append({
                "end": index,
                "seconds": seconds,
                "score": score,
                "reason": reason,
            })
        next_duration = board_duration(board_plan[index]) if index < len(board_plan) else 0
        must_close = index == len(board_plan) or (candidates and seconds + next_duration > episode_max_seconds)
        should_close = (
            candidates
            and (
                (seconds >= episode_target_seconds and candidates[-1]["score"] >= 5)
                or candidates[-1]["score"] >= 8
            )
        )
        if must_close or should_close:
            chosen = max(candidates, key=lambda item: (item["score"], -abs(item["seconds"] - episode_target_seconds))) if candidates else {
                "end": index,
                "seconds": seconds,
                "reason": "不足最小时长，保留剧情连续性",
            }
            episodes.append({
                "episode": len(episodes) + 1,
                "start": start,
                "end": chosen["end"],
                "seconds": chosen["seconds"],
                "reason": chosen["reason"],
            })
            start = chosen["end"] + 1
            seconds = sum(board_duration(item) for item in board_plan[start - 1:index])
            candidates = [
                candidate
                for candidate in candidates
                if candidate["end"] >= start and candidate["end"] <= index
            ]

    outline = {}
    for episode in episodes:
        segment_index = 1
        segment_start = episode["start"]
        while segment_start <= episode["end"]:
            segment_end = min(segment_start + segment_boards - 1, episode["end"])
            segment_items = board_plan[segment_start - 1:segment_end]
            segment_seconds = sum(board_duration(item) for item in segment_items)
            key = f"seg_{episode['episode']}_{segment_index}"
            outline[key] = {
                "episode": episode["episode"],
                "segment_index": segment_index,
                "episode_seconds": episode["seconds"],
                "episode_board_range": [episode["start"], episode["end"]],
                "episode_boundary_reason": episode["reason"],
                "scene_location": "",
                "characters_in_segment": [],
                "num_boards": len(segment_items),
                "segment_seconds": segment_seconds,
                "board_range": [segment_start, segment_end],
            }
            segment_index += 1
            segment_start = segment_end + 1
    return outline


def _protagonist_name(assets: dict) -> str:
    characters = assets.get("characters", {})
    if "沈砚" in characters:
        return "沈砚"
    return next(iter(characters.keys()), "我")


def _script_maps(script_plan: dict) -> tuple[dict, dict]:
    beats = {beat.get("beat_id"): beat for beat in script_plan.get("voice_beats", [])}
    slices = {item.get("slice_id"): item for item in script_plan.get("script_slices", [])}
    return beats, slices


def _is_snake_name(name: str) -> bool:
    return any(word in name for word in ("玄墨", "蟒", "蛇"))


def _character_aliases(name: str) -> list[str]:
    aliases = [name]
    if name == "郑教授":
        aliases.extend(["郑老先生", "郑老", "郑叔"])
    elif name == "沈砚":
        aliases.extend(["小沈"])
    elif _is_snake_name(name):
        aliases.extend(["玄墨", "蟒蛇", "蛇", "长虫", "宝贝"])
    return aliases


def _has_snake_evidence(text: str) -> bool:
    if any(word in text for word in ("玄墨", "蟒", "蛇", "长虫")):
        return True
    snake_cues = (
        "信子", "鳞片", "蜕皮", "饲养箱", "恒温箱", "加热石", "白鼠", "猎物",
        "吞咽", "投食", "镊子", "盘踞", "盘卷", "爬回", "爬出", "滑下",
        "前半身", "头部", "尾巴", "身体绷", "冰冷", "鳞",
    )
    return "它" in text and any(cue in text for cue in snake_cues)


def _resolve_dialogue_speaker(raw_speaker: str, assets: dict, narrator_name: str, previous_text: str,
                              current_text: str = "") -> str:
    if raw_speaker and raw_speaker not in {"角色", "恐惧语气", "前所未有", "电话那头"} and "语气" not in raw_speaker:
        return raw_speaker
    context = f"{previous_text}{current_text}"
    if ("郑叔" in current_text or current_text.startswith("您") or "您进来" in current_text) and narrator_name:
        return narrator_name
    if "小沈" in current_text and "郑教授" in (assets or {}).get("characters", {}):
        return "郑教授"
    if any(alias in context for alias in ("郑教授", "郑老先生", "郑老", "郑叔")) and "郑教授" in (assets or {}).get("characters", {}):
        return "郑教授"
    if narrator_name and any(word in current_text for word in ("我", "我的", "为你", "你", "别", "吧")):
        return narrator_name
    characters = list((assets or {}).get("characters", {}).keys())
    candidates = [name for name in characters if name != narrator_name and not _is_snake_name(name)]
    if "朋友" in previous_text and len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    return "角色"


def _voice_timeline_for_board(script_plan: dict, plan_item: dict, style: str, narrator_name: str,
                              assets: dict | None = None) -> list[dict]:
    beats_by_id, _ = _script_maps(script_plan)
    timeline = []
    cursor = 0
    previous_text = _previous_context_for_plan_item(script_plan, plan_item)
    board_context = "".join(
        (beats_by_id.get(beat_id) or {}).get("text", "")
        for beat_id in plan_item.get("voice_beat_ids", [])
    )
    for index, beat_id in enumerate(plan_item.get("voice_beat_ids", []), start=1):
        source = beats_by_id.get(beat_id)
        if not source:
            continue
        duration = int(source.get("duration") or 1)
        beat_type = source.get("type") or "narration"
        speaker = source.get("speaker") or ("旁白" if beat_type == "narration" else "角色")
        if style == "first_person" and beat_type == "narration":
            speaker = narrator_name
        if beat_type == "dialogue":
            speaker = _resolve_dialogue_speaker(
                speaker,
                assets or {},
                narrator_name,
                previous_text + board_context,
                source.get("text", ""),
            )
        timeline.append({
            "beat_id": f"v{index:02d}",
            "source_beat_id": beat_id,
            "type": beat_type,
            "text": source.get("text", ""),
            "speaker": speaker,
            "start": cursor,
            "end": cursor + duration,
            "duration": duration,
        })
        cursor += duration
        previous_text = source.get("text", "") or previous_text
    return timeline


def _previous_context_for_plan_item(script_plan: dict, plan_item: dict, count: int = 3) -> str:
    beats = script_plan.get("voice_beats", [])
    beat_ids = plan_item.get("voice_beat_ids", [])
    if not beat_ids:
        return ""
    first_index = next((idx for idx, beat in enumerate(beats) if beat.get("beat_id") == beat_ids[0]), -1)
    if first_index <= 0:
        return ""
    return "".join(beat.get("text", "") for beat in beats[max(0, first_index - count):first_index])


def _source_trace_for_board(script_plan: dict, plan_item: dict) -> list[dict]:
    _, slices = _script_maps(script_plan)
    trace = []
    seen = set()
    for slice_id in plan_item.get("source_slice_ids", []):
        if slice_id in seen:
            continue
        seen.add(slice_id)
        item = slices.get(slice_id)
        if not item:
            continue
        trace.append({
            "slice_id": slice_id,
            "source_start": item.get("source_start"),
            "source_end": item.get("source_end"),
            "kind": item.get("kind"),
            "speaker": item.get("speaker"),
            "text": item.get("text", ""),
        })
    return trace


def _short_story_text(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _beat_window_text(beats: list[dict], start: int, end: int) -> str:
    if start < 0 or end <= start:
        return ""
    return "".join(beat.get("text", "") for beat in beats[start:end])


_GENERIC_PROP_REFERENCES = (
    "这份", "那份", "这只", "那只", "这个东西", "那个东西",
    "物件", "道具", "食物", "猎物", "投喂物", "营养餐", "脂肪含量",
)
_UNSUPPORTED_REPLACEMENT_OBJECTS = ("肉块", "生肉", "肉片", "肉条")
_ACTION_COMPLETION_PHRASES = ("转身离开", "离开门口", "离开工作室", "走出门口", "走开", "离去")
_OBJECT_VERBS = ("拿着", "提着", "接过", "递给", "放下", "夹着", "握着", "拎着", "捧着")


def _sentence_evidence(text: str, keyword: str, limit: int = 90) -> str:
    if not text or not keyword:
        return ""
    parts = [part.strip() for part in re.split(r"(?<=[。！？!?；;，,])", text) if part.strip()]
    for part in parts:
        if keyword in part:
            return _short_story_text(part, limit)
    index = text.find(keyword)
    if index >= 0:
        start = max(0, index - 28)
        end = min(len(text), index + len(keyword) + 42)
        return _short_story_text(text[start:end], limit)
    return ""


def _has_generic_prop_reference(text: str) -> bool:
    return any(term in (text or "") for term in _GENERIC_PROP_REFERENCES)


def _prop_aliases_from_current_text(current_text: str) -> list[str]:
    aliases = [term for term in _GENERIC_PROP_REFERENCES if term in (current_text or "")]
    return list(dict.fromkeys(aliases))


def _clean_object_name(raw: str) -> str:
    name = (raw or "").strip(" ，。！？；、：:的")
    if "的" in name:
        name = name.rsplit("的", 1)[-1]
    name = name[-8:].strip(" ，。！？；、：:的")
    return name if len(name) >= 2 else ""


def _extract_object_mentions(text: str, known_props: set[str]) -> list[dict]:
    mentions = []
    text = text or ""
    for match in re.finditer(r"[一这那](?:只|条|份|袋|个|把|部|台|根|张|块|枚|颗|本|箱)([^，。！？；、]{1,18})", text):
        name = _clean_object_name(match.group(1))
        if not name or name in known_props:
            continue
        mentions.append({"name": name, "evidence": _sentence_evidence(text, name) or _short_story_text(match.group(0), 90)})
    for verb in _OBJECT_VERBS:
        for match in re.finditer(rf"{verb}([^，。！？；、]{{1,10}})", text):
            name = _clean_object_name(match.group(1))
            if not name or name in known_props:
                continue
            mentions.append({"name": name, "evidence": _sentence_evidence(text, name) or _short_story_text(match.group(0), 90)})
    unique = []
    seen = set()
    for item in mentions:
        if item["name"] in seen:
            continue
        seen.add(item["name"])
        unique.append(item)
    return unique[:4]


def _story_state_context(
    current_text: str,
    previous_context: str,
    assets: dict,
) -> dict:
    active_props = []
    current_text = current_text or ""
    previous_context = previous_context or ""
    for name in assets.get("props", {}):
        in_current = name in current_text
        in_previous = name in previous_context
        if not in_current and not (in_previous and _has_generic_prop_reference(current_text)):
            continue
        aliases = [name]
        if not in_current:
            aliases.extend(_prop_aliases_from_current_text(current_text))
        evidence = _sentence_evidence(current_text, name) if in_current else _sentence_evidence(previous_context, name)
        state = "当前原文直接提及" if in_current else "由上一段延续到当前泛称或动作"
        active_props.append({
            "name": name,
            "state": state,
            "aliases": list(dict.fromkeys(alias for alias in aliases if alias)),
            "evidence": evidence,
        })

    active_characters = []
    for name in assets.get("characters", {}):
        aliases = _character_aliases(name)
        current_hit = next((alias for alias in aliases if alias and alias in current_text), "")
        previous_hit = next((alias for alias in aliases if alias and alias in previous_context), "")
        if not current_hit and not previous_hit:
            continue
        evidence_text = current_text if current_hit else previous_context
        keyword = current_hit or previous_hit
        active_characters.append({
            "name": name,
            "state": "当前原文出现" if current_hit else "由上一段承接",
            "evidence": _sentence_evidence(evidence_text, keyword),
        })

    known_props = set(assets.get("props", {}).keys())
    active_objects = []
    for item in _extract_object_mentions(current_text, known_props):
        active_objects.append({
            "name": item["name"],
            "state": "当前原文出现的临时物件",
            "evidence": item["evidence"],
        })
    current_object_names = {item["name"] for item in active_objects}
    for item in _extract_object_mentions(previous_context, known_props):
        if item["name"] in current_object_names:
            continue
        if item["name"] in current_text or _has_generic_prop_reference(current_text):
            active_objects.append({
                "name": item["name"],
                "state": "由上一段承接的临时物件",
                "evidence": item["evidence"],
            })

    open_actions = []
    stripped = current_text.strip()
    if stripped.endswith(("，", "、", "；", "：", ",")):
        open_actions.append({
            "action": "当前原文以承接标点结尾，动作或对白尚未结束",
            "evidence": _short_story_text(stripped, 90),
        })
    if active_props and _has_generic_prop_reference(current_text):
        names = "、".join(item["name"] for item in active_props)
        open_actions.append({
            "action": f"当前泛称需绑定到已有道具：{names}",
            "evidence": _short_story_text(current_text, 90),
        })

    return {
        "active_characters": active_characters[:4],
        "active_props": active_props[:4],
        "active_objects": active_objects[:4],
        "open_actions": open_actions[:4],
        "incomplete_source": bool(open_actions and stripped.endswith(("，", "、", "；", "：", ","))),
    }


def _story_continuity_for_board(
    script_plan: dict,
    plan_item: dict,
    voice_timeline: list[dict],
    source_trace: list[dict],
    assets: dict,
) -> dict:
    beats = script_plan.get("voice_beats", [])
    beat_ids = plan_item.get("voice_beat_ids", [])
    indexes = [idx for idx, beat in enumerate(beats) if beat.get("beat_id") in beat_ids]
    current_text = "".join(item.get("text", "") for item in source_trace)
    if not current_text:
        current_text = "".join(beat.get("text", "") for beat in voice_timeline)
    first_index = min(indexes) if indexes else -1
    just_happened = _beat_window_text(beats, max(0, first_index - 2), first_index) if first_index > 0 else ""
    previous_context = _previous_context_for_plan_item(script_plan, plan_item)
    current_stage = ""
    for name in assets.get("scenes", {}):
        if name and name in current_text:
            current_stage = name
            break
    return {
        "just_happened": _short_story_text(just_happened),
        "previous_final_panel": "",
        "now_happening": _short_story_text(current_text, 140),
        "current_stage": current_stage,
        "visible_identity_refs": [],
        "state_context": _story_state_context(current_text, previous_context, assets),
    }


def _allowed_characters_for_board(voice_timeline: list[dict], source_trace: list[dict], assets: dict,
                                  narrator_name: str, previous_context: str = "") -> list[str]:
    characters = assets.get("characters", {})
    direct_evidence = "".join(item.get("text", "") for item in source_trace)
    direct_evidence += "".join(beat.get("text", "") for beat in voice_timeline)
    context_evidence = previous_context + direct_evidence
    allowed = []
    for beat in voice_timeline:
        speaker = beat.get("speaker")
        if speaker in characters and speaker not in allowed:
            allowed.append(speaker)
    if narrator_name in characters and any(word in direct_evidence for word in ("我", "我的", "小沈", narrator_name)):
        allowed.append(narrator_name)
    for name in characters:
        if name in allowed:
            continue
        alias_evidence = context_evidence if _is_snake_name(name) else direct_evidence
        if any(alias and alias in alias_evidence for alias in _character_aliases(name)):
            allowed.append(name)
    if "玄墨" in characters and "玄墨" not in allowed and _has_snake_evidence(context_evidence):
        allowed.append("玄墨")
    return list(dict.fromkeys(allowed))


def _asset_summary(assets: dict, allowed_characters: list[str] | None = None) -> dict:
    character_names = allowed_characters or list(assets.get("characters", {}).keys())
    return {
        "characters": {
            name: assets.get("characters", {}).get(name, {}).get("seed", "")
            for name in character_names
            if name in assets.get("characters", {})
        },
        "scenes": {name: info.get("seed", "") for name, info in assets.get("scenes", {}).items()},
        "props": {name: info.get("seed", "") for name, info in assets.get("props", {}).items()},
    }


def _visible_identity_refs_for_board(board: dict, assets: dict) -> list[str]:
    refs = board.get("asset_refs", {})
    lines = []
    index = 1
    characters = sorted(refs.get("characters", []) or [], key=lambda name: (0 if _is_snake_name(name) else 1, name))
    for name in characters:
        info = assets.get("characters", {}).get(name, {})
        seed = info.get("seed", "") if isinstance(info, dict) else ""
        if _is_snake_character_asset(name, seed):
            lines.append(f"图片{index}：{name}，黑色巨蟒，故事中唯一的蛇")
        else:
            lines.append(f"图片{index}：{name}，{_short_story_text(seed, 36) or '当前可见人物'}")
        index += 1
    scene = refs.get("scene")
    if scene:
        lines.append(f"图片{index}：{scene}，当前场景参考")
        index += 1
    for prop in refs.get("props", []) or []:
        lines.append(f"图片{index}：{prop}，当前道具参考")
        index += 1
    return lines


def _last_panel_summary(board: dict | None) -> str:
    if not board:
        return ""
    shots = board.get("shot_timeline", []) or []
    if not shots:
        return ""
    shot = shots[-1]
    parts = [
        _short_story_text(shot.get("visual", ""), 100),
    ]
    characters = shot.get("characters") or []
    if characters:
        parts.append(f"可见角色：{'、'.join(characters)}")
    camera = shot.get("camera")
    if camera:
        parts.append(f"镜头：{camera}")
    return "；".join(part for part in parts if part)


def _spatial_rules_for_board(previous_board: dict | None, board: dict) -> list[str]:
    rules = [
        "保持同一场景内人物、道具、门窗、饲养箱等关键物体的左右/前后关系稳定，不要左右互换。",
    ]
    refs = board.get("asset_refs", {}) or {}
    scene = refs.get("scene")
    if scene:
        rules.append(f"当前场景固定为：{scene}。镜头变化只改变景别和机位，不改变空间结构。")
    props = refs.get("props") or []
    if props:
        rules.append(f"关键道具位置需连续：{'、'.join(props)}。")
    previous = _last_panel_summary(previous_board)
    if previous:
        rules.append(f"承接上一板最后位置：{previous}。")
    return rules


def _refresh_storyboard_prompt_context(boards: list[dict], assets: dict) -> None:
    previous_board = None
    for board in boards:
        continuity = board.setdefault("story_continuity", {})
        continuity["visible_identity_refs"] = _visible_identity_refs_for_board(board, assets)
        continuity["previous_final_panel"] = _last_panel_summary(previous_board)
        continuity["spatial_rules"] = _spatial_rules_for_board(previous_board, board)
        _prepare_board_prompts(board)
        previous_board = board


def _planned_board_prompt(
    style: str,
    board_id: str,
    page: int,
    total_pages: int,
    voice_timeline: list[dict],
    board_duration: int,
    assets: dict,
    source_trace: list[dict] | None = None,
    allowed_characters: list[str] | None = None,
    story_continuity: dict | None = None,
) -> str:
    style_instruction = "第一人称主角内心独白" if style == "first_person" else "第三人称旁白"
    voice_json = json.dumps(voice_timeline, ensure_ascii=False, indent=2)
    source_json = json.dumps(source_trace or [], ensure_ascii=False, indent=2)
    assets_json = json.dumps(_asset_summary(assets, allowed_characters), ensure_ascii=False, indent=2)
    continuity_payload = {
        "just_happened": (story_continuity or {}).get("just_happened", ""),
        "now_happening": (story_continuity or {}).get("now_happening", ""),
        "state_context": (story_continuity or {}).get("state_context", {}),
    }
    continuity_json = json.dumps(continuity_payload, ensure_ascii=False, indent=2)
    compact_rule = "compact_page=true，shot_timeline 生成 3 个 shot" if board_duration < 5 else "compact_page=false，shot_timeline 生成 5 个 shot"
    technique_options = technique_options_for_prompt()
    palette_options = "suspense_cold_blue（悬疑/惊悚/夜戏）, family_warm_gray（家庭/日常/对白）, hospital_cold_white（医院/办公室/冷白室内）, domestic_brown_gray（压抑家庭/旧屋/饭桌冲突）"
    return f"""你是解说漫分镜导演。只输出合法 JSON，不要 markdown，不要解释。

任务：为一个固定声音时间轴设计画面分镜。不要改写旁白/对白，不要输出 voice_timeline。
旁白风格：{style_instruction}
board_id：{board_id}
page：{page}/{total_pages}
board_duration：{board_duration} 秒
镜头数量规则：{compact_rule}

固定声音时间轴：
{voice_json}

对应原文切片：
{source_json}

叙事连续性状态：
{continuity_json}

可用资产名称：
{assets_json}

导演手法：
- 每个 shot 必须体现明确镜头语言：景别、机位、运镜、主体调度和构图焦点。
- camera 不要只写“近景/中景”，应尽量写成“低角度近景、缓慢推近、跟拍、过肩、主观视角、俯拍特写”等可执行拍法。
- purpose 必须说明这个镜头为什么这样拍：铺垫、反应、揭示、压迫感、情绪落点或转场。
- 同一 board 内镜头应有节奏变化，不要连续使用同一种景别和机位。
- technique_id 是可选镜头技法宏，只能从可用技法中选择；普通交代镜头 technique_id 必须为空。
- 每张 board 最多 1-2 个非空 technique_id，只用于关键情绪、反转、危险逼近、线索出现、动作爆点。
- 可用技法：{technique_options}
- palette_id 是本 board 的视频色板参考，只能从可用色板中选择一个；同一场戏内优先保持一致。
- 可用色板：{palette_options}

输出结构：
{{
  "video_goal": "一句中文目标",
  "palette_id": "suspense_cold_blue",
  "shot_timeline": [
    {{
      "shot_id": "s01",
      "start": 0,
      "end": 2,
      "duration": 2,
      "voice_refs": ["v01"],
      "visual": "不超过22个中文的画面",
      "camera": "近景/中景/特写/远景/俯拍/跟拍",
      "characters": ["必须来自资产角色名"],
      "scene": "必须来自资产场景名，无法判断可为空字符串",
      "match_strategy": "sync",
      "purpose": "一句中文镜头意图",
      "audio_behavior": "narration_sync",
      "continuity_from_previous": null,
      "transition_type": null,
      "technique_id": ""
    }}
  ],
  "asset_refs": {{
    "characters": ["必须来自资产角色名"],
    "scene": "必须来自资产场景名，无法判断可为空字符串",
    "props": ["必须来自资产道具名"]
  }}
}}

硬规则：
1. shot_timeline 必须完整覆盖 0 到 {board_duration}，无间隙无重叠
2. 第一个 shot 的 continuity_from_previous 和 transition_type 必须为 null
3. 每个 voice_timeline beat 至少被一个 shot 的 voice_refs 引用
4. visual 写具体画面，不要出现字幕、文字、水印、分镜编号
5. asset_refs 只放本 board 画面真正需要保持一致的角色、场景、道具
5a. asset_refs.characters 只能从“可用资产名称.characters”中选择；如果原文没有证据，禁止加入其他角色
6. match_strategy 只能用 sync/supplement/contrast/foreshadow/reaction_first/reveal/emotional_landing/transition
7. audio_behavior 只能用 narration_sync/narration_over/dialogue_sync/dialogue_offscreen/phone_dialogue/ambient_only/sound_lead_in/dramatic_silence/ambient_transition
8. dialogue 的 speaker 必须是真实角色名；不能把“恐惧语气、低声、电话那头、朋友、他、她”这类状态词/关系词当 speaker
9. 如果对白来自手机或电话，当前镜头拍听电话者时 audio_behavior 必须用 phone_dialogue 或 dialogue_offscreen，不要写 dialogue_sync
10. 对“他/她/朋友/电话那头”要结合上一板和固定声音时间轴回溯指代；例如上一句是“开宠物医院的朋友”，下一句“他吼道”就是朋友在电话里说话，不是主角自言自语
11. 必须遵守叙事连续性状态：active_props 是当前仍在场或被指代的道具；active_characters 是当前仍有证据承接的角色；open_actions 是尚未结束的动作或对白
12. 不能把状态里的既有道具替换成无证据的新物体；如果当前原文用泛称称呼道具，要沿用 state_context.active_props 的 name
13. 如果 state_context.incomplete_source=true，当前原文尚未说完，只画已发生动作，不要补完下一步离开、进入、攻击、揭示等后续事件
14. technique_id 可以为空；每张 board 最多 1-2 个非空 technique_id；不要为了炫技破坏原文画面匹配
15. palette_id 不能为空；优先按场景/类型选择，不要在同一场戏内频繁切换"""


def _shot_ranges(duration: int, count: int) -> list[tuple[int, int]]:
    base = duration // count
    remainder = duration % count
    ranges = []
    cursor = 0
    for idx in range(count):
        step = base + (1 if idx < remainder else 0)
        start = cursor
        end = cursor + step
        ranges.append((start, end))
        cursor = end
    return ranges


def _normalize_refs(refs: dict, assets: dict) -> dict:
    characters = set(assets.get("characters", {}).keys())
    scenes = set(assets.get("scenes", {}).keys())
    props = set(assets.get("props", {}).keys())
    if not isinstance(refs, dict):
        refs = {}
    def as_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            return [value]
        return []
    raw_scene = refs.get("scene")
    if isinstance(raw_scene, list):
        raw_scene = next((item for item in raw_scene if isinstance(item, str) and item in scenes), "")
    if not isinstance(raw_scene, str):
        raw_scene = ""
    return {
        "characters": [name for name in as_list(refs.get("characters")) if name in characters][:3],
        "scene": raw_scene if raw_scene in scenes else "",
        "props": [name for name in as_list(refs.get("props")) if name in props][:3],
    }


def _fallback_asset_refs(voice_timeline: list[dict], assets: dict) -> dict:
    text = "".join(beat.get("text", "") for beat in voice_timeline)
    refs = {
        "characters": [name for name in assets.get("characters", {}) if name in text][:3],
        "scene": "",
        "props": [name for name in assets.get("props", {}) if name in text][:3],
    }
    for name in assets.get("scenes", {}):
        if name in text:
            refs["scene"] = name
            break
    if not refs["characters"] and "玄墨" in assets.get("characters", {}):
        refs["characters"] = ["玄墨"]
    return refs


def _sanitize_board_character_refs(board: dict, allowed_characters: list[str]) -> tuple[list[str], list[str]]:
    allowed = set(allowed_characters)
    refs = board.setdefault("asset_refs", {})
    original = list(refs.get("characters", []) or [])
    filtered = [name for name in original if name in allowed]
    removed = [name for name in original if name not in allowed]
    refs["characters"] = filtered
    for shot in board.get("shot_timeline", []) or []:
        shot_chars = list(shot.get("characters", []) or [])
        shot["characters"] = [name for name in shot_chars if name in allowed]
        removed.extend(name for name in shot_chars if name not in allowed)
    return filtered, list(dict.fromkeys(removed))


def _fallback_shots(board_duration: int, voice_timeline: list[dict], refs: dict) -> list[dict]:
    count = 3 if board_duration < 5 else 5
    voice_ids = [beat.get("beat_id") for beat in voice_timeline if beat.get("beat_id")]
    ranges = _shot_ranges(board_duration, count)
    cameras = ["近景", "中景", "特写", "跟拍", "远景"]
    strategies = ["sync", "supplement", "reaction_first", "foreshadow", "emotional_landing"]
    visuals = _fallback_shot_visuals(voice_timeline, refs, count)
    shots = []
    for idx, (start, end) in enumerate(ranges, start=1):
        shot_refs = []
        for beat in voice_timeline:
            if beat.get("end", 0) > start and beat.get("start", 0) < end:
                shot_refs.append(beat.get("beat_id"))
        if not shot_refs and voice_ids:
            shot_refs = [voice_ids[min(idx - 1, len(voice_ids) - 1)]]
        shots.append({
            "shot_id": f"s{idx:02d}",
            "start": start,
            "end": end,
            "duration": end - start,
            "voice_refs": shot_refs,
            "visual": visuals[min(idx - 1, len(visuals) - 1)],
            "camera": cameras[min(idx - 1, len(cameras) - 1)],
            "characters": refs.get("characters", []),
            "scene": refs.get("scene", ""),
            "match_strategy": strategies[min(idx - 1, len(strategies) - 1)],
            "purpose": "承接声音内容并推进情绪",
            "audio_behavior": "dialogue_sync" if any(beat.get("type") == "dialogue" for beat in voice_timeline) else "narration_sync",
            "continuity_from_previous": None if idx == 1 else "延续上一镜",
            "transition_type": None if idx == 1 else "cut",
        })
    return shots


def _fallback_shot_visuals(voice_timeline: list[dict], refs: dict, count: int) -> list[str]:
    text = _compact_source_text("".join(beat.get("text", "") for beat in voice_timeline))
    characters = refs.get("characters") or []

    pattern_visuals = [
        (
            ["这一养", "七年"],
            [
                "画面做时间流逝感，幼小的玄墨在饲养箱角落逐渐长大",
                "沈砚给饲养箱记录本写下年份，旁边的玄墨体型已明显变粗",
                "成年玄墨安静盘在饲养箱中，沈砚站在箱外观察",
            ],
        ),
        (
            ["基因突变", "缅甸蟒", "纯黑色"],
            [
                "玄墨纯黑色身体盘踞在饲养箱里，鳞片在灯下泛出暗紫金属光",
                "特写玄墨没有普通蟒蛇斑纹的黑色鳞片，质感深邃光滑",
                "沈砚隔着玻璃观察玄墨，画面强调它与普通缅甸蟒不同的纯黑外观",
                "玄墨缓慢抬头，黑色鳞片反射冷光，显得美丽又危险",
                "饲养箱全景中，玄墨像一条黑色绸缎盘在环境中央",
            ],
        ),
        (
            ["完全成年", "体长", "石雕"],
            [
                "成年玄墨盘踞在大型饲养箱里，粗壮黑色身体占据箱体中央",
                "低角度突出玄墨接近两米五的体长，身体从画面前景延伸到后方",
                "特写玄墨最粗处堪比成年人小腿，鳞片层层贴合",
                "玄墨一动不动盘在箱内，姿态像沉默的黑色石雕",
                "沈砚站在饲养箱外与玄墨形成体型对比，神情平静",
            ],
        ),
        (
            ["冷血动物", "养不熟", "感情"],
            [
                "沈砚站在饲养箱前注视玄墨，神情不认同外界对蛇的偏见",
                "玄墨在箱内安静抬头看向沈砚，玻璃隔开两者",
                "画面用旁人背影或模糊资料页象征外界刻板印象，重点仍在沈砚和玄墨",
                "沈砚伸手贴近玻璃，玄墨的蛇头停在玻璃另一侧",
                "沈砚低头记录玄墨状态，表情坚定而温和",
            ],
        ),
        (
            ["乌黑", "缓缓舒展", "信子", "投食口"],
            [
                "饲养箱内，玄墨乌黑的身体从盘绕姿态缓缓舒展开",
                "玄墨优雅抬起椭圆形头部，黑色鳞片贴着箱底移动",
                "特写玄墨分叉的信子快速探出，感知投食口方向",
                "玄墨顺着打开的投食口慢慢游出，蛇身一节节滑过边缘",
                "饲养箱外只见玄墨前半身探出，环境保持安静紧张",
            ],
        ),
        (
            ["不主动攻击", "温顺"],
            [
                "沈砚把手停在玄墨附近，玄墨没有攻击，只安静贴着他的手臂移动",
                "玄墨从沈砚手边缓慢爬过，蛇头保持低伏平静",
                "沈砚近距离观察玄墨，表情放松，动作熟练",
                "玄墨盘在饲养箱边缘，身体姿态温顺稳定",
                "沈砚轻轻收回手，玄墨依旧没有攻击动作",
            ],
        ),
        (
            ["手臂", "攀爬", "鳞片", "安心"],
            [
                "玄墨顺着沈砚手臂向上攀爬，黑色蛇身贴着皮肤移动",
                "特写冰凉光滑的黑色鳞片贴在沈砚手臂上",
                "沈砚低头看着玄墨，表情平静甚至安心",
                "玄墨的蛇头靠近沈砚肩侧，身体稳定缠住手臂支撑",
                "沈砚站在饲养箱旁，玄墨伏在他的手臂上，画面安静亲密",
            ],
        ),
        (
            ["白鼠", "瞥", "扭过头", "滑下"],
            [
                "沈砚把活蹦乱跳的白鼠递到玄墨面前，玄墨只冷淡瞥了一眼",
                "白鼠在镊子旁挣动，玄墨没有张口捕食",
                "玄墨扭过头避开白鼠，黑色蛇头偏向一侧",
                "玄墨顺着沈砚手臂向下滑落，沈砚露出疑惑",
                "玄墨离开沈砚手臂，白鼠仍停在投喂位置",
            ],
        ),
        (
            ["白鼠", "凑到", "烦躁", "饲养箱"],
            [
                "沈砚把白鼠凑到玄墨嘴边，手中的镊子停在近处",
                "玄墨身体绷紧，显出异常烦躁的姿态",
                "玄墨猛地甩头避开白鼠，沈砚的手微微后撤",
                "玄墨迅速爬向饲养箱入口，黑色身体贴着箱边移动",
                "玄墨回到饲养箱内，沈砚站在箱外困惑观察",
            ],
        ),
        (
            ["郑", "教授", "安全距离"],
            [
                "郑教授来到沈砚工作室门口，目光先落向饲养箱",
                "沈砚在工作室里与郑教授交谈，饲养箱摆在一旁",
                "郑教授隔着一段距离观察玄墨，身体下意识后倾",
                "玄墨在饲养箱内盘绕不动，郑教授保持安全距离",
                "沈砚看向郑教授，郑教授仍谨慎地站在远离饲养箱的位置",
            ],
        ),
        (
            ["胆子", "蟒蛇", "心情不好"],
            [
                "郑教授看着饲养箱里的玄墨，表情严肃地提醒沈砚",
                "沈砚站在饲养箱旁听郑教授说话，神情仍然放松",
                "玄墨庞大的黑色身体盘在箱内，体型占据画面重点",
                "郑教授抬手示意风险，与饲养箱保持明显距离",
                "沈砚转头看向玄墨，对郑教授的担忧不以为意",
            ],
        ),
    ]
    for keys, visuals in pattern_visuals:
        if all(key in text for key in keys):
            return visuals[:count]

    snippets = _visual_snippets_from_text(text, count)
    subject = "、".join(characters[:2]) if characters else "当前角色"
    if not snippets:
        snippets = ["本段原文事件"]
    visuals = []
    for idx in range(count):
        snippet = snippets[min(idx, len(snippets) - 1)]
        if "玄墨" in characters and len(characters) == 1:
            visuals.append(f"玄墨在当前场景中完成这一段动作，画面重点：{snippet}")
        elif "玄墨" in characters and "沈砚" in characters:
            visuals.append(f"沈砚观察玄墨的状态变化，画面重点：{snippet}")
        elif characters:
            visuals.append(f"{subject}在当前场景中推进这一段情节，画面重点：{snippet}")
        else:
            visuals.append(f"当前场景用具体动作呈现这一段内容，画面重点：{snippet}")
    return visuals


def _compact_source_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _visual_snippets_from_text(text: str, count: int) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[。！？!?；;，,])", text) if part.strip()]
    if not parts and text:
        parts = [text]
    snippets = []
    for part in parts:
        snippet = part[:42]
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= count:
            break
    return snippets


def _state_context_from_board(board: dict) -> dict:
    continuity = board.get("story_continuity") or {}
    state = continuity.get("state_context") or {}
    return state if isinstance(state, dict) else {}


def _active_state_props(board: dict, assets: dict) -> list[dict]:
    props = []
    available = assets.get("props", {})
    for item in (_state_context_from_board(board).get("active_props") or []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name in available:
            props.append(item)
    return props


def _board_source_text(board: dict) -> str:
    text = "".join(item.get("text", "") for item in board.get("source_trace", []) or [])
    if text:
        return text
    return "".join(beat.get("text", "") for beat in board.get("voice_timeline", []) or [])


def _apply_story_state_constraints(board: dict, assets: dict) -> list[str]:
    warnings = []
    state = _state_context_from_board(board)
    active_props = _active_state_props(board, assets)
    current_text = _board_source_text(board)
    refs = board.setdefault("asset_refs", {})
    refs.setdefault("props", [])

    for prop in active_props:
        name = prop.get("name")
        aliases = [alias for alias in (prop.get("aliases") or []) if alias]
        aliases.append(name)
        if any(alias in current_text for alias in aliases) or any(
            alias in (shot.get("visual") or "")
            for shot in board.get("shot_timeline", []) or []
            for alias in aliases
        ):
            if name not in refs["props"]:
                refs["props"].append(name)
                warnings.append(f"连续性状态补充道具：{name}")

    if active_props:
        primary_prop = active_props[0].get("name")
        for shot in board.get("shot_timeline", []) or []:
            visual = shot.get("visual") or ""
            if not visual:
                continue
            if any(term in visual for term in _UNSUPPORTED_REPLACEMENT_OBJECTS):
                if any(alias in visual for alias in ("营养餐", "食物", "猎物", "投喂物")):
                    repaired = f"特写沈砚手中作为当前投喂物的{primary_prop}，保持上一板道具连续"
                else:
                    repaired = f"特写当前既有道具{primary_prop}，保持上一板道具连续"
                if repaired != visual:
                    shot["visual"] = repaired
                    warnings.append(f"连续性状态修正无证据替换物：{visual} -> {repaired}")

    if state.get("incomplete_source"):
        for shot in board.get("shot_timeline", []) or []:
            visual = shot.get("visual") or ""
            repaired = visual
            for phrase in _ACTION_COMPLETION_PHRASES:
                if phrase in repaired:
                    if "门口" in repaired:
                        repaired = re.sub(r"，?[^，。；]*" + re.escape(phrase) + r"[^，。；]*", "，仍停在门口，没有补完离开动作", repaired)
                    else:
                        repaired = repaired.replace(phrase, "停在当前动作中，不补完后续")
            if repaired != visual:
                shot["visual"] = repaired
                warnings.append(f"未完成原文阻止补完动作：{visual} -> {repaired}")

    refs["props"] = [name for name in dict.fromkeys(refs.get("props", [])) if name in assets.get("props", {})][:3]
    return list(dict.fromkeys(warnings))


def _build_planned_board(
    *,
    plan_item: dict,
    page: int,
    total_pages: int,
    voice_timeline: list[dict],
    assets: dict,
    llm_result: dict | None,
    source_trace: list[dict] | None = None,
    allowed_characters: list[str] | None = None,
    story_continuity: dict | None = None,
) -> dict:
    voice_duration = sum(int(beat.get("duration") or 0) for beat in voice_timeline)
    board_duration = max(3, voice_duration)
    compact = board_duration < 5
    refs = _normalize_refs((llm_result or {}).get("asset_refs", {}), assets)
    if not refs.get("characters") and not refs.get("scene") and not refs.get("props"):
        refs = _fallback_asset_refs(voice_timeline, assets)
    removed_by_evidence = []
    if allowed_characters:
        removed_by_evidence = [name for name in refs.get("characters", []) if name not in allowed_characters]
        refs["characters"] = [name for name in refs.get("characters", []) if name in allowed_characters]
        if "玄墨" in allowed_characters and "玄墨" not in refs["characters"]:
            refs["characters"].append("玄墨")
        if not refs["characters"]:
            refs["characters"] = allowed_characters[:3]
    shots = (llm_result or {}).get("shot_timeline") or []
    video_goal = (llm_result or {}).get("video_goal") or "呈现本段关键情节"
    board = {
        "board_id": plan_item.get("board_id") or f"b{page:04d}",
        "page": page,
        "total_pages": total_pages,
        "compact_page": compact,
        "voice_duration": voice_duration,
        "visual_duration": board_duration,
        "board_duration": board_duration,
        "video_goal": video_goal,
        "palette_id": (llm_result or {}).get("palette_id") or "",
        "voice_timeline": voice_timeline,
        "shot_timeline": shots,
        "storyboard_image": {"status": "needed", "prompt": "", "task_id": None, "url": None, "local_path": None},
        "video": {"status": "needed", "duration": board_duration, "prompt": "", "task_id": None, "url": None, "local_path": None},
        "asset_refs": refs,
        "source_trace": source_trace or [],
        "story_continuity": story_continuity or {},
    }
    if validate_board_page(board):
        board["shot_timeline"] = _fallback_shots(board_duration, voice_timeline, refs)
        board["video_goal"] = video_goal or "呈现本段关键情节"
    state_warnings = _apply_story_state_constraints(board, assets)
    filtered, removed = _sanitize_board_character_refs(board, allowed_characters or list(assets.get("characters", {}).keys()))
    removed = list(dict.fromkeys(removed_by_evidence + removed))
    warnings = list(state_warnings)
    if removed:
        warnings.append(f"已移除无原文证据角色：{'、'.join(removed)}")
    board["review"] = {
        "source_excerpt": "".join(item.get("text", "") for item in board["source_trace"]),
        "allowed_characters": allowed_characters or filtered,
        "removed_characters": removed,
        "warnings": warnings,
    }
    board["story_continuity"] = {
        **(board.get("story_continuity") or {}),
        "visible_identity_refs": _visible_identity_refs_for_board(board, assets),
    }
    _prepare_board_prompts(board)
    return board


async def _generate_planned_board_page(
    api_key: str,
    style: str,
    plan_item: dict,
    page: int,
    total_pages: int,
    script_plan: dict,
    assets: dict,
    narrator_name: str,
) -> tuple[int, dict, list[str]]:
    voice_timeline = _voice_timeline_for_board(script_plan, plan_item, style, narrator_name, assets)
    voice_duration = sum(int(beat.get("duration") or 0) for beat in voice_timeline)
    board_duration = max(3, voice_duration)
    source_trace = _source_trace_for_board(script_plan, plan_item)
    previous_context = _previous_context_for_plan_item(script_plan, plan_item)
    allowed_characters = _allowed_characters_for_board(
        voice_timeline, source_trace, assets, narrator_name, previous_context
    )
    story_continuity = _story_continuity_for_board(script_plan, plan_item, voice_timeline, source_trace, assets)
    warnings = []
    result = None
    try:
        result = await _call_llm(
            api_key,
            _planned_board_prompt(
                style,
                plan_item.get("board_id") or f"b{page:04d}",
                page,
                total_pages,
                voice_timeline,
                board_duration,
                assets,
                source_trace,
                allowed_characters,
                story_continuity,
            ),
            "为这个固定声音时间轴生成画面分镜 JSON。",
            attempts=2,
        )
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
        warnings.append(f"{plan_item.get('board_id')} 使用本地兜底分镜：{detail.get('error_type') or detail.get('message')}")

    try:
        board = _build_planned_board(
            plan_item=plan_item,
            page=page,
            total_pages=total_pages,
            voice_timeline=voice_timeline,
            assets=assets,
            llm_result=result,
            source_trace=source_trace,
            allowed_characters=allowed_characters,
            story_continuity=story_continuity,
        )
    except Exception as e:
        warnings.append(f"{plan_item.get('board_id')} 解析分镜失败，使用本地兜底：{type(e).__name__}: {e}")
        board = _build_planned_board(
            plan_item=plan_item,
            page=page,
            total_pages=total_pages,
            voice_timeline=voice_timeline,
            assets=assets,
            llm_result=None,
            source_trace=source_trace,
            allowed_characters=allowed_characters,
            story_continuity=story_continuity,
        )
    validation_errors = validate_board_page(board)
    if validation_errors:
        warnings.append(f"{board.get('board_id')} 校验警告：{validation_errors}")
    for warning in board.get("review", {}).get("warnings", []):
        warnings.append(f"{board.get('board_id')} 复检：{warning}")
    return page - 1, board, warnings


def _single_board_prompt(base_prompt: str, page: int, total_pages: int) -> str:
    return "\n\n".join([
        base_prompt,
        "## 本次生成范围",
        f"只生成第 {page} 页，共 {total_pages} 页。",
        '输出仍必须是 {"boards": [BOARD_PAGE]}，boards 数组只能包含 1 个对象。',
        f"该对象 page 必须等于 {page}，total_pages 必须等于 {total_pages}，board_id 后缀必须是 _p{page:02d}。",
        "不要生成其他页，避免 JSON 过长。",
    ])


def _compact_single_board_prompt(
    style: str,
    seg_key: str,
    scene_location: str,
    characters: list,
    page: int,
    total_pages: int,
) -> str:
    style_instruction = "第三人称旁白" if style == "third_person" else "第一人称旁白"
    return f"""你是解说漫分镜师。请只输出合法 JSON，不要 markdown，不要解释。

任务：根据故事原文，为段落 {seg_key} 生成第 {page}/{total_pages} 页视频 board。
风格：{style_instruction}
场景：{scene_location}
角色：{", ".join(characters)}

输出必须是这个结构：
{{
  "boards": [{{
    "board_id": "{seg_key}_p{page:02d}",
    "page": {page},
    "total_pages": {total_pages},
    "compact_page": false,
    "voice_duration": 10,
    "visual_duration": 10,
    "board_duration": 10,
    "video_goal": "一句中文目标",
    "voice_timeline": [
      {{"beat_id":"v01","type":"narration","text":"一句旁白","speaker":"旁白","start":0,"end":5,"duration":5}},
      {{"beat_id":"v02","type":"narration","text":"一句旁白","speaker":"旁白","start":5,"end":10,"duration":5}}
    ],
    "shot_timeline": [
      {{"shot_id":"s01","start":0,"end":2,"duration":2,"voice_refs":["v01"],"visual":"一句画面","camera":"近景","characters":[],"scene":"{scene_location}","match_strategy":"sync","purpose":"一句意图","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null}},
      {{"shot_id":"s02","start":2,"end":4,"duration":2,"voice_refs":["v01"],"visual":"一句画面","camera":"中景","characters":[],"scene":"{scene_location}","match_strategy":"supplement","purpose":"一句意图","audio_behavior":"narration_over","continuity_from_previous":"延续上一镜","transition_type":"cut"}},
      {{"shot_id":"s03","start":4,"end":6,"duration":2,"voice_refs":["v02"],"visual":"一句画面","camera":"特写","characters":[],"scene":"{scene_location}","match_strategy":"reaction_first","purpose":"一句意图","audio_behavior":"narration_over","continuity_from_previous":"延续上一镜","transition_type":"cut"}},
      {{"shot_id":"s04","start":6,"end":8,"duration":2,"voice_refs":["v02"],"visual":"一句画面","camera":"俯拍","characters":[],"scene":"{scene_location}","match_strategy":"foreshadow","purpose":"一句意图","audio_behavior":"ambient_only","continuity_from_previous":"延续上一镜","transition_type":"cut"}},
      {{"shot_id":"s05","start":8,"end":10,"duration":2,"voice_refs":["v02"],"visual":"一句画面","camera":"远景","characters":[],"scene":"{scene_location}","match_strategy":"emotional_landing","purpose":"一句意图","audio_behavior":"narration_sync","continuity_from_previous":"延续上一镜","transition_type":"cut"}}
    ],
    "storyboard_image": {{"status":"needed","prompt":"","task_id":null,"url":null,"local_path":null}},
    "video": {{"status":"needed","duration":10,"prompt":"","task_id":null,"url":null,"local_path":null}},
    "asset_refs": {{"characters": {json.dumps(characters, ensure_ascii=False)}, "scene": "{scene_location}", "props": []}}
  }}]
}}

硬规则：只输出 1 个 board；shot_timeline 必须正好 5 个镜头；每个 visual 不超过 22 个中文字符；不要输出长段落。"""


@router.post("/decompose")
async def decompose_project(req: DecomposeRequest):
    """Run multi-step LLM decomposition on the project's source text."""
    project_dir = _check_project(req.project_name)
    data = pl.read_pipeline(project_dir)
    source_text = data.get("source_text", "")
    if not source_text:
        raise HTTPException(status_code=400, detail="项目没有 source_text，无法拆解")
    if req.narration_style in {"third_person", "first_person"}:
        data["narration_style"] = req.narration_style

    _write_decompose_progress(
        project_dir,
        data,
        status="running",
        stage="script_plan",
        message="正在规划声音时长和 board 数量",
    )
    script_plan, script_plan_changed = _ensure_script_plan(data)
    if script_plan_changed:
        data["script_plan"] = script_plan
    _write_decompose_progress(
        project_dir,
        data,
        status="running",
        stage="script_plan",
        message="声音规划完成，准备检查 LLM 配置",
        current=len(script_plan.get("board_plan", [])),
        total=len(script_plan.get("board_plan", [])),
    )

    try:
        api_key = _require_api_key()
    except HTTPException as e:
        _write_decompose_failure(project_dir, data, "script_plan", e)
        raise
    style = data.get("narration_style", "third_person")

    # ── Step 1: Extract assets ──
    existing_assets = data.get("assets", {})
    if any(existing_assets.get(bucket) for bucket in ("characters", "scenes", "props")):
        assets = normalize_assets({"assets": existing_assets})["assets"]
        logger.info(
            "Step 1: Reusing assets: %s chars, %s scenes, %s props",
            len(assets["characters"]),
            len(assets["scenes"]),
            len(assets["props"]),
        )
        _write_decompose_progress(
            project_dir,
            data,
            status="running",
            stage="assets",
            message=f"复用已有资产：{len(assets['characters'])} 角色、{len(assets['scenes'])} 场景、{len(assets['props'])} 道具",
            current=1,
            total=3,
        )
    else:
        logger.info("Step 1: Extracting assets...")
        _write_decompose_progress(
            project_dir,
            data,
            status="running",
            stage="assets",
            message="正在从故事里提取角色、场景和道具",
            current=0,
            total=3,
        )
        step1_prompt = build_step1_prompt(style)
        try:
            assets_result = await _call_llm(api_key, step1_prompt, source_text)
        except HTTPException as e:
            _write_decompose_failure(project_dir, data, "assets", e)
            raise

        assets = {
            "characters": assets_result.get("characters", {}),
            "scenes": assets_result.get("scenes", {}),
            "props": assets_result.get("props", {}),
        }
        assets = normalize_assets({"assets": assets})["assets"]
        logger.info(f"Step 1 done: {len(assets['characters'])} chars, {len(assets['scenes'])} scenes, {len(assets['props'])} props")

        data["assets"] = assets
        _write_decompose_progress(
            project_dir,
            data,
            status="running",
            stage="assets",
            message=f"资产提取完成：{len(assets['characters'])} 角色、{len(assets['scenes'])} 场景、{len(assets['props'])} 道具",
            current=3,
            total=3,
        )

    # ── Step 2/3: Build deterministic board structure, then generate each board page ──
    board_plan = script_plan.get("board_plan", [])
    total_boards = len(board_plan)
    existing_segments = data.get("narration_segments", {})
    segments_outline = _build_segments_outline(board_plan, script_plan=script_plan)
    data["_decomposition_outline"] = {
        "segments": segments_outline,
        "script_plan_checksum": script_plan.get("source_checksum"),
        "planner_version": script_plan.get("planner_version"),
        "mode": "planned_boards",
    }
    data["script_plan"] = script_plan
    data["assets"] = assets
    _write_decompose_progress(
        project_dir,
        data,
        status="running",
        stage="outline",
        message=f"本地大纲完成：{len(segments_outline)} 组，{total_boards} 个 board；跳过大 JSON 大纲生成",
        current=len(segments_outline),
        total=len(segments_outline),
        extra={"segments_planned": len(segments_outline), "boards_planned": total_boards},
    )

    validation_errors = []
    narrator_name = _protagonist_name(assets)
    boards_by_index: list[dict | None] = [None] * total_boards
    for seg in existing_segments.values():
        for board in seg.get("boards", []):
            page = board.get("page")
            if isinstance(page, int) and 1 <= page <= total_boards:
                boards_by_index[page - 1] = board
    semaphore = asyncio.Semaphore(_decompose_concurrency())

    async def generate_one(index: int, plan_item: dict):
        async with semaphore:
            logger.info("Step 3: Generating planned board %s/%s", index + 1, total_boards)
            return await _generate_planned_board_page(
                api_key,
                style,
                plan_item,
                index + 1,
                total_boards,
                script_plan,
                assets,
                narrator_name,
            )

    _write_decompose_progress(
        project_dir,
        data,
        status="running",
        stage="boards",
        message=f"开始并发生成故事板：0/{total_boards}，并发 {_decompose_concurrency()}",
        current=0,
        total=total_boards,
        extra={"segments_planned": len(segments_outline), "boards_planned": total_boards},
    )

    tasks = [
        asyncio.create_task(generate_one(idx, item))
        for idx, item in enumerate(board_plan)
        if boards_by_index[idx] is None
    ]
    completed = sum(1 for item in boards_by_index if item is not None)
    if completed:
        _write_decompose_progress(
            project_dir,
            data,
            status="running",
            stage="boards",
            message=f"继续生成故事板：已复用 {completed}/{total_boards}",
            current=completed,
            total=total_boards,
            extra={"segments_planned": len(segments_outline), "boards_planned": total_boards},
        )
    try:
        for task in asyncio.as_completed(tasks):
            index, board, warnings = await task
            boards_by_index[index] = board
            validation_errors.extend(warnings)
            completed += 1

            narration_segments = {}
            for seg_idx, (seg_key, seg_info) in enumerate(segments_outline.items()):
                start, end = seg_info["board_range"]
                segment_boards = [b for b in boards_by_index[start - 1:end] if b]
                if not segment_boards:
                    continue
                refs_chars = []
                scene_location = ""
                for item in segment_boards:
                    refs = item.get("asset_refs", {})
                    if not scene_location and refs.get("scene"):
                        scene_location = refs.get("scene")
                    for name in refs.get("characters", []):
                        if name not in refs_chars:
                            refs_chars.append(name)
                narration_segments[seg_key] = {
                    "episode": seg_info.get("episode", 1),
                    "segment_index": seg_info.get("segment_index", seg_idx + 1),
                    "characters_in_segment": refs_chars,
                    "scene_location": scene_location,
                    "boards": segment_boards,
                }
            data["narration_segments"] = narration_segments
            _write_decompose_progress(
                project_dir,
                data,
                status="running",
                stage="boards",
                message=f"故事板生成中：{completed}/{total_boards}",
                current=completed,
                total=total_boards,
                extra={"segments_planned": len(segments_outline), "boards_planned": total_boards},
            )
    except HTTPException as e:
        for task in tasks:
            task.cancel()
        _write_decompose_failure(project_dir, data, "boards", e)
        raise

    narration_segments = {}
    for seg_idx, (seg_key, seg_info) in enumerate(segments_outline.items()):
        start, end = seg_info["board_range"]
        segment_boards = [b for b in boards_by_index[start - 1:end] if b]
        refs_chars = []
        scene_location = ""
        for item in segment_boards:
            refs = item.get("asset_refs", {})
            if not scene_location and refs.get("scene"):
                scene_location = refs.get("scene")
            for name in refs.get("characters", []):
                if name not in refs_chars:
                    refs_chars.append(name)
        narration_segments[seg_key] = {
            "episode": seg_info.get("episode", 1),
            "segment_index": seg_info.get("segment_index", seg_idx + 1),
            "characters_in_segment": refs_chars,
            "scene_location": scene_location,
            "boards": segment_boards,
        }
    data["narration_segments"] = narration_segments
    sync_board_metadata(data)
    apply_default_visual_macros(data)
    pl.write_pipeline(project_dir, data)

    # ── Assemble final result ──
    result = {
        "project": data["project"],
        "narration_style": style,
        "source_text": data["source_text"],
        "script_plan": script_plan,
        "assets": assets,
        "_decomposition_outline": {
            "segments": segments_outline,
            "script_plan_checksum": script_plan.get("source_checksum"),
            "planner_version": script_plan.get("planner_version"),
            "mode": "planned_boards",
        },
        "narration_segments": narration_segments,
    }
    ensure_referenced_assets(result)
    sync_board_metadata(result)
    apply_default_visual_macros(result)
    _write_decompose_progress(
        project_dir,
        result,
        status="completed",
        stage="done",
        message=f"拆解完成：{len(narration_segments)} 段，{_completed_board_total(narration_segments)} 个 board",
        current=_completed_board_total(narration_segments),
        total=total_boards,
        extra={"segments_planned": len(segments_outline), "boards_planned": total_boards},
    )

    # Assemble prompts after all boards exist so each board can reference the previous final panel.
    all_boards = [
        board
        for seg in result.get("narration_segments", {}).values()
        for board in seg.get("boards", [])
    ]
    _refresh_storyboard_prompt_context(
        sorted(all_boards, key=lambda item: int(item.get("page") or 0)),
        assets,
    )

    pl.write_pipeline(project_dir, result)

    return {
        "ok": True,
        "validation_errors": validation_errors,
        "stats": {
            "segments": len(narration_segments),
            "boards": sum(len(s.get("boards", [])) for s in narration_segments.values()),
            "script_boards": len(script_plan.get("board_plan", [])),
            "script_estimated_seconds": script_plan.get("stats", {}).get("estimated_seconds", 0),
            "characters": len(assets["characters"]),
            "scenes": len(assets["scenes"]),
            "props": len(assets["props"]),
        },
    }


@router.get("/status")
async def project_status(project_name: str):
    """Read pipeline status for a project."""
    project_dir = pl.get_project_root(project_name)
    if not (project_dir / "pipeline.json").exists():
        raise HTTPException(status_code=404, detail=f"项目不存在: {project_name}")
    data = pl.read_pipeline(project_dir)
    ensure_referenced_assets(data)
    ensure_audio_refs(data, project_dir)
    ensure_bgm_refs(data, project_dir)
    sync_board_metadata(data)
    apply_default_visual_macros(data)
    normalize_video_output_paths(project_dir, data)
    data["project_dir"] = str(project_dir)
    pl.write_pipeline(project_dir, data)
    return data


@router.get("/episodes")
async def project_episodes(project_name: str):
    project_dir = _check_project(project_name)
    data = pl.read_pipeline(project_dir)
    ensure_referenced_assets(data)
    ensure_audio_refs(data, project_dir)
    ensure_bgm_refs(data, project_dir)
    sync_board_metadata(data)
    apply_default_visual_macros(data)
    pl.write_pipeline(project_dir, data)
    return {
        "project": data.get("project") or project_name,
        "audio_refs": data.get("audio_refs", {}),
        "bgm_refs": data.get("bgm_refs", {}),
        "episodes": _episode_groups(data),
    }


@router.put("/audio-ref-selection")
async def select_audio_ref(req: SelectAudioRefRequest):
    project_dir = _check_project(req.project_name)
    data = pl.read_pipeline(project_dir)
    ensure_audio_refs(data, project_dir)
    selected = (req.selected or "").strip()
    if selected and selected not in data.get("audio_refs", {}).get("options", {}):
        raise HTTPException(status_code=404, detail=f"音色不存在: {selected}")
    data.setdefault("audio_refs", {})["selected"] = selected
    pl.write_pipeline(project_dir, data)
    return {"ok": True, "audio_refs": data.get("audio_refs", {})}


@router.put("/bgm-ref-selection")
async def select_bgm_ref(req: SelectBgmRefRequest):
    project_dir = _check_project(req.project_name)
    data = pl.read_pipeline(project_dir)
    ensure_bgm_refs(data, project_dir)
    selected = (req.selected or "").strip()
    if selected and selected not in data.get("bgm_refs", {}).get("options", {}):
        raise HTTPException(status_code=404, detail=f"BGM 不存在: {selected}")
    data.setdefault("bgm_refs", {})["selected"] = selected
    pl.write_pipeline(project_dir, data)
    return {"ok": True, "bgm_refs": data.get("bgm_refs", {})}


@router.post("/episodes/concat")
async def concat_episode(req: ConcatEpisodeRequest):
    project_dir = _check_project(req.project_name)
    data = pl.read_pipeline(project_dir)
    ensure_bgm_refs(data, project_dir)
    episodes = {item["episode"]: item for item in _episode_groups(data)}
    episode = episodes.get(req.episode)
    if not episode:
        raise HTTPException(status_code=404, detail=f"集不存在: {req.episode}")
    if not episode["can_concat"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error_type": "episode_not_ready",
                "message": "本集还有视频片段缺失，不能合成",
                "missing_parts": episode["missing_parts"],
            },
        )

    inputs = [Path(part["local_path"]) for part in episode["parts"]]
    missing_files = [str(path) for path in inputs if not path.exists()]
    if missing_files:
        raise HTTPException(
            status_code=400,
            detail={
                "error_type": "missing_video_files",
                "message": "本地视频文件不存在，不能合成",
                "missing_files": missing_files,
            },
        )

    output = project_dir / "episodes" / f"第{req.episode}集.mp4"
    bgm_ref = selected_bgm_ref(data)
    bgm_path = Path(bgm_ref["path"]) if bgm_ref else None
    if bgm_path and not bgm_path.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "error_type": "missing_bgm_file",
                "message": "已选择的 BGM 文件不存在，不能合成",
                "missing_bgm_file": str(bgm_path),
            },
        )
    try:
        subtitle_events = _episode_subtitle_events(data, episode["parts"])
        if bgm_path:
            await asyncio.to_thread(_concat_video_files, inputs, output, subtitle_events, bgm_path)
        else:
            await asyncio.to_thread(_concat_video_files, inputs, output, subtitle_events)
    except Exception as e:
        logger.error("合成本集失败: %s", e)
        raise HTTPException(status_code=500, detail=f"合成本集失败: {e}")

    export = {
        "status": "completed",
        "local_path": str(output),
        "duration": episode["seconds"],
        "subtitle_style": "large_white_black_outline",
        "parts": [{"segment_key": p["segment_key"], "board_index": p["board_index"]} for p in episode["parts"]],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if bgm_ref:
        export["bgm"] = {"name": bgm_ref["name"], "local_path": bgm_ref.get("local_path"), "volume": bgm_ref.get("volume", 0.18)}
    if req.copy_to_desktop:
        desktop_path = Path.home() / "Desktop" / f"{data.get('project') or req.project_name}_第{req.episode}集.mp4"
        shutil.copyfile(output, desktop_path)
        export["desktop_path"] = str(desktop_path)

    data.setdefault("episode_exports", {})[str(req.episode)] = export
    pl.write_pipeline(project_dir, data)
    return {"ok": True, "export": export, "episodes": _episode_groups(data)}
