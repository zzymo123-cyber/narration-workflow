import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import pipeline as pl
from api.decomposition import (
    build_step1_prompt, build_step2_prompt, build_step3_prompt,
    parse_decomposition_response,
)
from api.prompts import assemble_storyboard_prompt, assemble_video_prompt
from api.validation import validate_board_page
from api.llm import generate_prompt_async, get_llm_config, _resolve_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateProjectRequest(BaseModel):
    project_name: str
    source_text: str
    narration_style: str = "third_person"


class DecomposeRequest(BaseModel):
    project_name: str


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


async def _call_llm(api_key: str, system_prompt: str, user_message: str) -> dict:
    """Call LLM and return parsed JSON dict. Raises HTTPException on any failure."""
    # Call LLM
    try:
        raw_response = await generate_prompt_async(api_key, system_prompt, user_message)
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
        if extracted[-1] != "}":
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

@router.post("/decompose")
async def decompose_project(req: DecomposeRequest):
    """Run multi-step LLM decomposition on the project's source text."""
    project_dir = _check_project(req.project_name)
    data = pl.read_pipeline(project_dir)
    source_text = data.get("source_text", "")
    if not source_text:
        raise HTTPException(status_code=400, detail="项目没有 source_text，无法拆解")

    api_key = _require_api_key()
    style = data.get("narration_style", "third_person")

    # ── Step 1: Extract assets ──
    logger.info("Step 1: Extracting assets...")
    step1_prompt = build_step1_prompt(style)
    assets_result = await _call_llm(api_key, step1_prompt, source_text)

    assets = {
        "characters": assets_result.get("characters", {}),
        "scenes": assets_result.get("scenes", {}),
        "props": assets_result.get("props", {}),
    }
    logger.info(f"Step 1 done: {len(assets['characters'])} chars, {len(assets['scenes'])} scenes, {len(assets['props'])} props")

    # Save intermediate result
    data["assets"] = assets
    pl.write_pipeline(project_dir, data)

    # ── Step 2: Generate segment outline ──
    logger.info("Step 2: Generating segment outline...")
    step2_prompt = build_step2_prompt(style, assets)
    outline_result = await _call_llm(api_key, step2_prompt, source_text)

    segments_outline = outline_result.get("segments", {})
    logger.info(f"Step 2 done: {len(segments_outline)} segments planned")

    # ── Step 3: Generate boards for each segment ──
    narration_segments = {}
    validation_errors = []
    seg_keys = sorted(segments_outline.keys())

    for seg_idx, seg_key in enumerate(seg_keys):
        seg_info = segments_outline[seg_key]
        scene_location = seg_info.get("scene_location", "")
        characters_in_segment = seg_info.get("characters_in_segment", [])
        num_boards = seg_info.get("num_boards", 2)

        logger.info(f"Step 3: Generating boards for {seg_key} ({num_boards} boards, {seg_idx+1}/{len(seg_keys)})...")

        step3_prompt = build_step3_prompt(
            style=style,
            seg_key=seg_key,
            scene_location=scene_location,
            characters=characters_in_segment,
            num_boards=num_boards,
            assets=assets,
        )
        boards_result = await _call_llm(api_key, step3_prompt, source_text)

        boards = boards_result.get("boards", [])

        # Build segment
        narration_segments[seg_key] = {
            "episode": seg_info.get("episode", 1),
            "segment_index": seg_info.get("segment_index", seg_idx + 1),
            "characters_in_segment": characters_in_segment,
            "scene_location": scene_location,
            "boards": boards,
        }

        # Validate boards
        for i, board in enumerate(boards):
            errors = validate_board_page(board)
            if errors:
                validation_errors.append(f"{seg_key}/board[{i}]: {errors}")

    # ── Assemble final result ──
    result = {
        "project": data["project"],
        "narration_style": style,
        "source_text": data["source_text"],
        "assets": assets,
        "narration_segments": narration_segments,
    }

    # Assemble prompts for each board
    for seg_key, seg in result.get("narration_segments", {}).items():
        for board in seg.get("boards", []):
            board["storyboard_image"]["prompt"] = assemble_storyboard_prompt(board)
            board["video"]["prompt"] = assemble_video_prompt(board)

    pl.write_pipeline(project_dir, result)

    return {
        "ok": True,
        "validation_errors": validation_errors,
        "stats": {
            "segments": len(narration_segments),
            "boards": sum(len(s.get("boards", [])) for s in narration_segments.values()),
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
    return data
