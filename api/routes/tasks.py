import logging
import asyncio
import os
import re
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from api import pipeline as pl
from api.color_palettes import color_palette_video_note, ensure_color_palette_reference_image
from api.prompts import assemble_storyboard_prompt, assemble_video_prompt
from api.routes.settings import get_api_key
from api.validation import validate_board_page
from api.generation import (
    ASSET_TYPES,
    append_generation_history,
    apply_default_visual_macros,
    asset_output_path,
    asset_prompt,
    collect_asset_reference_items,
    collect_video_asset_urls,
    collect_video_reference_images,
    ensure_audio_refs,
    ensure_referenced_assets,
    missing_asset_references,
    missing_character_references,
    normalize_assets,
    selected_audio_ref,
    sync_board_metadata,
    video_output_path,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class SubmitStoryboardRequest(BaseModel):
    project_name: str
    segment_key: str
    board_index: int
    prompt: Optional[str] = None
    force: bool = False


class SubmitVideoRequest(BaseModel):
    project_name: str
    segment_key: str
    board_index: int
    prompt: Optional[str] = None
    force: bool = False


class SubmitAssetRequest(BaseModel):
    project_name: str
    asset_type: str
    asset_name: str
    prompt: Optional[str] = None
    force: bool = False


class ProjectRequest(BaseModel):
    project_name: str


def _get_project_data(project_name: str) -> tuple[Path, dict]:
    project_dir = pl.get_project_root(project_name)
    if not (project_dir / "pipeline.json").exists():
        raise HTTPException(status_code=404, detail=f"项目不存在: {project_name}")
    data = pl.read_pipeline(project_dir)
    ensure_referenced_assets(data)
    ensure_audio_refs(data, project_dir)
    sync_board_metadata(data)
    apply_default_visual_macros(data)
    pl.write_pipeline(project_dir, data)
    return project_dir, data


def _get_board(data: dict, segment_key: str, board_index: int) -> dict:
    segment = data.get("narration_segments", {}).get(segment_key)
    if not segment:
        raise HTTPException(status_code=404, detail=f"段不存在: {segment_key}")
    boards = segment.get("boards", [])
    if board_index >= len(boards):
        raise HTTPException(status_code=400, detail=f"board_index {board_index} 超出范围")
    return boards[board_index]


def _iter_boards(data: dict):
    for seg_key, seg in data.get("narration_segments", {}).items():
        for idx, board in enumerate(seg.get("boards", [])):
            yield seg_key, idx, board


def _submit_concurrency(env_name: str, default: int, limit: int) -> int:
    try:
        return max(1, min(limit, int(os.environ.get(env_name, str(default)))))
    except ValueError:
        return default


def _storyboard_prompt_with_reference_notes(prompt: str, reference_items: list[dict]) -> str:
    if not reference_items:
        return prompt
    lines = ["参考图顺序说明："]
    for index, item in enumerate(reference_items, start=1):
        lines.append(f"参考图{index}：{item.get('note', '')}")
    lines.append("生成时必须按上述身份使用参考图，不要把角色、场景、道具互相混用。")
    return prompt + "\n\n" + "\n".join(lines)


def _storyboard_source_excerpt(board: dict) -> str:
    review_excerpt = (board.get("review") or {}).get("source_excerpt") or ""
    trace_excerpt = "".join(item.get("text", "") for item in board.get("source_trace", []) or [])
    return (review_excerpt or trace_excerpt).strip()


def _custom_video_prompt(board: dict, requested_prompt: str | None) -> str | None:
    prompt = (requested_prompt or "").strip()
    if not prompt:
        return None
    saved_prompt = ((board.get("video") or {}).get("prompt") or "").strip()
    if saved_prompt and prompt == saved_prompt:
        return None
    return prompt


def _video_prompt_with_palette_note(prompt: str, board: dict) -> str:
    palette_note = color_palette_video_note(board)
    if not palette_note or palette_note in prompt:
        return prompt
    return f"{prompt}\n\n色板参考：\n{palette_note}"


def _mark_video_stale_after_storyboard_regen(board: dict) -> None:
    video = board.setdefault("video", {})
    if video.get("status") != "completed":
        return
    video["previous_task_id"] = video.get("task_id")
    video["previous_url"] = video.get("url")
    video["previous_local_path"] = video.get("local_path")
    video["previous_prompt"] = video.get("prompt")
    video["status"] = "needed"
    video["task_id"] = None
    video["url"] = None
    video["local_path"] = None
    video["stale_reason"] = "故事板已重新生成，需要重新生成视频"
    append_generation_history(
        video,
        "mark_stale",
        reason=video["stale_reason"],
        previous_task_id=video.get("previous_task_id"),
        previous_local_path=video.get("previous_local_path"),
    )


def _error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)
        return str(detail)
    return str(exc)


def _elapsed_ms(start: float) -> int:
    return max(0, int(round((time.perf_counter() - start) * 1000)))


def _validate_storyboard_prompt_for_submit(board: dict, prompt: str, submitted_prompt: str) -> None:
    combined = f"{prompt}\n{submitted_prompt}"
    errors = []
    blocked_markers = [
        "角色在关键情节中反应",
        "围绕原文事件行动",
        "画面呈现原文事件",
        "郑教授，黑色巨蟒",
        "接下来会发生什么",
    ]
    for marker in blocked_markers:
        if marker in combined:
            errors.append(f"提示词包含禁止内容：{marker}")
    for beat in board.get("voice_timeline", []) or []:
        text = (beat.get("text") or "").strip()
        if text.startswith(("，", ",", "、", "；", ";")):
            errors.append(f"声音切片以承接标点开头：{text[:24]}")
    if board.get("voice_timeline") and not _storyboard_source_excerpt(board):
        errors.append("当前板缺少原文依据 source_excerpt/source_trace")
    if re.search(r"现在发生什么：\s*\n\s*(通过镜头|展现)", prompt):
        errors.append("现在发生什么仍是概括性镜头目标，不是原文事件")
    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error_type": "invalid_storyboard_prompt",
                "message": "故事板提示词未通过提交前校验",
                "errors": errors,
            },
        )


@router.post("/submit-asset")
async def submit_asset(req: SubmitAssetRequest):
    """Submit character, scene, or prop reference image generation to Vidu."""
    if req.asset_type not in ASSET_TYPES:
        raise HTTPException(status_code=400, detail=f"asset_type 不支持: {req.asset_type}")
    project_dir, data = _get_project_data(req.project_name)
    assets = data.get("assets", {}).get(req.asset_type, {})
    asset = assets.get(req.asset_name)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"资产不存在: {req.asset_type}/{req.asset_name}")

    if asset.get("status") == "submitted" and not req.force:
        return {"ok": True, "task_id": asset.get("task_id"), "message": "已提交"}
    if asset.get("status") == "completed" and not req.force:
        return {"ok": True, "task_id": asset.get("task_id"), "message": "已完成"}

    vidu_api_key = get_api_key("VIDU_API_KEY")
    if not vidu_api_key:
        raise HTTPException(status_code=500, detail="缺少 VIDU_API_KEY")

    prompt = (req.prompt or "").strip() or asset.get("prompt") or asset_prompt(req.asset_type, req.asset_name, asset)
    try:
        from api.vidu import submit_image_task
        result = await asyncio.to_thread(
            submit_image_task,
            api_key=vidu_api_key,
            prompt=prompt,
            image_paths=[],
            ratio="3:4" if req.asset_type == "characters" else "16:9",
        )
    except Exception as e:
        logger.error(f"Vidu 资产提交失败: {e}")
        raise HTTPException(status_code=500, detail=f"Vidu 资产提交失败: {str(e)}")

    asset["status"] = "submitted"
    asset["task_id"] = result["task_id"]
    asset["prompt"] = prompt
    asset["error"] = None
    data = pl.read_pipeline(project_dir)
    asset = data.get("assets", {}).get(req.asset_type, {}).get(req.asset_name)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"资产不存在: {req.asset_type}/{req.asset_name}")
    asset["status"] = "submitted"
    asset["task_id"] = result["task_id"]
    asset["prompt"] = prompt
    asset["error"] = None
    if req.force:
        asset["previous_url"] = asset.get("url")
        asset["previous_local_path"] = asset.get("local_path")
        asset["url"] = None
        asset["local_path"] = None
    asset["output_path"] = str(asset_output_path(project_dir, req.asset_type, req.asset_name))
    append_generation_history(
        asset,
        "submit_asset",
        task_id=result["task_id"],
        prompt=prompt,
        force=req.force,
        output_path=asset.get("output_path"),
    )
    pl.write_pipeline(project_dir, data)
    return {"ok": True, "task_id": result["task_id"]}


@router.post("/submit-assets")
async def submit_assets(req: ProjectRequest):
    """Submit all needed asset reference images."""
    project_dir, data = _get_project_data(req.project_name)
    submitted = []
    for asset_type in sorted(ASSET_TYPES):
        for name, info in data.get("assets", {}).get(asset_type, {}).items():
            if info.get("status") == "needed":
                result = await submit_asset(SubmitAssetRequest(
                    project_name=req.project_name,
                    asset_type=asset_type,
                    asset_name=name,
                ))
                submitted.append({"asset_type": asset_type, "asset_name": name, **result})
    data = pl.read_pipeline(project_dir)
    normalize_assets(data)
    return {"ok": True, "submitted": submitted, "assets": data.get("assets", {})}


@router.post("/submit-storyboard")
async def submit_storyboard(req: SubmitStoryboardRequest):
    """Submit storyboard image generation to Vidu."""
    project_dir, data = _get_project_data(req.project_name)
    board = _get_board(data, req.segment_key, req.board_index)
    sb = board.get("storyboard_image", {})
    if sb.get("status") == "submitted" and not req.force:
        return {"ok": True, "task_id": sb.get("task_id"), "message": "已提交"}
    if sb.get("status") == "completed" and not req.force:
        return {"ok": True, "task_id": sb.get("task_id"), "message": "已完成"}

    prompt = (req.prompt or "").strip() or sb.get("prompt") or assemble_storyboard_prompt(board)
    if not prompt:
        raise HTTPException(status_code=400, detail="分镜板提示词为空")

    vidu_api_key = get_api_key("VIDU_API_KEY")
    if not vidu_api_key:
        raise HTTPException(status_code=500, detail="缺少 VIDU_API_KEY")

    missing_refs = missing_asset_references(data, board)
    if missing_refs:
        raise HTTPException(
            status_code=400,
            detail={
                "error_type": "missing_reference_images",
                "message": "资产参考图未完成，请先生成资产参考图",
                "missing_asset_refs": missing_refs,
            },
        )

    try:
        from api.vidu import submit_image_task
        reference_items = collect_asset_reference_items(data, board)
        reference_images = [item["path"] for item in reference_items]
        prompt_for_generation = _storyboard_prompt_with_reference_notes(prompt, reference_items)
        _validate_storyboard_prompt_for_submit(board, prompt, prompt_for_generation)
        result = await asyncio.to_thread(
            submit_image_task,
            api_key=vidu_api_key,
            prompt=prompt_for_generation,
            image_paths=reference_images,
            ratio="9:16",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vidu 提交失败: {e}")
        raise HTTPException(status_code=500, detail=f"Vidu 提交失败: {str(e)}")

    data = pl.read_pipeline(project_dir)
    ensure_referenced_assets(data)
    sync_board_metadata(data)
    board = _get_board(data, req.segment_key, req.board_index)
    board.setdefault("storyboard_image", {})
    board["storyboard_image"]["status"] = "submitted"
    board["storyboard_image"]["task_id"] = result["task_id"]
    board["storyboard_image"]["prompt"] = prompt
    board["storyboard_image"]["submitted_prompt"] = prompt_for_generation
    board["storyboard_image"]["reference_images"] = reference_images
    board["storyboard_image"]["reference_image_labels"] = reference_items
    board["storyboard_image"]["output_path"] = str(project_dir / "storyboards" / f"{req.segment_key}_p{req.board_index + 1:02d}.jpg")
    if req.force:
        board["storyboard_image"]["previous_url"] = board["storyboard_image"].get("url")
        board["storyboard_image"]["previous_local_path"] = board["storyboard_image"].get("local_path")
        board["storyboard_image"]["url"] = None
        board["storyboard_image"]["local_path"] = None
        _mark_video_stale_after_storyboard_regen(board)
    append_generation_history(
        board["storyboard_image"],
        "submit_storyboard",
        task_id=result["task_id"],
        prompt=prompt,
        force=req.force,
        output_path=board["storyboard_image"].get("output_path"),
        reference_images=reference_images,
    )
    pl.write_pipeline(project_dir, data)

    return {"ok": True, "task_id": result["task_id"], "reference_images": reference_images}


@router.post("/submit-storyboards")
async def submit_storyboards(req: ProjectRequest):
    """Submit all needed storyboard image tasks."""
    project_dir, data = _get_project_data(req.project_name)
    vidu_api_key = get_api_key("VIDU_API_KEY")
    if not vidu_api_key:
        raise HTTPException(status_code=500, detail="缺少 VIDU_API_KEY")

    submitted = []
    skipped = []
    targets = []
    for seg_key, idx, board in _iter_boards(data):
        if board.get("storyboard_image", {}).get("status") == "needed":
            prompt = board.get("storyboard_image", {}).get("prompt") or assemble_storyboard_prompt(board)
            if not prompt:
                raise HTTPException(status_code=400, detail="分镜板提示词为空")
            missing_refs = missing_asset_references(data, board)
            if missing_refs:
                skipped.append({
                    "segment_key": seg_key,
                    "board_index": idx,
                    "reason": "资产参考图未完成",
                    "missing_asset_refs": missing_refs,
                })
                continue
            targets.append({
                "segment_key": seg_key,
                "board_index": idx,
                "board": board,
                "prompt": prompt,
                "reference_items": collect_asset_reference_items(data, board),
            })

    semaphore = asyncio.Semaphore(_submit_concurrency("STORYBOARD_SUBMIT_CONCURRENCY", 10, 10))

    async def submit_one(target: dict) -> dict:
        async with semaphore:
            start = time.perf_counter()
            try:
                from api.vidu import submit_image_task
                submitted_prompt = _storyboard_prompt_with_reference_notes(target["prompt"], target["reference_items"])
                _validate_storyboard_prompt_for_submit(target["board"], target["prompt"], submitted_prompt)
                result = await asyncio.to_thread(
                    submit_image_task,
                    api_key=vidu_api_key,
                    prompt=submitted_prompt,
                    image_paths=[item["path"] for item in target["reference_items"]],
                    ratio="9:16",
                )
            except Exception as e:
                logger.error(f"Vidu 提交失败: {e}")
                return {**target, "ok": False, "error": _error_message(e), "duration_ms": _elapsed_ms(start)}
            return {**target, "ok": True, "task_id": result["task_id"], "duration_ms": _elapsed_ms(start)}

    for result in await asyncio.gather(*(submit_one(target) for target in targets)):
        board = result["board"]
        board.setdefault("storyboard_image", {})
        if not result["ok"]:
            board["storyboard_image"]["status"] = "failed"
            board["storyboard_image"]["error"] = result["error"]
            append_generation_history(
                board["storyboard_image"],
                "submit_storyboard_failed",
                error=result["error"],
                prompt=result["prompt"],
                force=False,
                duration_ms=result["duration_ms"],
            )
            skipped.append({
                "segment_key": result["segment_key"],
                "board_index": result["board_index"],
                "reason": result["error"],
                "duration_ms": result["duration_ms"],
            })
            continue
        board["storyboard_image"]["status"] = "submitted"
        board["storyboard_image"]["task_id"] = result["task_id"]
        board["storyboard_image"]["prompt"] = result["prompt"]
        board["storyboard_image"]["submitted_prompt"] = _storyboard_prompt_with_reference_notes(result["prompt"], result["reference_items"])
        board["storyboard_image"]["reference_images"] = [item["path"] for item in result["reference_items"]]
        board["storyboard_image"]["reference_image_labels"] = result["reference_items"]
        board["storyboard_image"]["output_path"] = str(project_dir / "storyboards" / f"{result['segment_key']}_p{result['board_index'] + 1:02d}.jpg")
        append_generation_history(
            board["storyboard_image"],
            "submit_storyboard",
            task_id=result["task_id"],
            prompt=result["prompt"],
            force=False,
            output_path=board["storyboard_image"].get("output_path"),
            reference_images=[item["path"] for item in result["reference_items"]],
            duration_ms=result["duration_ms"],
        )
        submitted.append({
            "segment_key": result["segment_key"],
            "board_index": result["board_index"],
            "ok": True,
            "task_id": result["task_id"],
            "reference_images": [item["path"] for item in result["reference_items"]],
            "duration_ms": result["duration_ms"],
        })

    pl.write_pipeline(project_dir, data)
    return {"ok": True, "submitted": submitted, "skipped": skipped, "narration_segments": data.get("narration_segments", {})}


@router.post("/submit-video")
async def submit_video(req: SubmitVideoRequest):
    """Submit video generation to Wetoken/Seedance."""
    project_dir, data = _get_project_data(req.project_name)
    board = _get_board(data, req.segment_key, req.board_index)
    video = board.get("video", {})
    if video.get("status") == "submitted" and not req.force:
        return {"ok": True, "task_id": video.get("task_id"), "message": "已提交"}
    if video.get("status") == "completed" and not req.force:
        return {"ok": True, "task_id": video.get("task_id"), "message": "已完成"}

    # Require storyboard image to be completed before video
    sb = board.get("storyboard_image", {})
    if sb.get("status") != "completed":
        raise HTTPException(status_code=400, detail="分镜板图片未完成，请先生成分镜板")

    validation_errors = validate_board_page(board)
    timing_errors = [error for error in validation_errors if "voice_timeline" in error or "voice_duration" in error]
    if timing_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error_type": "invalid_board_timing",
                "message": "分镜旁白时长不足，请重新拆板后再生成视频",
                "validation_errors": timing_errors,
            },
        )

    missing_refs = missing_character_references(data, board)
    if missing_refs:
        raise HTTPException(
            status_code=400,
            detail={
                "error_type": "missing_reference_images",
                "message": "角色参考图未完成，请先生成角色参考图",
                "missing_asset_refs": missing_refs,
            },
        )

    reference_images = collect_video_reference_images(data, board)
    palette_reference_image = ensure_color_palette_reference_image(project_dir, board)
    if palette_reference_image:
        reference_images.append(palette_reference_image)
    asset_urls = collect_video_asset_urls(data, board)
    audio_ref = selected_audio_ref(data)
    reference_audios = [audio_ref["path"]] if audio_ref else []
    storyboard_ref = board.get("storyboard_image", {}).get("local_path") or board.get("storyboard_image", {}).get("url")
    custom_prompt = _custom_video_prompt(board, req.prompt)
    prompt = custom_prompt or assemble_video_prompt(
        board,
        asset_urls=asset_urls,
        storyboard_image_path=storyboard_ref,
        audio_reference_name=audio_ref["name"] if audio_ref else None,
    )
    if custom_prompt:
        prompt = _video_prompt_with_palette_note(prompt, board)
    if not prompt:
        raise HTTPException(status_code=400, detail="视频提示词为空")

    wetoken_api_key = get_api_key("WETOKEN_API_KEY")
    if not wetoken_api_key:
        raise HTTPException(status_code=500, detail="缺少 WETOKEN_API_KEY")

    duration = board.get("board_duration", 10)

    start = time.perf_counter()
    try:
        from api.wetoken import submit_video_task
        task_id = submit_video_task(
            api_key=wetoken_api_key,
            prompt=prompt,
            image_paths=reference_images,
            audio_paths=reference_audios,
            duration=duration,
            ratio="9:16",
            generate_audio=True,
            project_dir=project_dir,
        )
    except Exception as e:
        logger.error(f"Seedance 提交失败: {e}")
        raise HTTPException(status_code=500, detail=f"Seedance 提交失败: {str(e)}")
    duration_ms = _elapsed_ms(start)

    data = pl.read_pipeline(project_dir)
    sync_board_metadata(data)
    board = _get_board(data, req.segment_key, req.board_index)
    video = board.setdefault("video", {})
    video["status"] = "submitted"
    video["task_id"] = task_id
    video["prompt"] = prompt
    video["prompt_source"] = "custom" if custom_prompt else "assembled"
    video["reference_images"] = reference_images
    video["reference_audios"] = reference_audios
    video["output_path"] = str(video_output_path(project_dir, board, req.segment_key, req.board_index))
    if req.force:
        video["previous_url"] = video.get("url")
        video["previous_local_path"] = video.get("local_path")
        video["url"] = None
        video["local_path"] = None
    append_generation_history(
        video,
        "submit_video",
        task_id=task_id,
        prompt=prompt,
        prompt_source=video.get("prompt_source"),
        force=req.force,
        output_path=video.get("output_path"),
        reference_images=reference_images,
        reference_audios=reference_audios,
        duration_ms=duration_ms,
    )
    if audio_ref:
        data.setdefault("audio_refs", {}).setdefault("options", {}).setdefault(audio_ref["name"], {}).update({
            "name": audio_ref["name"],
            "local_path": audio_ref.get("local_path"),
            "asset_uri": audio_ref.get("asset_uri"),
            "status": audio_ref.get("status", "local"),
            "role": "narration",
        })
    pl.write_pipeline(project_dir, data)

    return {
        "ok": True,
        "task_id": task_id,
        "reference_images": reference_images,
        "reference_audios": reference_audios,
        "duration_ms": duration_ms,
    }


@router.post("/submit-videos")
async def submit_videos(req: ProjectRequest):
    """Submit all needed video tasks whose storyboard images are complete."""
    project_dir, data = _get_project_data(req.project_name)
    submitted = []
    skipped = []
    for seg_key, idx, board in _iter_boards(data):
        if board.get("video", {}).get("status") != "needed":
            continue
        if board.get("storyboard_image", {}).get("status") != "completed":
            skipped.append({"segment_key": seg_key, "board_index": idx, "reason": "分镜板未完成"})
            continue
        validation_errors = validate_board_page(board)
        timing_errors = [error for error in validation_errors if "voice_timeline" in error or "voice_duration" in error]
        if timing_errors:
            skipped.append({
                "segment_key": seg_key,
                "board_index": idx,
                "reason": "分镜旁白时长不足",
                "validation_errors": timing_errors,
            })
            continue
        missing_refs = missing_character_references(data, board)
        if missing_refs:
            skipped.append({
                "segment_key": seg_key,
                "board_index": idx,
                "reason": "角色参考图未完成",
                "missing_asset_refs": missing_refs,
            })
            continue
        try:
            start = time.perf_counter()
            result = await submit_video(SubmitVideoRequest(
                project_name=req.project_name,
                segment_key=seg_key,
                board_index=idx,
            ))
            submitted.append({"segment_key": seg_key, "board_index": idx, **result})
        except HTTPException as e:
            data = pl.read_pipeline(project_dir)
            sync_board_metadata(data)
            board = _get_board(data, seg_key, idx)
            video = board.setdefault("video", {})
            video["status"] = "failed"
            video["error"] = _error_message(e)
            duration_ms = _elapsed_ms(start)
            append_generation_history(
                video,
                "submit_video_failed",
                error=video["error"],
                force=False,
                duration_ms=duration_ms,
            )
            pl.write_pipeline(project_dir, data)
            skipped.append({
                "segment_key": seg_key,
                "board_index": idx,
                "reason": video["error"],
                "duration_ms": duration_ms,
            })
    data = pl.read_pipeline(project_dir)
    return {"ok": True, "submitted": submitted, "skipped": skipped, "narration_segments": data.get("narration_segments", {})}
