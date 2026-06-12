import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from api import pipeline as pl
from api.prompts import assemble_storyboard_prompt, assemble_video_prompt
from api.routes.settings import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


class SubmitStoryboardRequest(BaseModel):
    project_name: str
    segment_key: str
    board_index: int


class SubmitVideoRequest(BaseModel):
    project_name: str
    segment_key: str
    board_index: int


@router.post("/submit-storyboard")
async def submit_storyboard(req: SubmitStoryboardRequest):
    """Submit storyboard image generation to Vidu."""
    project_dir = pl.get_project_root(req.project_name)
    if not (project_dir / "pipeline.json").exists():
        raise HTTPException(status_code=404, detail=f"项目不存在: {req.project_name}")

    data = pl.read_pipeline(project_dir)
    segment = data.get("narration_segments", {}).get(req.segment_key)
    if not segment:
        raise HTTPException(status_code=404, detail=f"段不存在: {req.segment_key}")

    boards = segment.get("boards", [])
    if req.board_index >= len(boards):
        raise HTTPException(status_code=400, detail=f"board_index {req.board_index} 超出范围")

    board = boards[req.board_index]
    sb = board.get("storyboard_image", {})
    if sb.get("status") == "submitted":
        return {"ok": True, "task_id": sb.get("task_id"), "message": "已提交"}

    prompt = sb.get("prompt") or assemble_storyboard_prompt(board)
    if not prompt:
        raise HTTPException(status_code=400, detail="分镜板提示词为空")

    vidu_api_key = get_api_key("VIDU_API_KEY")
    if not vidu_api_key:
        raise HTTPException(status_code=500, detail="缺少 VIDU_API_KEY")

    try:
        from api.vidu import submit_reference2image
        task_id = submit_reference2image(
            api_key=vidu_api_key,
            prompt=prompt,
            ratio="9:16",
        )
    except Exception as e:
        logger.error(f"Vidu 提交失败: {e}")
        raise HTTPException(status_code=500, detail=f"Vidu 提交失败: {str(e)}")

    board["storyboard_image"]["status"] = "submitted"
    board["storyboard_image"]["task_id"] = task_id
    board["storyboard_image"]["prompt"] = prompt
    pl.write_pipeline(project_dir, data)

    return {"ok": True, "task_id": task_id}


@router.post("/submit-video")
async def submit_video(req: SubmitVideoRequest):
    """Submit video generation to Wetoken/Seedance."""
    project_dir = pl.get_project_root(req.project_name)
    if not (project_dir / "pipeline.json").exists():
        raise HTTPException(status_code=404, detail=f"项目不存在: {req.project_name}")

    data = pl.read_pipeline(project_dir)
    segment = data.get("narration_segments", {}).get(req.segment_key)
    if not segment:
        raise HTTPException(status_code=404, detail=f"段不存在: {req.segment_key}")

    boards = segment.get("boards", [])
    if req.board_index >= len(boards):
        raise HTTPException(status_code=400, detail=f"board_index {req.board_index} 超出范围")

    board = boards[req.board_index]
    video = board.get("video", {})
    if video.get("status") == "submitted":
        return {"ok": True, "task_id": video.get("task_id"), "message": "已提交"}

    # Require storyboard image to be completed before video
    sb = board.get("storyboard_image", {})
    if sb.get("status") != "completed":
        raise HTTPException(status_code=400, detail="分镜板图片未完成，请先生成分镜板")

    prompt = video.get("prompt") or assemble_video_prompt(board)
    if not prompt:
        raise HTTPException(status_code=400, detail="视频提示词为空")

    wetoken_api_key = get_api_key("WETOKEN_API_KEY")
    if not wetoken_api_key:
        raise HTTPException(status_code=500, detail="缺少 WETOKEN_API_KEY")

    # Collect reference images
    image_paths = []
    # Storyboard image
    if sb.get("local_path"):
        image_paths.append(sb["local_path"])
    # Character sheets
    asset_refs = board.get("asset_refs", {})
    for char_name in asset_refs.get("characters", []):
        char_data = data.get("assets", {}).get("characters", {}).get(char_name, {})
        if char_data.get("local_path"):
            image_paths.append(char_data["local_path"])

    duration = board.get("board_duration", 10)

    try:
        from api.wetoken import submit_video_task
        task_id = submit_video_task(
            api_key=wetoken_api_key,
            prompt=prompt,
            image_paths=image_paths,
            duration=duration,
            ratio="9:16",
            generate_audio=True,
            project_dir=project_dir,
        )
    except Exception as e:
        logger.error(f"Seedance 提交失败: {e}")
        raise HTTPException(status_code=500, detail=f"Seedance 提交失败: {str(e)}")

    board["video"]["status"] = "submitted"
    board["video"]["task_id"] = task_id
    board["video"]["prompt"] = prompt
    pl.write_pipeline(project_dir, data)

    return {"ok": True, "task_id": task_id}
