import asyncio
import logging
from pathlib import Path

from api import pipeline as pl
from api.generation import ASSET_TYPES, asset_output_path, ensure_referenced_assets, normalize_video_output_paths, storyboard_output_path, sync_board_metadata, video_output_path
from api.routes.settings import get_api_key

logger = logging.getLogger(__name__)

_poller_task: asyncio.Task | None = None
POLL_INTERVAL = 15  # seconds


def _scan_projects() -> list[Path]:
    """Scan narration_studio root for projects with pipeline.json."""
    root = Path.home() / "Desktop" / "narration_studio"
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir() and (p / "pipeline.json").exists()]


def _collect_pending_tasks(data: dict) -> list[dict]:
    """Walk all board_pages and collect pending storyboard/video tasks."""
    pending = []
    for asset_type in ASSET_TYPES:
        for asset_name, asset in data.get("assets", {}).get(asset_type, {}).items():
            if asset.get("status") == "submitted" and asset.get("task_id"):
                pending.append({
                    "type": "asset",
                    "asset_type": asset_type,
                    "asset_name": asset_name,
                    "task_id": asset["task_id"],
                })
    for seg_key, seg in data.get("narration_segments", {}).items():
        for i, board in enumerate(seg.get("boards", [])):
            sb = board.get("storyboard_image", {})
            if sb.get("status") == "submitted" and sb.get("task_id"):
                pending.append({
                    "type": "storyboard",
                    "segment_key": seg_key,
                    "board_index": i,
                    "task_id": sb["task_id"],
                })
            video = board.get("video", {})
            if video.get("status") == "submitted" and video.get("task_id"):
                pending.append({
                    "type": "video",
                    "segment_key": seg_key,
                    "board_index": i,
                    "task_id": video["task_id"],
                })
    return pending


def _get_latest_task_target(data: dict, task: dict) -> dict | None:
    if task["type"] == "asset":
        return data.get("assets", {}).get(task["asset_type"], {}).get(task["asset_name"])
    if task["type"] == "storyboard":
        seg = data.get("narration_segments", {}).get(task["segment_key"], {})
        boards = seg.get("boards", [])
        if task["board_index"] < len(boards):
            return boards[task["board_index"]].get("storyboard_image")
    if task["type"] == "video":
        seg = data.get("narration_segments", {}).get(task["segment_key"], {})
        boards = seg.get("boards", [])
        if task["board_index"] < len(boards):
            return boards[task["board_index"]].get("video")
    return None


def _write_task_result(project_dir: Path, task: dict, result: dict) -> None:
    data = pl.read_pipeline(project_dir)
    target = _get_latest_task_target(data, task)
    if not isinstance(target, dict):
        return
    if target.get("status") != "submitted" or target.get("task_id") != task["task_id"]:
        return

    target["status"] = result["status"]
    if result.get("url"):
        target["url"] = result["url"]
    if result.get("local_path"):
        target["local_path"] = result["local_path"]
    if result.get("error"):
        target["error"] = result["error"]
    pl.write_pipeline(project_dir, data)


def _merge_synced_existing_outputs(project_dir: Path, stale_data: dict) -> bool:
    latest = pl.read_pipeline(project_dir)
    changed = False

    for asset_type in ASSET_TYPES:
        for asset_name, stale_asset in stale_data.get("assets", {}).get(asset_type, {}).items():
            if stale_asset.get("status") != "completed" or not stale_asset.get("local_path"):
                continue
            latest_asset = latest.get("assets", {}).get(asset_type, {}).get(asset_name)
            if not isinstance(latest_asset, dict) or latest_asset.get("status") == "submitted":
                continue
            if latest_asset.get("local_path") != stale_asset.get("local_path") or latest_asset.get("status") != "completed":
                latest_asset["status"] = "completed"
                latest_asset["local_path"] = stale_asset["local_path"]
                changed = True

    for seg_key, stale_seg in stale_data.get("narration_segments", {}).items():
        latest_seg = latest.get("narration_segments", {}).get(seg_key)
        if not isinstance(latest_seg, dict):
            continue
        latest_boards = latest_seg.get("boards", [])
        for index, stale_board in enumerate(stale_seg.get("boards", [])):
            if index >= len(latest_boards):
                continue
            latest_board = latest_boards[index]
            stale_storyboard = stale_board.get("storyboard_image", {})
            latest_storyboard = latest_board.get("storyboard_image")
            if (
                isinstance(latest_storyboard, dict)
                and stale_storyboard.get("status") == "completed"
                and stale_storyboard.get("local_path")
                and latest_storyboard.get("status") != "submitted"
                and (
                    latest_storyboard.get("local_path") != stale_storyboard.get("local_path")
                    or latest_storyboard.get("status") != "completed"
                )
            ):
                latest_storyboard["status"] = "completed"
                latest_storyboard["local_path"] = stale_storyboard["local_path"]
                changed = True

            stale_video = stale_board.get("video", {})
            latest_video = latest_board.get("video")
            if (
                isinstance(latest_video, dict)
                and stale_video.get("status") == "completed"
                and stale_video.get("local_path")
                and latest_video.get("status") != "submitted"
                and (
                    latest_video.get("local_path") != stale_video.get("local_path")
                    or latest_video.get("status") != "completed"
                )
            ):
                latest_video["status"] = "completed"
                latest_video["local_path"] = stale_video["local_path"]
                if stale_video.get("output_path"):
                    latest_video["output_path"] = stale_video["output_path"]
                changed = True

    if changed:
        pl.write_pipeline(project_dir, latest)
    return changed


async def _poll_once():
    """Single poll cycle across all projects."""
    vidu_api_key = get_api_key("VIDU_API_KEY")
    wetoken_api_key = get_api_key("WETOKEN_API_KEY")

    for project_dir in _scan_projects():
        try:
            data = pl.read_pipeline(project_dir)
            ensure_referenced_assets(data)
            sync_board_metadata(data)
        except Exception:
            continue

        changed = _sync_existing_outputs(project_dir, data)
        if changed:
            _merge_synced_existing_outputs(project_dir, data)
        for task in _collect_pending_tasks(data):
            try:
                if task["type"] == "asset" and vidu_api_key:
                    result = await _poll_vidu_image(
                        vidu_api_key,
                        task["task_id"],
                        asset_output_path(project_dir, task["asset_type"], task["asset_name"]),
                    )
                    if result:
                        _write_task_result(project_dir, task, result)
                elif task["type"] == "storyboard" and vidu_api_key:
                    result = await _poll_storyboard(
                        vidu_api_key,
                        task["task_id"],
                        storyboard_output_path(project_dir, task["segment_key"], task["board_index"]),
                    )
                    if result:
                        _write_task_result(project_dir, task, result)
                elif task["type"] == "video" and wetoken_api_key:
                    seg = data["narration_segments"][task["segment_key"]]
                    board = seg["boards"][task["board_index"]]
                    output_path = board.get("video", {}).get("output_path")
                    result = await _poll_video(
                        wetoken_api_key,
                        task["task_id"],
                        project_dir,
                        Path(output_path) if output_path else video_output_path(project_dir, board, task["segment_key"], task["board_index"]),
                    )
                    if result:
                        _write_task_result(project_dir, task, result)
            except Exception as e:
                logger.warning(f"Poll {task['type']} task {task['task_id']} failed: {e}")


def _sync_existing_outputs(project_dir: Path, data: dict) -> bool:
    changed = False
    for asset_type in ASSET_TYPES:
        for asset_name, asset in data.get("assets", {}).get(asset_type, {}).items():
            path = asset_output_path(project_dir, asset_type, asset_name)
            if asset.get("status") == "submitted":
                continue
            if path.exists() and asset.get("local_path") != str(path):
                asset["status"] = "completed"
                asset["local_path"] = str(path)
                changed = True
    for seg_key, seg in data.get("narration_segments", {}).items():
        for i, board in enumerate(seg.get("boards", [])):
            path = storyboard_output_path(project_dir, seg_key, i)
            sb = board.get("storyboard_image", {})
            if sb.get("status") == "submitted":
                continue
            if path.exists() and sb.get("local_path") != str(path):
                sb["status"] = "completed"
                sb["local_path"] = str(path)
                changed = True
    if normalize_video_output_paths(project_dir, data):
        changed = True
    return changed


async def _poll_storyboard(api_key: str, task_id: str, dest_path: Path | None = None) -> dict | None:
    return await _poll_vidu_image(api_key, task_id, dest_path)


async def _poll_vidu_image(api_key: str, task_id: str, dest_path: Path | None = None) -> dict | None:
    """Poll Vidu storyboard image task. Returns result dict or None if still pending."""
    from api.vidu import download_image_async, poll_task
    try:
        result = await asyncio.to_thread(poll_task, api_key, task_id)
    except Exception as e:
        logger.warning(f"Vidu poll error: {e}")
        return None

    if result["status"] in ("success", "completed"):
        local_path = None
        image_url = result.get("image_url")
        if image_url and dest_path:
            try:
                await download_image_async(image_url, dest_path)
                local_path = str(dest_path)
            except Exception as e:
                logger.error(f"Storyboard download failed: {e}")
        return {"status": "completed", "url": image_url, "local_path": local_path}
    elif result["status"] == "failed":
        return {"status": "failed", "url": None, "local_path": None}
    return None


async def _poll_video(api_key: str, task_id: str, project_dir: Path, dest_path: Path | None = None) -> dict | None:
    """Poll Wetoken video task. Returns result dict or None if still pending."""
    from api.wetoken import poll_task_async, download_video_async
    try:
        result = await poll_task_async(api_key, task_id)
    except Exception as e:
        logger.warning(f"Wetoken poll error: {e}")
        return None

    if result["status"] == "completed" and result.get("video_url"):
        # Download video to project dir
        dest = dest_path or project_dir / "videos" / f"{task_id}.mp4"
        try:
            await download_video_async(result["video_url"], dest)
            return {"status": "completed", "url": result["video_url"], "local_path": str(dest)}
        except Exception as e:
            logger.error(f"Video download failed: {e}")
            return {"status": "completed", "url": result["video_url"], "local_path": None}
    elif result["status"] == "failed":
        return {"status": "failed", "url": None, "local_path": None, "error": result.get("error")}
    return None


async def _poller_loop():
    while True:
        try:
            await _poll_once()
        except Exception as e:
            logger.error(f"Poller loop error: {e}")
        await asyncio.sleep(POLL_INTERVAL)


async def start():
    global _poller_task
    _poller_task = asyncio.create_task(_poller_loop())
    logger.info("Poller started")


async def stop():
    global _poller_task
    if _poller_task:
        _poller_task.cancel()
        try:
            await _poller_task
        except asyncio.CancelledError:
            pass
        _poller_task = None

    from api.wetoken import close_async_client
    await close_async_client()
    logger.info("Poller stopped")
