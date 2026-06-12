import asyncio
import logging
from pathlib import Path

from api import pipeline as pl
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


async def _poll_once():
    """Single poll cycle across all projects."""
    vidu_api_key = get_api_key("VIDU_API_KEY")
    wetoken_api_key = get_api_key("WETOKEN_API_KEY")

    for project_dir in _scan_projects():
        try:
            data = pl.read_pipeline(project_dir)
        except Exception:
            continue

        changed = False
        for task in _collect_pending_tasks(data):
            try:
                if task["type"] == "storyboard" and vidu_api_key:
                    result = await _poll_storyboard(vidu_api_key, task["task_id"])
                    if result:
                        seg = data["narration_segments"][task["segment_key"]]
                        board = seg["boards"][task["board_index"]]
                        board["storyboard_image"]["status"] = result["status"]
                        if result.get("url"):
                            board["storyboard_image"]["url"] = result["url"]
                        if result.get("local_path"):
                            board["storyboard_image"]["local_path"] = result["local_path"]
                        changed = True
                elif task["type"] == "video" and wetoken_api_key:
                    result = await _poll_video(wetoken_api_key, task["task_id"], project_dir)
                    if result:
                        seg = data["narration_segments"][task["segment_key"]]
                        board = seg["boards"][task["board_index"]]
                        board["video"]["status"] = result["status"]
                        if result.get("url"):
                            board["video"]["url"] = result["url"]
                        if result.get("local_path"):
                            board["video"]["local_path"] = result["local_path"]
                        changed = True
            except Exception as e:
                logger.warning(f"Poll {task['type']} task {task['task_id']} failed: {e}")

        if changed:
            pl.write_pipeline(project_dir, data)


async def _poll_storyboard(api_key: str, task_id: str) -> dict | None:
    """Poll Vidu storyboard image task. Returns result dict or None if still pending."""
    from api.vidu import poll_task
    try:
        result = await asyncio.to_thread(poll_task, api_key, task_id)
    except Exception as e:
        logger.warning(f"Vidu poll error: {e}")
        return None

    if result["status"] == "completed":
        return {"status": "completed", "url": result.get("image_url"), "local_path": None}
    elif result["status"] == "failed":
        return {"status": "failed", "url": None, "local_path": None}
    return None


async def _poll_video(api_key: str, task_id: str, project_dir: Path) -> dict | None:
    """Poll Wetoken video task. Returns result dict or None if still pending."""
    from api.wetoken import poll_task_async, download_video_async
    try:
        result = await poll_task_async(api_key, task_id)
    except Exception as e:
        logger.warning(f"Wetoken poll error: {e}")
        return None

    if result["status"] == "completed" and result.get("video_url"):
        # Download video to project dir
        dest = project_dir / "videos" / f"{task_id}.mp4"
        try:
            await download_video_async(result["video_url"], dest)
            return {"status": "completed", "url": result["video_url"], "local_path": str(dest)}
        except Exception as e:
            logger.error(f"Video download failed: {e}")
            return {"status": "completed", "url": result["video_url"], "local_path": None}
    elif result["status"] == "failed":
        return {"status": "failed", "url": None, "local_path": None}
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
