from pathlib import Path

from api.generation import (
    apply_default_visual_macros,
    asset_output_path,
    asset_prompt,
    collect_asset_reference_images,
    collect_video_reference_images,
    ensure_bgm_refs,
    ensure_referenced_assets,
    normalize_video_output_paths,
    selected_bgm_ref,
    sync_board_metadata,
)
import api.poller as poller
from api.routes.settings import update_settings
from api.routes.settings import SettingsModel
from api.routes.settings import get_api_key
from api.routes.settings import get_settings
from api.routes.project import list_projects, plan_project_script, project_preflight, reset_project_storyboards
from api.routes.project import (
    ConcatEpisodeRequest,
    DecomposeRequest,
    SelectAudioRefRequest,
    SelectBgmRefRequest,
    _build_ffmpeg_concat_command,
    _build_planned_board,
    _build_segments_outline,
    _call_llm,
    _episode_subtitle_events,
    _episode_groups,
    _allowed_characters_for_board,
    _segment_sort_key,
    _refresh_storyboard_prompt_context,
    _render_subtitle_sequence,
    _source_trace_for_board,
    _story_continuity_for_board,
    _planned_board_prompt,
    _visible_identity_refs_for_board,
    _subtitle_chunks,
    _voice_timeline_for_board,
    concat_episode,
    decompose_project,
    select_audio_ref,
    select_bgm_ref,
)
from api.routes.tasks import (
    ProjectRequest,
    SubmitAssetRequest,
    SubmitStoryboardRequest,
    SubmitVideoRequest,
    submit_asset,
    submit_storyboards,
    submit_storyboard,
    submit_video,
    submit_videos,
)
from api.llm import get_llm_config
from api.wetoken import _get_cached_asset_uri, submit_video_task
from api.prompts import assemble_storyboard_prompt, assemble_video_prompt


def test_masked_settings_values_preserve_existing_secret(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("api.routes.settings.SETTINGS_PATH", settings_path)

    settings_path.write_text(
        '{"vidu_api_key":"real-vidu-secret","wetoken_api_key":"real-wetoken-secret",'
        '"idealab_api_key":"","idealab_base_url":"https://api.idealab.com/v1",'
        '"llm_provider":"deepseek","deepseek_api_key":"real-deepseek-secret",'
        '"deepseek_base_url":"https://api.deepseek.com","deepseek_model":"deepseek-v4-flash",'
        '"gh_token":"","gh_owner":"","gh_repo":""}',
        encoding="utf-8",
    )

    import asyncio

    asyncio.run(update_settings(SettingsModel(
        vidu_api_key="***********cret",
        wetoken_api_key="",
        deepseek_api_key="***************cret",
        deepseek_model="deepseek-v4-flash",
    )))

    saved = settings_path.read_text(encoding="utf-8")
    assert "real-vidu-secret" in saved
    assert "real-deepseek-secret" in saved
    assert '"wetoken_api_key": ""' in saved


def test_masked_settings_values_do_not_count_as_configured_key(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("api.routes.settings.SETTINGS_PATH", settings_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("VIDU_API_KEY", raising=False)

    settings_path.write_text(
        '{"vidu_api_key":"********o0Yg","wetoken_api_key":"","idealab_api_key":"",'
        '"idealab_base_url":"https://api.idealab.com/v1","llm_provider":"deepseek",'
        '"deepseek_api_key":"********f36b","deepseek_base_url":"https://api.deepseek.com",'
        '"deepseek_model":"deepseek-v4-flash","gh_token":"","gh_owner":"","gh_repo":""}',
        encoding="utf-8",
    )

    assert get_api_key("VIDU_API_KEY") == ""
    assert get_llm_config()["has_api_key"] is False

    import asyncio

    public = asyncio.run(get_settings())
    assert public["_masked_vidu"] is True
    assert public["_masked_deepseek"] is True
    assert public["_has_vidu"] is False
    assert public["_has_deepseek"] is False


def test_call_llm_empty_response_returns_structured_error(monkeypatch):
    async def fake_generate(api_key, system_prompt, user_message):
        return ""

    monkeypatch.setattr("api.routes.project.generate_prompt_async", fake_generate)

    import asyncio
    from fastapi import HTTPException

    try:
        asyncio.run(_call_llm("key", "system", "user"))
    except HTTPException as exc:
        assert exc.status_code == 500
        assert exc.detail["error_type"] == "empty_llm_response"
    else:
        raise AssertionError("_call_llm should reject empty LLM responses")


def test_decompose_reuses_assets_and_saves_outline_and_boards(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","narration_style":"third_person","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"seed":"黑色巨蟒"}},"scenes":{"工作室":{"seed":"工作室"}},"props":{}},'
        '"narration_segments":{}}',
        encoding="utf-8",
    )
    calls = []

    async def fake_call_llm(api_key, system_prompt, user_message, attempts=1):
        calls.append(system_prompt)
        return {
            "video_goal": "展示玄墨异常行为",
            "shot_timeline": [
                {
                    "shot_id": f"s{i:02d}",
                    "start": i - 1,
                    "end": i,
                    "duration": 1,
                    "voice_refs": ["v01"],
                    "visual": "玄墨盘踞在箱中",
                    "camera": "近景",
                    "characters": ["玄墨"],
                    "scene": "",
                    "match_strategy": "sync",
                    "purpose": "表现异常",
                    "audio_behavior": "narration_sync",
                    "continuity_from_previous": None if i == 1 else "延续上一镜",
                    "transition_type": None if i == 1 else "cut",
                }
                for i in range(1, 4)
            ],
            "asset_refs": {"characters": ["玄墨"], "scene": ["工作室"], "props": []},
        }

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.project._resolve_api_key", lambda: "llm-key")
    monkeypatch.setattr("api.routes.project._call_llm", fake_call_llm)

    import asyncio
    import json

    result = asyncio.run(decompose_project(DecomposeRequest(project_name="大蟒蛇")))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["stats"]["segments"] == 1
    assert "_decomposition_outline" in saved
    assert saved["_decompose_progress"]["status"] == "completed"
    assert saved["_decompose_progress"]["stage"] == "done"
    board = saved["narration_segments"]["seg_1_1"]["boards"][0]
    assert board["storyboard_image"]["prompt"]
    assert board["asset_refs"]["scene"] == "工作室"
    assert len(calls) == 1


def test_decompose_falls_back_when_board_llm_fails(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","narration_style":"third_person","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"seed":"黑色巨蟒"}},"scenes":{},"props":{}},'
        '"narration_segments":{}}',
        encoding="utf-8",
    )

    async def fake_call_llm(api_key, system_prompt, user_message, attempts=1):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail={"error_type": "json_parse_error", "message": "LLM 返回不是合法 JSON"},
        )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.project._resolve_api_key", lambda: "llm-key")
    monkeypatch.setattr("api.routes.project._call_llm", fake_call_llm)

    import asyncio
    import json

    result = asyncio.run(decompose_project(DecomposeRequest(project_name="大蟒蛇")))

    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    board = saved["narration_segments"]["seg_1_1"]["boards"][0]
    assert result["ok"] is True
    assert saved["_decompose_progress"]["status"] == "completed"
    assert board["video_goal"] == "呈现本段关键情节"
    assert board["shot_timeline"]
    assert result["validation_errors"]


def test_prompt_assembly_tolerates_null_shot_fields():
    board = {
        "board_duration": 2,
        "video_goal": "测试",
        "voice_timeline": [{"beat_id": "v01", "start": 0, "end": 2, "speaker": "旁白", "text": "测试"}],
        "shot_timeline": [{
            "shot_id": "s01",
            "start": 0,
            "end": 2,
            "camera": None,
            "visual": None,
            "audio_behavior": None,
            "match_strategy": None,
            "purpose": None,
            "voice_refs": ["v01"],
        }],
        "asset_refs": {"characters": [], "scene": "", "props": []},
    }

    assert "导演分镜板" in assemble_storyboard_prompt(board)
    assert "生成要求：" in assemble_video_prompt(board)


def test_segment_sort_key_orders_numbered_segments_naturally():
    keys = ["seg_1_10", "seg_1_2", "seg_1_1"]

    assert sorted(keys, key=_segment_sort_key) == ["seg_1_1", "seg_1_2", "seg_1_10"]


def test_poll_storyboard_accepts_vidu_success_and_downloads_image(tmp_path, monkeypatch):
    image_path = tmp_path / "storyboards" / "seg_01_01_p01.jpg"

    async def fake_download(url, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"image")

    monkeypatch.setattr("api.vidu.poll_task", lambda api_key, task_id: {
        "status": "success",
        "image_url": "https://example.test/storyboard.jpg",
        "error": None,
    })
    monkeypatch.setattr("api.vidu.download_image_async", fake_download)

    import asyncio

    result = asyncio.run(poller._poll_storyboard(
        "vidu-key",
        "task-1",
        image_path,
    ))

    assert result["status"] == "completed"
    assert result["url"] == "https://example.test/storyboard.jpg"
    assert result["local_path"] == str(image_path)
    assert image_path.exists()


def test_submit_video_uses_all_reference_images_without_github(tmp_path, monkeypatch):
    image_path = tmp_path / "storyboard.jpg"
    image_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    )
    captured = {}

    class FakeResponse:
        is_success = True

        def json(self):
            return {"id": "video-task-1"}

    def fake_post(url, headers, json, timeout, **kwargs):
        captured["body"] = json
        return FakeResponse()

    monkeypatch.setattr("api.wetoken._get_gh_config", lambda: ("", "", ""))
    monkeypatch.setattr("api.wetoken.httpx.post", fake_post)

    task_id = submit_video_task(
        api_key="wetoken-key",
        prompt="生成视频",
        image_paths=[str(image_path), "https://example.test/character.jpg"],
        project_dir=tmp_path,
    )

    assert task_id == "video-task-1"
    refs = [
        item["image_url"]["url"]
        for item in captured["body"]["content"]
        if item.get("role") == "reference_image"
    ]
    assert len(refs) == 2
    assert refs[0].startswith("data:image/")
    assert refs[1] == "https://example.test/character.jpg"


def test_submit_video_retries_transient_network_timeout(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        is_success = True

        def json(self):
            return {"id": "video-task-1"}

    def flaky_post(url, headers, json, timeout, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("_ssl.c:990: The handshake operation timed out")
        return FakeResponse()

    monkeypatch.setattr("api.wetoken.httpx.post", flaky_post)
    monkeypatch.setattr("api.wetoken.time.sleep", lambda _: None)

    task_id = submit_video_task(
        api_key="wetoken-key",
        prompt="生成视频",
        image_paths=[],
    )

    assert task_id == "video-task-1"
    assert calls["count"] == 2


def test_submit_video_includes_reference_audio(tmp_path, monkeypatch):
    image_path = tmp_path / "storyboard.jpg"
    image_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    )
    audio_path = tmp_path / "男声.mp3"
    audio_path.write_bytes(b"fake audio")
    captured = {}

    class FakeResponse:
        is_success = True

        def json(self):
            return {"id": "video-task-1"}

    def fake_post(url, headers, json, timeout, **kwargs):
        captured["body"] = json
        return FakeResponse()

    monkeypatch.setattr("api.wetoken._get_gh_config", lambda: ("token", "owner", "repo"))
    monkeypatch.setattr("api.wetoken.upload_local_image", lambda *args, **kwargs: "asset://image-1")
    monkeypatch.setattr("api.wetoken.upload_local_audio", lambda *args, **kwargs: "asset://audio-1")
    monkeypatch.setattr("api.wetoken.httpx.post", fake_post)

    task_id = submit_video_task(
        api_key="wetoken-key",
        prompt="生成视频",
        image_paths=[str(image_path)],
        audio_paths=[str(audio_path)],
        project_dir=tmp_path,
    )

    assert task_id == "video-task-1"
    audio_refs = [
        item["audio_url"]["url"]
        for item in captured["body"]["content"]
        if item.get("role") == "reference_audio"
    ]
    assert audio_refs == ["asset://audio-1"]


def test_submit_video_auto_selects_local_audio_ref_for_consistent_narration(tmp_path, monkeypatch):
    storyboard = tmp_path / "storyboard.jpg"
    character = tmp_path / "xuanmo.jpg"
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    audio_dir = project / "audio_refs"
    audio_ref = audio_dir / "稳定男声.mp3"
    project.mkdir(parents=True)
    audio_dir.mkdir()
    for path in (storyboard, character, audio_ref):
        path.write_bytes(b"file")
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{},"props":{}},'
        '"audio_refs":{"selected":"","options":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        '"page":1,"total_pages":1,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
        '"video_goal":"测试视频",'
        '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
        '"shot_timeline":['
        '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
        '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
        '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
        '"video":{"status":"needed","prompt":"做视频"}}]}}}',
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")
    monkeypatch.setattr("api.wetoken.submit_video_task", lambda **kwargs: captured.update(kwargs) or "video-task")

    import asyncio
    import json

    asyncio.run(submit_video(SubmitVideoRequest(project_name="大蟒蛇", segment_key="seg_01", board_index=0)))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    video = saved["narration_segments"]["seg_01"]["boards"][0]["video"]

    assert saved["audio_refs"]["selected"] == "稳定男声"
    assert captured["audio_paths"] == [str(audio_ref)]
    assert "稳定男声" in captured["prompt"]
    assert video["reference_audios"] == [str(audio_ref)]


def test_submit_video_requires_asset_upload_when_github_configured(tmp_path, monkeypatch):
    image_path = tmp_path / "storyboard.jpg"
    image_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    )
    monkeypatch.setattr("api.wetoken._get_gh_config", lambda: ("token", "owner", "repo"))
    monkeypatch.setattr("api.wetoken.upload_local_image", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("github failed")))

    import pytest

    with pytest.raises(RuntimeError, match="github failed"):
        submit_video_task(
            api_key="wetoken-key",
            prompt="生成视频",
            image_paths=[str(image_path)],
            project_dir=tmp_path,
        )


def test_wetoken_cached_asset_uri_requires_current_source_file(tmp_path):
    image_path = tmp_path / "storyboard.jpg"
    image_path.write_bytes(b"old storyboard")
    stat = image_path.stat()
    ledger = {
        "assets": {
            "asset-1": {
                "asset_uri": "asset://asset-1",
                "source_path": str(image_path),
                "source_mtime": stat.st_mtime,
                "source_size": stat.st_size,
                "type": "Image",
                "status": "Active",
            },
            "legacy-asset": {
                "asset_uri": "asset://legacy-asset",
                "source_path": str(image_path),
                "type": "Image",
                "status": "Active",
            },
        }
    }

    assert _get_cached_asset_uri(ledger, str(image_path), "Image") == "asset://asset-1"

    image_path.write_bytes(b"new storyboard with regenerated content")

    assert _get_cached_asset_uri(ledger, str(image_path), "Image") is None
    assert _get_cached_asset_uri({"assets": {"legacy-asset": ledger["assets"]["legacy-asset"]}}, str(image_path), "Image") is None


def test_episode_groups_report_readiness_and_export(tmp_path):
    data = {
        "narration_segments": {
            "seg_1_1": {"episode": 1, "boards": [
                {"board_duration": 8, "video": {"status": "completed", "local_path": "/tmp/a.mp4"}},
                {"board_duration": 7, "video": {"status": "needed", "local_path": None}},
            ]},
            "seg_2_1": {"episode": 2, "boards": [
                {"board_duration": 6, "video": {"status": "completed", "local_path": "/tmp/b.mp4"}},
            ]},
        },
        "episode_exports": {
            "2": {"status": "completed", "local_path": "/tmp/e2.mp4", "duration": 6}
        },
    }

    episodes = _episode_groups(data)

    assert episodes[0]["episode"] == 1
    assert episodes[0]["videos_completed"] == 1
    assert episodes[0]["videos_total"] == 2
    assert episodes[0]["can_concat"] is False
    assert episodes[1]["export"]["local_path"] == "/tmp/e2.mp4"


def test_select_audio_ref_persists_choice(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","assets":{"characters":{},"scenes":{},"props":{}},'
        '"audio_refs":{"selected":"男声","options":{"男声":{"name":"男声","local_path":"/tmp/male.mp3"}}},'
        '"narration_segments":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import asyncio
    result = asyncio.run(select_audio_ref(SelectAudioRefRequest(project_name="大蟒蛇", selected="")))

    assert result["audio_refs"]["selected"] == ""


def test_bgm_refs_scan_project_and_selection_persists(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    bgm = project / "bgm" / "悬疑氛围.mp3"
    bgm.parent.mkdir(parents=True)
    bgm.write_bytes(b"fake bgm")
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","assets":{"characters":{},"scenes":{},"props":{}},'
        '"narration_segments":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import asyncio
    import json

    data = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    ensure_bgm_refs(data, project)
    assert "悬疑氛围" in data["bgm_refs"]["options"]
    assert selected_bgm_ref(data) is None

    result = asyncio.run(select_bgm_ref(SelectBgmRefRequest(project_name="大蟒蛇", selected="悬疑氛围")))
    assert result["bgm_refs"]["selected"] == "悬疑氛围"
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    assert selected_bgm_ref(saved)["path"] == str(bgm)


def test_concat_episode_records_export(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    video = tmp_path / "part.mp4"
    video.write_bytes(b"video")
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","assets":{"characters":{},"scenes":{},"props":{}},'
        f'"narration_segments":{{"seg_1_1":{{"episode":1,"boards":[{{"board_duration":4,"video":{{"status":"completed","local_path":"{video}"}}}}]}}}}}}',
        encoding="utf-8",
    )

    def fake_concat(inputs, output, subtitle_events=None):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"merged")
        assert subtitle_events == []

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.project._concat_video_files", fake_concat)

    import asyncio
    result = asyncio.run(concat_episode(ConcatEpisodeRequest(project_name="大蟒蛇", episode=1, copy_to_desktop=False)))

    assert result["export"]["status"] == "completed"
    assert result["export"]["subtitle_style"] == "large_white_black_outline"
    assert Path(result["export"]["local_path"]).exists()


def test_concat_episode_passes_selected_bgm_to_concat_and_records_export(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    video = tmp_path / "part.mp4"
    bgm = project / "bgm" / "悬疑氛围.mp3"
    video.write_bytes(b"video")
    bgm.parent.mkdir(parents=True)
    bgm.write_bytes(b"bgm")
    project.mkdir(parents=True, exist_ok=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","assets":{"characters":{},"scenes":{},"props":{}},'
        '"bgm_refs":{"selected":"悬疑氛围","options":{"悬疑氛围":{"name":"悬疑氛围","local_path":"' + str(bgm) + '"}}},'
        f'"narration_segments":{{"seg_1_1":{{"episode":1,"boards":[{{"board_duration":4,"video":{{"status":"completed","local_path":"{video}"}}}}]}}}}}}',
        encoding="utf-8",
    )
    captured = {}

    def fake_concat(inputs, output, subtitle_events=None, bgm_path=None):
        captured["bgm_path"] = bgm_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"merged")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.project._concat_video_files", fake_concat)

    import asyncio
    result = asyncio.run(concat_episode(ConcatEpisodeRequest(project_name="大蟒蛇", episode=1, copy_to_desktop=False)))

    assert captured["bgm_path"] == bgm
    assert result["export"]["bgm"]["name"] == "悬疑氛围"


def test_ffmpeg_concat_command_reencodes_with_hard_cuts(tmp_path):
    inputs = [tmp_path / f"part{i}.mp4" for i in range(4)]
    output = tmp_path / "episode.mp4"

    command = _build_ffmpeg_concat_command(
        "/opt/homebrew/bin/ffmpeg",
        inputs,
        output,
        durations=[11.0, 15.0, 12.0, 9.0],
        audio_flags=[True, True, True, True],
        subtitle_sequence={"pattern": tmp_path / "sub_%05d.png", "fps": 6, "y": 880},
    )
    command_text = " ".join(command)

    assert "-c copy" not in command_text
    assert "-loop" not in command
    assert "-framerate" in command
    assert "xfade=" not in command_text
    assert "acrossfade" not in command_text
    assert "concat=n=4:v=1:a=1" in command_text
    assert command_text.count("overlay=0:880:shortest=1") == 1
    assert "libx264" in command
    assert "yuv420p" in command


def test_ffmpeg_concat_command_mixes_bgm_under_existing_audio(tmp_path):
    inputs = [tmp_path / "part.mp4"]
    bgm = tmp_path / "bgm.mp3"
    output = tmp_path / "episode.mp4"

    command = _build_ffmpeg_concat_command(
        "/opt/homebrew/bin/ffmpeg",
        inputs,
        output,
        durations=[8.0],
        audio_flags=[True],
        bgm_path=bgm,
    )
    command_text = " ".join(command)

    assert str(bgm) in command
    assert "volume=0.18" in command_text
    assert "amix=inputs=2" in command_text
    assert "-map [aout]" in command_text


def test_ffmpeg_concat_command_overlays_subtitles_for_single_clip(tmp_path):
    command = _build_ffmpeg_concat_command(
        "/opt/homebrew/bin/ffmpeg",
        [tmp_path / "part.mp4"],
        tmp_path / "episode.mp4",
        durations=[8.0],
        audio_flags=[True],
        subtitle_sequence={"pattern": tmp_path / "sub_%05d.png", "fps": 6, "y": 880},
    )
    command_text = " ".join(command)

    assert "-filter_complex" in command
    assert "overlay=0:880:shortest=1" in command_text
    assert "0:a?" in command


def test_episode_subtitle_events_split_long_voice_text():
    data = {
        "narration_segments": {
            "seg_1_1": {
                "boards": [{
                    "voice_timeline": [{
                        "start": 0,
                        "end": 4,
                        "text": "你这孩子胆子是真的大，但是现在必须马上离开这里。",
                    }],
                    "video": {"local_path": "/tmp/a.mp4"},
                }]
            }
        }
    }

    events = _episode_subtitle_events(data, [{"segment_key": "seg_1_1", "board_index": 0}])

    assert _subtitle_chunks("你这孩子胆子是真的大") == ["你这孩子胆子是真的大"]
    assert _subtitle_chunks("第2集：你这孩子胆子是真的大，但是现在必须马上离开这里。") == [
        "你这孩子胆子是真的大",
        "但是现在必须马上离开这里",
    ]
    assert len(events) >= 2
    assert all(event["part_index"] == 0 for event in events)
    assert events[0]["text"] == "你这孩子胆子是真的大"
    assert events[0]["start"] == 0
    assert events[-1]["end"] == 4


def test_render_subtitle_sequence_scales_long_text_inside_strip(tmp_path):
    from PIL import Image

    result = _render_subtitle_sequence(
        [{"part_index": 0, "start": 0, "end": 1, "text": "超长字幕" * 20}],
        [0.0],
        tmp_path,
        total_duration=1.0,
        fps=1,
    )

    assert result is not None
    frame = Image.open(tmp_path / "subtitle_00000.png")
    alpha_bbox = frame.getchannel("A").getbbox()
    assert alpha_bbox is not None
    assert alpha_bbox[0] >= 12
    assert alpha_bbox[2] <= 708


def test_planned_board_review_removes_characters_without_source_evidence():
    assets = {
        "characters": {
            "沈砚": {"seed": "主角"},
            "玄墨": {"seed": "黑色蟒蛇"},
            "郑教授": {"seed": "退休生物学教授"},
            "苏晴": {"seed": "主角女友"},
        },
        "scenes": {"工作室": {"seed": "开放式工作室"}},
        "props": {},
    }
    script_plan = {
        "script_slices": [
            {
                "slice_id": "s001",
                "source_start": 0,
                "source_end": 35,
                "kind": "narration",
                "speaker": "旁白",
                "text": "郑教授摆了摆手，",
            },
            {
                "slice_id": "s002",
                "source_start": 35,
                "source_end": 70,
                "kind": "dialogue",
                "speaker": "角色",
                "text": "你这孩子胆子是真的大，这可是体型不小的蟒蛇。",
            },
        ],
        "voice_beats": [
            {
                "beat_id": "v001",
                "source_slice_ids": ["s001"],
                "type": "narration",
                "speaker": "旁白",
                "text": "郑教授摆了摆手，",
                "duration": 2,
            },
            {
                "beat_id": "v002",
                "source_slice_ids": ["s002"],
                "type": "dialogue",
                "speaker": "角色",
                "text": "你这孩子胆子是真的大，这可是体型不小的蟒蛇。",
                "duration": 5,
            },
        ],
    }
    plan_item = {
        "board_id": "b0028",
        "voice_beat_ids": ["v001", "v002"],
        "source_slice_ids": ["s001", "s002"],
        "duration": 7,
    }

    voice = _voice_timeline_for_board(script_plan, plan_item, "third_person", "沈砚", assets)
    trace = _source_trace_for_board(script_plan, plan_item)
    allowed = _allowed_characters_for_board(voice, trace, assets, "沈砚")
    board = _build_planned_board(
        plan_item=plan_item,
        page=28,
        total_pages=407,
        voice_timeline=voice,
        assets=assets,
        llm_result={
            "video_goal": "错误地写成苏晴反应",
            "asset_refs": {"characters": ["沈砚", "苏晴", "玄墨"], "scene": "工作室", "props": []},
            "shot_timeline": [
                {
                    "shot_id": "s01",
                    "start": 0,
                    "end": 7,
                    "duration": 7,
                    "voice_refs": ["v01", "v02"],
                    "visual": "苏晴看着玄墨后退",
                    "camera": "近景",
                    "characters": ["苏晴", "玄墨"],
                    "scene": "工作室",
                    "match_strategy": "sync",
                    "purpose": "错误示例",
                    "audio_behavior": "dialogue_sync",
                    "continuity_from_previous": None,
                    "transition_type": None,
                }
            ],
        },
        source_trace=trace,
        allowed_characters=allowed,
    )

    assert "郑教授" in allowed
    assert "玄墨" in allowed
    assert "苏晴" not in allowed
    assert "苏晴" not in board["asset_refs"]["characters"]
    assert "苏晴" in board["review"]["removed_characters"]
    assert "郑教授摆了摆手" in board["review"]["source_excerpt"]


def test_planned_board_keeps_xuanmo_for_pronoun_snake_evidence():
    assets = {
        "characters": {
            "沈砚": {"seed": "主角"},
            "玄墨": {"seed": "一条成年基因突变的缅甸蟒"},
            "苏晴": {"seed": "女友"},
        },
        "scenes": {"工作室": {"seed": "开放式Loft工作室"}},
        "props": {},
    }
    script_plan = {
        "script_slices": [
            {
                "slice_id": "s001",
                "kind": "narration",
                "speaker": "旁白",
                "text": "它将扁平的头部轻轻搁在我的肩窝，分叉的信子在空气中探知。",
            },
        ],
        "voice_beats": [
            {
                "beat_id": "v001",
                "source_slice_ids": ["s001"],
                "type": "narration",
                "speaker": "旁白",
                "text": "它将扁平的头部轻轻搁在我的肩窝，分叉的信子在空气中探知。",
                "duration": 6,
            },
        ],
    }
    plan_item = {"board_id": "b0016", "voice_beat_ids": ["v001"], "source_slice_ids": ["s001"], "duration": 6}

    voice = _voice_timeline_for_board(script_plan, plan_item, "third_person", "沈砚", assets)
    trace = _source_trace_for_board(script_plan, plan_item)
    allowed = _allowed_characters_for_board(voice, trace, assets, "沈砚")
    board = _build_planned_board(
        plan_item=plan_item,
        page=16,
        total_pages=82,
        voice_timeline=voice,
        assets=assets,
        llm_result={
            "video_goal": "主角和蛇亲近",
            "asset_refs": {"characters": ["沈砚"], "scene": "工作室", "props": []},
            "shot_timeline": [],
        },
        source_trace=trace,
        allowed_characters=allowed,
    )

    assert "玄墨" in allowed
    assert "玄墨" in board["asset_refs"]["characters"]
    assert "苏晴" not in board["asset_refs"]["characters"]


def test_planned_board_does_not_carry_previous_human_into_snake_only_board():
    assets = {
        "characters": {
            "沈砚": {"seed": "主角"},
            "玄墨": {"seed": "一条成年基因突变的缅甸蟒"},
            "郑教授": {"seed": "退休生物学教授"},
        },
        "scenes": {"工作室": {"seed": "开放式Loft工作室"}},
        "props": {},
    }
    script_plan = {
        "script_slices": [
            {"slice_id": "s001", "kind": "narration", "speaker": "旁白", "text": "郑教授摆了摆手。"},
            {
                "slice_id": "s002",
                "kind": "narration",
                "speaker": "旁白",
                "text": "它正趴在加热石上，享受着模拟日光的照射。",
            },
        ],
        "voice_beats": [
            {
                "beat_id": "v001",
                "source_slice_ids": ["s001"],
                "type": "narration",
                "speaker": "旁白",
                "text": "郑教授摆了摆手。",
                "duration": 2,
            },
            {
                "beat_id": "v002",
                "source_slice_ids": ["s002"],
                "type": "narration",
                "speaker": "旁白",
                "text": "它正趴在加热石上，享受着模拟日光的照射。",
                "duration": 6,
            },
        ],
    }
    plan_item = {"board_id": "b0030", "voice_beat_ids": ["v002"], "source_slice_ids": ["s002"], "duration": 6}

    voice = _voice_timeline_for_board(script_plan, plan_item, "third_person", "沈砚", assets)
    trace = _source_trace_for_board(script_plan, plan_item)
    allowed = _allowed_characters_for_board(voice, trace, assets, "沈砚", "郑教授摆了摆手。")
    board = _build_planned_board(
        plan_item=plan_item,
        page=30,
        total_pages=82,
        voice_timeline=voice,
        assets=assets,
        llm_result={
            "video_goal": "玄墨趴在加热石上",
            "asset_refs": {"characters": ["玄墨", "郑教授"], "scene": "工作室", "props": []},
            "shot_timeline": [],
        },
        source_trace=trace,
        allowed_characters=allowed,
    )

    assert "玄墨" in allowed
    assert "郑教授" not in allowed
    assert board["asset_refs"]["characters"] == ["玄墨"]
    assert "郑教授" in board["review"]["removed_characters"]


def test_first_person_pet_dialogue_resolves_to_narrator_without_other_speaker_context():
    assets = {
        "characters": {
            "沈砚": {"seed": "年轻男性主角"},
            "玄墨": {"seed": "黑色巨蟒"},
        },
        "scenes": {},
        "props": {},
    }
    script_plan = {
        "script_slices": [
            {"slice_id": "s001", "kind": "narration", "text": "玄墨蹭了蹭我的脖颈。"},
            {"slice_id": "s002", "kind": "dialogue", "speaker": "角色", "text": "好了好了，别撒娇了，赶紧进食吧。"},
        ],
        "voice_beats": [
            {"beat_id": "v001", "source_slice_ids": ["s001"], "type": "narration", "text": "玄墨蹭了蹭我的脖颈。", "duration": 4},
            {"beat_id": "v002", "source_slice_ids": ["s002"], "type": "dialogue", "speaker": "角色", "text": "好了好了，别撒娇了，赶紧进食吧。", "duration": 4},
        ],
    }
    plan_item = {"board_id": "b0001", "voice_beat_ids": ["v001", "v002"], "source_slice_ids": ["s001", "s002"]}

    voice = _voice_timeline_for_board(script_plan, plan_item, "first_person", "沈砚", assets)

    assert voice[1]["speaker"] == "沈砚"


def test_dialogue_response_uses_board_context_to_resolve_professor_speaker():
    assets = {
        "characters": {
            "沈砚": {"seed": "年轻男性主角"},
            "郑教授": {"seed": "退休教授"},
            "玄墨": {"seed": "黑色巨蟒"},
        },
        "scenes": {},
        "props": {},
    }
    script_plan = {
        "script_slices": [
            {"slice_id": "s001", "kind": "narration", "text": "我笑着走过去接过杨梅，"},
            {"slice_id": "s002", "kind": "dialogue", "speaker": "角色", "text": "您进来喝杯茶？"},
            {"slice_id": "s003", "kind": "dialogue", "speaker": "角色", "text": "不了不了，我看着它盘起来的样子，心里就发怵。"},
            {"slice_id": "s004", "kind": "narration", "text": "郑教授摆了摆手，"},
        ],
        "voice_beats": [
            {"beat_id": "v001", "source_slice_ids": ["s001"], "type": "narration", "text": "我笑着走过去接过杨梅，", "duration": 2},
            {"beat_id": "v002", "source_slice_ids": ["s002"], "type": "dialogue", "speaker": "角色", "text": "您进来喝杯茶？", "duration": 2},
            {"beat_id": "v003", "source_slice_ids": ["s003"], "type": "dialogue", "speaker": "角色", "text": "不了不了，我看着它盘起来的样子，心里就发怵。", "duration": 5},
            {"beat_id": "v004", "source_slice_ids": ["s004"], "type": "narration", "text": "郑教授摆了摆手，", "duration": 2},
        ],
    }
    plan_item = {"board_id": "b0001", "voice_beat_ids": ["v001", "v002", "v003", "v004"], "source_slice_ids": ["s001", "s002", "s003", "s004"]}

    voice = _voice_timeline_for_board(script_plan, plan_item, "first_person", "沈砚", assets)

    assert voice[1]["speaker"] == "沈砚"
    assert voice[2]["speaker"] == "郑教授"


def test_story_continuity_uses_previous_summary_and_current_event_without_next_context():
    assets = {
        "characters": {"沈砚": {"seed": "年轻男性主角"}, "玄墨": {"seed": "黑色巨蟒"}},
        "scenes": {"工作室": {"seed": "工作室饲养箱环境"}},
        "props": {},
    }
    script_plan = {
        "script_slices": [
            {"slice_id": "s001", "kind": "narration", "text": "沈砚刚用镊子把白鼠递到玄墨面前。"},
            {"slice_id": "s002", "kind": "narration", "text": "玄墨回到饲养箱中央，一动不动地凝视沈砚。"},
            {"slice_id": "s003", "kind": "narration", "text": "深夜里，床边传来鳞片摩擦声。"},
        ],
        "voice_beats": [
            {"beat_id": "v001", "source_slice_ids": ["s001"], "type": "narration", "text": "沈砚刚用镊子把白鼠递到玄墨面前。", "duration": 4},
            {"beat_id": "v002", "source_slice_ids": ["s002"], "type": "narration", "text": "玄墨回到饲养箱中央，一动不动地凝视沈砚。", "duration": 5},
            {"beat_id": "v003", "source_slice_ids": ["s003"], "type": "narration", "text": "深夜里，床边传来鳞片摩擦声。", "duration": 4},
        ],
    }
    plan_item = {"board_id": "b0002", "voice_beat_ids": ["v002"], "source_slice_ids": ["s002"], "duration": 5}

    voice = _voice_timeline_for_board(script_plan, plan_item, "third_person", "沈砚", assets)
    trace = _source_trace_for_board(script_plan, plan_item)
    continuity = _story_continuity_for_board(script_plan, plan_item, voice, trace, assets)

    assert continuity["just_happened"] == "沈砚刚用镊子把白鼠递到玄墨面前。"
    assert continuity["now_happening"] == "玄墨回到饲养箱中央，一动不动地凝视沈砚。"
    assert continuity["previous_final_panel"] == ""
    assert "next_happens" not in continuity


def test_story_continuity_tracks_prior_prop_used_by_generic_current_reference():
    assets = {
        "characters": {"沈砚": {"seed": "年轻男性主角"}, "玄墨": {"seed": "黑色巨蟒"}},
        "scenes": {"工作室": {"seed": "工作室饲养箱环境"}},
        "props": {"白鼠": {"seed": "一只成年白色实验鼠，活蹦乱跳"}},
    }
    script_plan = {
        "script_slices": [
            {"slice_id": "s001", "kind": "narration", "text": "可今天，它只是漠然地瞥了一眼那只活蹦乱跳的白鼠，随即扭过头。"},
            {"slice_id": "s002", "kind": "dialogue", "speaker": "沈砚", "text": "哟？这是要改吃素了？这可是我特意为你准备的营养餐，脂肪含量刚好达标。"},
        ],
        "voice_beats": [
            {"beat_id": "v001", "source_slice_ids": ["s001"], "type": "narration", "text": "可今天，它只是漠然地瞥了一眼那只活蹦乱跳的白鼠，随即扭过头。", "duration": 5},
            {"beat_id": "v002", "source_slice_ids": ["s002"], "type": "dialogue", "speaker": "沈砚", "text": "哟？这是要改吃素了？这可是我特意为你准备的营养餐，脂肪含量刚好达标。", "duration": 5},
        ],
    }
    plan_item = {"board_id": "b0002", "voice_beat_ids": ["v002"], "source_slice_ids": ["s002"], "duration": 5}

    voice = _voice_timeline_for_board(script_plan, plan_item, "first_person", "沈砚", assets)
    trace = _source_trace_for_board(script_plan, plan_item)
    continuity = _story_continuity_for_board(script_plan, plan_item, voice, trace, assets)
    state = continuity["state_context"]

    assert state["active_props"][0]["name"] == "白鼠"
    assert "营养餐" in state["active_props"][0]["aliases"]
    assert "白鼠" in state["active_props"][0]["evidence"]


def test_story_continuity_tracks_non_asset_object_for_scene_logic():
    assets = {
        "characters": {"沈砚": {"seed": "年轻男性主角"}, "郑教授": {"seed": "退休教授"}},
        "scenes": {"工作室": {"seed": "工作室"}},
        "props": {},
    }
    script_plan = {
        "script_slices": [
            {"slice_id": "s001", "kind": "narration", "text": "郑教授扶着门框，手里提着一袋刚从院子里摘的杨梅，死活不肯往里多走一步。"},
            {"slice_id": "s002", "kind": "narration", "text": "我笑着走过去接过杨梅，您进来喝杯茶？郑教授摆了摆手，"},
        ],
        "voice_beats": [
            {"beat_id": "v001", "source_slice_ids": ["s001"], "type": "narration", "text": "郑教授扶着门框，手里提着一袋刚从院子里摘的杨梅，死活不肯往里多走一步。", "duration": 6},
            {"beat_id": "v002", "source_slice_ids": ["s002"], "type": "narration", "text": "我笑着走过去接过杨梅，您进来喝杯茶？郑教授摆了摆手，", "duration": 6},
        ],
    }
    plan_item = {"board_id": "b0002", "voice_beat_ids": ["v002"], "source_slice_ids": ["s002"], "duration": 6}

    voice = _voice_timeline_for_board(script_plan, plan_item, "first_person", "沈砚", assets)
    trace = _source_trace_for_board(script_plan, plan_item)
    continuity = _story_continuity_for_board(script_plan, plan_item, voice, trace, assets)
    state = continuity["state_context"]

    assert any(item["name"] == "杨梅" for item in state["active_objects"])
    assert state["incomplete_source"] is True


def test_planned_board_prompt_includes_evidence_driven_continuity_state():
    assets = {
        "characters": {"沈砚": {"seed": "年轻男性主角"}, "玄墨": {"seed": "黑色巨蟒"}},
        "scenes": {"工作室": {"seed": "工作室饲养箱环境"}},
        "props": {"白鼠": {"seed": "一只成年白色实验鼠，活蹦乱跳"}},
    }
    continuity = {
        "just_happened": "玄墨刚避开白鼠。",
        "now_happening": "沈砚调侃营养餐。",
        "state_context": {
            "active_props": [{"name": "白鼠", "state": "仍是当前投喂物", "aliases": ["营养餐"], "evidence": "上一板白鼠仍在投喂位置"}],
            "open_actions": [{"action": "投喂尚未结束", "evidence": "上一板白鼠仍在投喂位置"}],
        },
    }

    prompt = _planned_board_prompt(
        "first_person",
        "b0002",
        2,
        10,
        [{"beat_id": "v01", "type": "dialogue", "speaker": "沈砚", "text": "这是营养餐。", "start": 0, "end": 3, "duration": 3}],
        3,
        assets,
        [{"text": "这是营养餐。"}],
        ["沈砚", "玄墨"],
        continuity,
    )

    assert "叙事连续性状态" in prompt
    assert "导演手法" in prompt
    assert "景别" in prompt
    assert "运镜" in prompt
    assert "白鼠" in prompt
    assert "营养餐" in prompt
    assert '"palette_id": "suspense_cold_blue"' in prompt
    assert "palette_id 不能为空" in prompt
    assert '"technique_id": ""' in prompt
    assert "每张 board 最多 1-2 个非空 technique_id" in prompt
    assert "普通交代镜头 technique_id 必须为空" in prompt
    assert "不能把状态里的既有道具替换成无证据的新物体" in prompt


def test_build_planned_board_preserves_prior_prop_when_llm_replaces_it_with_new_object():
    assets = {
        "characters": {"沈砚": {"seed": "年轻男性主角"}, "玄墨": {"seed": "黑色巨蟒"}},
        "scenes": {"工作室": {"seed": "工作室饲养箱环境"}},
        "props": {"白鼠": {"seed": "一只成年白色实验鼠，活蹦乱跳"}, "饲养箱": {"seed": "恒温玻璃箱"}},
    }
    continuity = {
        "state_context": {
            "active_props": [{"name": "白鼠", "state": "仍是当前投喂物", "aliases": ["营养餐", "食物"], "evidence": "上一板白鼠仍在投喂位置"}],
        }
    }
    voice = [{"beat_id": "v01", "start": 0, "end": 5, "duration": 5, "type": "dialogue", "speaker": "沈砚", "text": "这是我特意为你准备的营养餐。"}]
    shots = []
    for idx in range(5):
        shots.append({
            "shot_id": f"s{idx + 1:02d}",
            "start": idx,
            "end": idx + 1,
            "duration": 1,
            "voice_refs": ["v01"],
            "visual": "特写沈砚手中的营养餐，肉块脂肪分布均匀" if idx == 0 else "沈砚继续看着玄墨和投喂物",
            "camera": "特写",
            "characters": ["沈砚"],
            "scene": "工作室",
            "match_strategy": "sync",
            "purpose": "表现投喂物",
            "audio_behavior": "dialogue_sync",
            "continuity_from_previous": None if idx == 0 else "延续上一镜",
            "transition_type": None if idx == 0 else "cut",
        })
    board = _build_planned_board(
        plan_item={"board_id": "b0002"},
        page=2,
        total_pages=10,
        voice_timeline=voice,
        assets=assets,
        llm_result={
            "asset_refs": {"characters": ["沈砚", "玄墨"], "scene": "工作室", "props": ["饲养箱"]},
            "shot_timeline": shots,
        },
        source_trace=[{"text": "这是我特意为你准备的营养餐。"}],
        allowed_characters=["沈砚", "玄墨"],
        story_continuity=continuity,
    )

    assert "白鼠" in board["asset_refs"]["props"]
    assert "肉块" not in board["shot_timeline"][0]["visual"]
    assert "白鼠" in board["shot_timeline"][0]["visual"]
    assert any("连续性状态修正" in warning for warning in board["review"]["warnings"])


def test_build_planned_board_does_not_complete_action_when_source_is_incomplete():
    assets = {
        "characters": {"沈砚": {"seed": "年轻男性主角"}, "郑教授": {"seed": "退休教授"}},
        "scenes": {"工作室": {"seed": "工作室"}},
        "props": {},
    }
    continuity = {
        "state_context": {
            "open_actions": [{"action": "当前原文以承接标点结尾，动作或对白尚未结束", "evidence": "郑教授摆了摆手，"}],
            "incomplete_source": True,
        }
    }
    voice = [{"beat_id": "v01", "start": 0, "end": 3, "duration": 3, "type": "narration", "speaker": "沈砚", "text": "郑教授摆了摆手，"}]
    shots = []
    for idx in range(3):
        shots.append({
            "shot_id": f"s{idx + 1:02d}",
            "start": idx,
            "end": idx + 1,
            "duration": 1,
            "voice_refs": ["v01"],
            "visual": "郑教授摆了摆手，转身离开门口" if idx == 0 else "郑教授仍站在门口摆手",
            "camera": "中景",
            "characters": ["郑教授", "沈砚"],
            "scene": "工作室",
            "match_strategy": "sync",
            "purpose": "表现拒绝",
            "audio_behavior": "narration_sync",
            "continuity_from_previous": None if idx == 0 else "延续上一镜",
            "transition_type": None if idx == 0 else "cut",
        })
    board = _build_planned_board(
        plan_item={"board_id": "b0003"},
        page=3,
        total_pages=10,
        voice_timeline=voice,
        assets=assets,
        llm_result={
            "asset_refs": {"characters": ["沈砚", "郑教授"], "scene": "工作室", "props": []},
            "shot_timeline": shots,
        },
        source_trace=[{"text": "郑教授摆了摆手，"}],
        allowed_characters=["沈砚", "郑教授"],
        story_continuity=continuity,
    )

    assert "转身离开" not in board["shot_timeline"][0]["visual"]
    assert "没有补完离开动作" in board["shot_timeline"][0]["visual"]
    assert any("未完成原文" in warning for warning in board["review"]["warnings"])


def test_fallback_storyboard_shots_use_source_specific_visuals():
    assets = {
        "characters": {"沈砚": {"seed": "年轻男性主角"}, "玄墨": {"seed": "黑色巨蟒"}},
        "scenes": {"工作室": {"seed": "工作室饲养箱环境"}},
        "props": {},
    }
    voice = [{
        "beat_id": "v001",
        "start": 0,
        "end": 5,
        "duration": 5,
        "type": "narration",
        "speaker": "沈砚",
        "text": "我把白鼠凑到它嘴边，它却显得异常烦躁，猛地一甩头，迅速爬回了饲养箱。",
    }]
    board = _build_planned_board(
        plan_item={"board_id": "b0001"},
        page=1,
        total_pages=1,
        voice_timeline=voice,
        assets=assets,
        llm_result={"asset_refs": {"characters": ["沈砚", "玄墨"], "scene": "工作室", "props": []}, "shot_timeline": []},
        source_trace=[{"text": voice[0]["text"]}],
        allowed_characters=["沈砚", "玄墨"],
    )

    visuals = [shot["visual"] for shot in board["shot_timeline"]]
    assert all("角色在关键情节中反应" not in visual for visual in visuals)
    assert all("围绕原文事件行动" not in visual for visual in visuals)
    assert all("画面呈现原文事件" not in visual for visual in visuals)
    assert any("白鼠" in visual for visual in visuals)
    assert any("饲养箱" in visual for visual in visuals)


def test_fallback_shots_use_concrete_visuals_for_time_passage():
    assets = {
        "characters": {
            "沈砚": {"seed": "年轻男性主角"},
            "玄墨": {"seed": "黑色巨蟒"},
        },
        "scenes": {"工作室": {"seed": "工作室"}},
        "props": {},
    }
    voice = [{
        "beat_id": "v001",
        "start": 0,
        "end": 3,
        "duration": 3,
        "type": "narration",
        "speaker": "沈砚",
        "text": "这一养，就是整整七年。",
    }]
    board = _build_planned_board(
        plan_item={"board_id": "b0001"},
        page=1,
        total_pages=1,
        voice_timeline=voice,
        assets=assets,
        llm_result={"asset_refs": {"characters": ["沈砚", "玄墨"], "scene": "工作室", "props": []}, "shot_timeline": []},
        source_trace=[{"text": voice[0]["text"]}],
        allowed_characters=["沈砚", "玄墨"],
    )

    visuals = [shot["visual"] for shot in board["shot_timeline"]]
    joined = "\n".join(visuals)
    assert "围绕原文事件行动" not in joined
    assert "画面呈现原文事件" not in joined
    assert "七年" in joined or "长大" in joined
    assert "玄墨" in joined


def test_visible_identity_refs_do_not_mark_professor_as_snake_from_seed_context():
    assets = {
        "characters": {
            "玄墨": {"seed": "一条成年基因突变的缅甸蟒，全身纯黑色鳞片"},
            "郑教授": {"seed": "退休生物学教授，年长，戴眼镜，对蛇类保持距离"},
        },
        "scenes": {},
        "props": {},
    }
    board = {"asset_refs": {"characters": ["玄墨", "郑教授"], "scene": "", "props": []}}

    refs = _visible_identity_refs_for_board(board, assets)

    assert refs[0] == "图片1：玄墨，黑色巨蟒，故事中唯一的蛇"
    assert "郑教授，黑色巨蟒" not in "；".join(refs)
    assert any("郑教授，退休生物学教授" in item for item in refs)


def test_refresh_storyboard_prompt_context_adds_previous_final_panel():
    assets = {
        "characters": {
            "沈砚": {"seed": "年轻男性主角"},
            "玄墨": {"seed": "黑色巨蟒"},
        },
        "scenes": {"工作室": {"seed": "工作室饲养箱环境"}},
        "props": {},
    }
    board1 = {
        "page": 1,
        "board_duration": 5,
        "video_goal": "玄墨拒食",
        "voice_timeline": [],
        "shot_timeline": [{
            "shot_id": "s05",
            "start": 4,
            "end": 5,
            "camera": "中景",
            "visual": "沈砚站在饲养箱旁，手里仍拿着镊子，玄墨从他手臂滑下。",
            "characters": ["沈砚", "玄墨"],
            "audio_behavior": "ambient_only",
            "voice_refs": [],
        }],
        "asset_refs": {"characters": ["沈砚", "玄墨"], "scene": "工作室", "props": []},
        "storyboard_image": {"status": "needed", "prompt": ""},
        "video": {"status": "needed", "prompt": ""},
    }
    board2 = {
        "page": 2,
        "board_duration": 5,
        "video_goal": "玄墨盘踞凝视",
        "story_continuity": {
            "just_happened": "玄墨刚拒绝捕食。",
            "now_happening": "玄墨回到饲养箱中央后，一动不动地凝视沈砚。",
        },
        "voice_timeline": [],
        "shot_timeline": [{
            "shot_id": "s01",
            "start": 0,
            "end": 1,
            "camera": "近景",
            "visual": "玄墨滑入饲养箱中央。",
            "characters": ["玄墨"],
            "audio_behavior": "ambient_only",
            "voice_refs": [],
        }],
        "asset_refs": {"characters": ["玄墨"], "scene": "工作室", "props": []},
        "storyboard_image": {"status": "needed", "prompt": ""},
        "video": {"status": "needed", "prompt": ""},
    }

    _refresh_storyboard_prompt_context([board1, board2], assets)

    assert "沈砚站在饲养箱旁" in board2["story_continuity"]["previous_final_panel"]
    assert "上一张故事板最后一格：沈砚站在饲养箱旁" in board2["storyboard_image"]["prompt"]
    assert "接下来会发生什么" not in board2["storyboard_image"]["prompt"]


def test_refresh_storyboard_prompt_context_adds_spatial_rules_to_prompts():
    assets = {
        "characters": {"沈砚": {"seed": "青年男性"}, "玄墨": {"seed": "黑色巨蟒"}},
        "scenes": {"工作室": {"seed": "工作室"}},
        "props": {"饲养箱": {"seed": "玻璃饲养箱"}},
    }
    board1 = {
        "page": 1,
        "board_duration": 5,
        "video_goal": "沈砚观察玄墨",
        "voice_timeline": [],
        "shot_timeline": [{
            "shot_id": "s01",
            "start": 0,
            "end": 5,
            "camera": "中景",
            "visual": "沈砚站在画面左侧，玄墨在右侧饲养箱里",
            "characters": ["沈砚", "玄墨"],
            "audio_behavior": "ambient_only",
            "voice_refs": [],
        }],
        "asset_refs": {"characters": ["沈砚", "玄墨"], "scene": "工作室", "props": ["饲养箱"]},
        "storyboard_image": {"status": "needed", "prompt": ""},
        "video": {"status": "needed", "prompt": ""},
    }
    board2 = {
        "page": 2,
        "board_duration": 5,
        "video_goal": "玄墨继续异常",
        "voice_timeline": [],
        "shot_timeline": [{
            "shot_id": "s01",
            "start": 0,
            "end": 5,
            "camera": "近景",
            "visual": "玄墨仍在饲养箱里",
            "characters": ["玄墨"],
            "audio_behavior": "ambient_only",
            "voice_refs": [],
        }],
        "asset_refs": {"characters": ["玄墨"], "scene": "工作室", "props": ["饲养箱"]},
        "storyboard_image": {"status": "needed", "prompt": ""},
        "video": {"status": "needed", "prompt": ""},
    }

    _refresh_storyboard_prompt_context([board1, board2], assets)

    assert "空间方位" in board2["storyboard_image"]["prompt"]
    assert "沈砚站在画面左侧" in board2["storyboard_image"]["prompt"]
    assert "空间方位" in board2["video"]["prompt"]
    assert "不要左右互换" in board2["video"]["prompt"]


def test_submit_video_clamps_duration_to_seedance_range(monkeypatch):
    captured = {}

    class FakeResponse:
        is_success = True

        def json(self):
            return {"id": "video-task-1"}

    def fake_post(url, headers, json, timeout, **kwargs):
        captured["body"] = json
        return FakeResponse()

    monkeypatch.setattr("api.wetoken.httpx.post", fake_post)

    task_id = submit_video_task(
        api_key="wetoken-key",
        prompt="生成视频",
        image_paths=[],
        duration=3,
    )

    assert task_id == "video-task-1"
    assert captured["body"]["duration"] == 4


def test_asset_prompt_is_specific_to_asset_type():
    prompt = asset_prompt("characters", "玄墨", {"seed": "纯黑色成年缅甸蟒，暗紫金属光泽"})
    human_prompt = asset_prompt("characters", "郑教授", {"seed": "退休生物学教授，提着一袋杨梅，对蛇类保持距离"})

    assert "角色设定板" in prompt
    assert "玄墨" in prompt
    assert "纯黑色成年缅甸蟒" in prompt
    assert "四视图蛇类角色参考板" in prompt
    assert "同一条玄墨" in prompt
    assert "禁止画成蜥蜴" in prompt
    assert "不要人物手部" in prompt
    assert "四视图真人角色参考板" in human_prompt
    assert "四视图蛇类角色参考板" not in human_prompt
    assert "手里不要拿任何东西" in human_prompt
    assert "不要道具" in human_prompt
    assert "不要文字" in prompt


def test_submit_asset_force_regenerates_completed_asset_with_custom_prompt(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    old_image = project / "characters" / "医生.jpg"
    old_image.parent.mkdir()
    old_image.write_bytes(b"old")
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","narration_style":"third_person","source_text":"故事",'
        '"assets":{"characters":{"医生":{"seed":"医生","status":"completed",'
        '"task_id":"old-task","prompt":"旧提示词","local_path":"' + str(old_image) + '"}},'
        '"scenes":{},"props":{}},"narration_segments":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda name: "vidu-key")
    captured = {}

    def fake_submit_image_task(api_key, prompt, image_paths, ratio):
        captured.update({"api_key": api_key, "prompt": prompt, "image_paths": image_paths, "ratio": ratio})
        return {"task_id": "new-task"}

    import api.vidu
    monkeypatch.setattr(api.vidu, "submit_image_task", fake_submit_image_task)

    import asyncio
    import json

    result = asyncio.run(submit_asset(SubmitAssetRequest(
        project_name="大蟒蛇",
        asset_type="characters",
        asset_name="医生",
        prompt="新医生提示词",
        force=True,
    )))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    asset = saved["assets"]["characters"]["医生"]

    assert result["task_id"] == "new-task"
    assert captured["prompt"] == "新医生提示词"
    assert captured["ratio"] == "3:4"
    assert asset["status"] == "submitted"
    assert asset["task_id"] == "new-task"
    assert asset["prompt"] == "新医生提示词"
    assert asset["local_path"] is None
    assert asset["previous_local_path"] == str(old_image)


def test_submit_storyboard_force_uses_custom_prompt_and_clears_old_image(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    old_image = project / "storyboards" / "seg_01_p01.jpg"
    ref_image = project / "characters" / "玄墨.jpg"
    old_image.parent.mkdir(parents=True)
    ref_image.parent.mkdir(parents=True)
    old_image.write_bytes(b"old storyboard")
    ref_image.write_bytes(b"ref")
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(ref_image) + '","seed":"黑色巨蟒"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        '"storyboard_image":{"status":"completed","task_id":"old-task","prompt":"旧分镜提示词","local_path":"' + str(old_image) + '"},'
        '"video":{"status":"needed"}}]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "vidu-key")
    captured = {}

    def fake_submit_image_task(api_key, prompt, image_paths, ratio):
        captured.update({"prompt": prompt, "image_paths": image_paths, "ratio": ratio})
        return {"task_id": "new-storyboard-task"}

    import api.vidu
    monkeypatch.setattr(api.vidu, "submit_image_task", fake_submit_image_task)

    import asyncio
    import json

    result = asyncio.run(submit_storyboard(SubmitStoryboardRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
        prompt="新分镜提示词",
        force=True,
    )))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    storyboard = saved["narration_segments"]["seg_01"]["boards"][0]["storyboard_image"]

    assert result["task_id"] == "new-storyboard-task"
    assert captured["prompt"].startswith("新分镜提示词")
    assert storyboard["status"] == "submitted"
    assert storyboard["prompt"] == "新分镜提示词"
    assert storyboard["local_path"] is None
    assert storyboard["previous_local_path"] == str(old_image)
    assert storyboard["history"][-1]["event"] == "submit_storyboard"
    assert storyboard["history"][-1]["prompt"] == "新分镜提示词"


def test_submit_storyboard_force_marks_completed_video_stale(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    old_storyboard = project / "storyboards" / "seg_01_p01.jpg"
    old_video = project / "videos" / "old.mp4"
    ref_image = project / "characters" / "玄墨.jpg"
    old_storyboard.parent.mkdir(parents=True)
    old_video.parent.mkdir(parents=True)
    ref_image.parent.mkdir(parents=True)
    old_storyboard.write_bytes(b"old storyboard")
    old_video.write_bytes(b"old video")
    ref_image.write_bytes(b"ref")
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(ref_image) + '","seed":"黑色巨蟒"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        '"storyboard_image":{"status":"completed","task_id":"old-storyboard","prompt":"旧分镜提示词","local_path":"' + str(old_storyboard) + '"},'
        '"video":{"status":"completed","task_id":"old-video-task","prompt":"旧视频提示词","local_path":"' + str(old_video) + '"}}]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "vidu-key")
    monkeypatch.setattr("api.vidu.submit_image_task", lambda **kwargs: {"task_id": "new-storyboard-task"})

    import asyncio
    import json

    asyncio.run(submit_storyboard(SubmitStoryboardRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
        prompt="新分镜提示词",
        force=True,
    )))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    video = saved["narration_segments"]["seg_01"]["boards"][0]["video"]

    assert video["status"] == "needed"
    assert video["task_id"] is None
    assert video["local_path"] is None
    assert video["previous_task_id"] == "old-video-task"
    assert video["previous_local_path"] == str(old_video)
    assert video["history"][-1]["event"] == "mark_stale"
    assert video["history"][-1]["reason"] == "故事板已重新生成，需要重新生成视频"


def test_sync_existing_outputs_does_not_complete_submitted_asset_from_old_file(tmp_path):
    project = tmp_path / "大蟒蛇"
    old_image = project / "characters" / "医生.jpg"
    old_image.parent.mkdir(parents=True)
    old_image.write_bytes(b"old")
    data = {
        "assets": {
            "characters": {
                "医生": {
                    "status": "submitted",
                    "task_id": "new-task",
                    "local_path": str(old_image),
                }
            },
            "scenes": {},
            "props": {},
        },
        "narration_segments": {},
    }

    changed = poller._sync_existing_outputs(project, data)

    assert changed is False
    assert data["assets"]["characters"]["医生"]["status"] == "submitted"


def test_sync_existing_outputs_does_not_complete_submitted_storyboard_from_old_file(tmp_path):
    project = tmp_path / "大蟒蛇"
    old_image = project / "storyboards" / "seg_01_p01.jpg"
    old_image.parent.mkdir(parents=True)
    old_image.write_bytes(b"old")
    data = {
        "assets": {"characters": {}, "scenes": {}, "props": {}},
        "narration_segments": {
            "seg_01": {
                "boards": [
                    {
                        "storyboard_image": {
                            "status": "submitted",
                            "task_id": "new-task",
                            "local_path": None,
                        }
                    }
                ]
            }
        },
    }

    changed = poller._sync_existing_outputs(project, data)

    storyboard = data["narration_segments"]["seg_01"]["boards"][0]["storyboard_image"]
    assert changed is False
    assert storyboard["status"] == "submitted"
    assert storyboard["local_path"] is None


def test_poller_storyboard_completion_preserves_concurrent_video_submission(tmp_path, monkeypatch):
    import asyncio
    import json

    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        json.dumps(
            {
                "project": "大蟒蛇",
                "assets": {"characters": {}, "scenes": {}, "props": {}},
                "narration_segments": {
                    "seg_01": {
                        "boards": [
                            {
                                "page": 1,
                                "storyboard_image": {
                                    "status": "submitted",
                                    "task_id": "storyboard-task",
                                    "prompt": "画分镜",
                                },
                                "video": {
                                    "status": "needed",
                                    "prompt": "做视频",
                                    "task_id": None,
                                },
                            }
                        ]
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.poller.get_api_key", lambda key: "vidu-key" if key == "VIDU_API_KEY" else "")

    async def fake_poll_storyboard(api_key, task_id, dest_path=None):
        current = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
        video = current["narration_segments"]["seg_01"]["boards"][0]["video"]
        video["status"] = "submitted"
        video["task_id"] = "video-task"
        video["prompt"] = "做视频"
        (project / "pipeline.json").write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        return {
            "status": "completed",
            "url": "https://example.com/storyboard.jpg",
            "local_path": str(project / "storyboards" / "seg_01_p01.jpg"),
        }

    monkeypatch.setattr(poller, "_poll_storyboard", fake_poll_storyboard)

    asyncio.run(poller._poll_once())

    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    board = saved["narration_segments"]["seg_01"]["boards"][0]
    assert board["storyboard_image"]["status"] == "completed"
    assert board["video"]["status"] == "submitted"
    assert board["video"]["task_id"] == "video-task"


def test_poller_sync_existing_outputs_preserves_concurrent_video_submission(tmp_path, monkeypatch):
    import asyncio
    import json

    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    old_storyboard = project / "storyboards" / "seg_01_p01.jpg"
    old_storyboard.parent.mkdir(parents=True)
    old_storyboard.write_bytes(b"old storyboard")
    (project / "pipeline.json").write_text(
        json.dumps(
            {
                "project": "大蟒蛇",
                "assets": {"characters": {}, "scenes": {}, "props": {}},
                "narration_segments": {
                    "seg_01": {
                        "boards": [
                            {
                                "page": 1,
                                "storyboard_image": {
                                    "status": "needed",
                                    "prompt": "画分镜",
                                    "local_path": None,
                                },
                                "video": {
                                    "status": "needed",
                                    "prompt": "做视频",
                                    "task_id": None,
                                },
                            }
                        ]
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.poller.get_api_key", lambda key: "")

    original_sync = poller._sync_existing_outputs

    def sync_with_concurrent_submit(project_dir, data):
        changed = original_sync(project_dir, data)
        current = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
        video = current["narration_segments"]["seg_01"]["boards"][0]["video"]
        video["status"] = "submitted"
        video["task_id"] = "video-task"
        (project / "pipeline.json").write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        return changed

    monkeypatch.setattr(poller, "_sync_existing_outputs", sync_with_concurrent_submit)

    asyncio.run(poller._poll_once())

    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    board = saved["narration_segments"]["seg_01"]["boards"][0]
    assert board["storyboard_image"]["status"] == "completed"
    assert board["video"]["status"] == "submitted"
    assert board["video"]["task_id"] == "video-task"


def test_normalize_video_output_paths_moves_completed_legacy_task_file(tmp_path):
    project = tmp_path / "大蟒蛇"
    legacy = project / "videos" / "cgt-task-1.mp4"
    expected = project / "videos" / "p001_seg_01_b0001.mp4"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"video")
    data = {
        "narration_segments": {
            "seg_01": {
                "boards": [
                    {
                        "page": 1,
                        "board_id": "b0001",
                        "video": {
                            "status": "completed",
                            "local_path": str(legacy),
                            "output_path": str(expected),
                        },
                    }
                ]
            }
        }
    }

    changed = normalize_video_output_paths(project, data)

    video = data["narration_segments"]["seg_01"]["boards"][0]["video"]
    assert changed is True
    assert video["local_path"] == str(expected)
    assert expected.read_bytes() == b"video"
    assert not legacy.exists()


def test_sync_existing_outputs_normalizes_completed_video_output_path(tmp_path):
    project = tmp_path / "大蟒蛇"
    legacy = project / "videos" / "cgt-task-1.mp4"
    expected = project / "videos" / "p001_seg_01_b0001.mp4"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"video")
    data = {
        "assets": {"characters": {}, "scenes": {}, "props": {}},
        "narration_segments": {
            "seg_01": {
                "boards": [
                    {
                        "page": 1,
                        "board_id": "b0001",
                        "storyboard_image": {"status": "completed"},
                        "video": {
                            "status": "completed",
                            "local_path": str(legacy),
                            "output_path": str(expected),
                        },
                    }
                ]
            }
        },
    }

    changed = poller._sync_existing_outputs(project, data)

    video = data["narration_segments"]["seg_01"]["boards"][0]["video"]
    assert changed is True
    assert video["local_path"] == str(expected)
    assert expected.exists()


def test_reference_collection_uses_full_assets_for_storyboards_and_light_refs_for_videos(tmp_path):
    local_storyboard = tmp_path / "storyboard.jpg"
    local_storyboard.write_bytes(b"image")
    data = {
        "assets": {
            "characters": {
                "玄墨": {"local_path": str(tmp_path / "xuanmo.jpg")},
                "沈砚": {"url": "https://example.test/shenyan.jpg"},
            },
            "scenes": {"Loft工作室": {"url": "https://example.test/loft.jpg"}},
            "props": {"铜锁": {"local_path": str(tmp_path / "lock.jpg")}},
        }
    }
    board = {
        "storyboard_image": {"local_path": str(local_storyboard)},
        "asset_refs": {
            "characters": ["玄墨", "沈砚"],
            "scene": "Loft工作室",
            "props": ["铜锁"],
        },
    }

    storyboard_refs = collect_asset_reference_images(data, board)
    video_refs = collect_video_reference_images(data, board)

    assert storyboard_refs == [
        str(tmp_path / "xuanmo.jpg"),
        "https://example.test/shenyan.jpg",
        "https://example.test/loft.jpg",
        str(tmp_path / "lock.jpg"),
    ]
    assert video_refs == [
        str(local_storyboard),
        str(tmp_path / "xuanmo.jpg"),
        "https://example.test/shenyan.jpg",
    ]


def test_snake_character_reference_is_prioritized_for_storyboards(tmp_path):
    data = {
        "assets": {
            "characters": {
                "沈砚": {"local_path": str(tmp_path / "shenyan.jpg"), "seed": "青年男性"},
                "玄墨": {"local_path": str(tmp_path / "xuanmo.jpg"), "seed": "黑色巨蟒"},
            },
            "scenes": {},
            "props": {},
        }
    }
    board = {
        "asset_refs": {
            "characters": ["沈砚", "玄墨"],
            "scene": "",
            "props": [],
        },
    }

    assert collect_asset_reference_images(data, board) == [
        str(tmp_path / "xuanmo.jpg"),
        str(tmp_path / "shenyan.jpg"),
    ]


def test_build_segments_outline_prefers_semantic_episode_boundaries():
    script_plan = {
        "voice_beats": [
            {
                "beat_id": f"v{i:04d}",
                "text": "玄墨夜里爬到床边，沈砚以为它只是生病。",
                "duration": 15,
            }
            for i in range(1, 11)
        ] + [
            {
                "beat_id": "v0011",
                "text": "秦越看完视频后突然警告：它不是撒娇，它是在量你的尺寸！",
                "duration": 15,
            },
            {
                "beat_id": "v0012",
                "text": "第二天沈砚带着玄墨赶往宠物医院。",
                "duration": 15,
            },
        ]
    }
    board_plan = [
        {"duration": 15, "voice_beat_ids": [f"v{i:04d}"]}
        for i in range(1, 13)
    ]

    outline = _build_segments_outline(
        board_plan,
        script_plan=script_plan,
        episode_min_seconds=150,
        episode_target_seconds=180,
        episode_max_seconds=210,
        segment_boards=5,
    )

    assert list(outline) == ["seg_1_1", "seg_1_2", "seg_1_3", "seg_2_1"]
    assert outline["seg_1_3"]["board_range"] == [11, 11]
    assert outline["seg_1_3"]["episode_seconds"] == 165
    assert "语义边界" in outline["seg_1_3"]["episode_boundary_reason"]
    assert outline["seg_2_1"]["episode"] == 2
    assert outline["seg_2_1"]["board_range"] == [12, 12]


def test_build_segments_outline_does_not_split_incomplete_sentence_boundary():
    script_plan = {
        "voice_beats": [
            {"beat_id": f"v{i:04d}", "text": "玄墨安静趴在加热石上。", "duration": 15}
            for i in range(1, 11)
        ] + [
            {"beat_id": "v0011", "text": "谁也没料到，这条长虫", "duration": 15},
            {"beat_id": "v0012", "text": "，最近染上了一个让我无法理解的新怪癖。", "duration": 15},
            {"beat_id": "v0013", "text": "那天深夜，我听见床边传来摩擦声。", "duration": 15},
        ]
    }
    board_plan = [
        {"duration": 15, "voice_beat_ids": [f"v{i:04d}"]}
        for i in range(1, 14)
    ]

    outline = _build_segments_outline(
        board_plan,
        script_plan=script_plan,
        episode_min_seconds=150,
        episode_target_seconds=165,
        episode_max_seconds=170,
        segment_boards=20,
    )

    episodes = list(outline.values())
    assert episodes[0]["episode_board_range"] == [1, 10]
    assert episodes[1]["episode_board_range"] == [11, 13]


def test_ensure_referenced_assets_adds_missing_board_refs():
    data = {
        "assets": {"characters": {}, "scenes": {}, "props": {}},
        "narration_segments": {
            "seg_1_1": {"boards": [{
                "asset_refs": {
                    "characters": ["医生"],
                    "scene": "医院",
                    "props": ["手机"],
                }
            }]}
        },
    }

    ensure_referenced_assets(data)

    assert data["assets"]["characters"]["医生"]["status"] == "needed"
    assert data["assets"]["scenes"]["医院"]["status"] == "needed"
    assert data["assets"]["props"]["手机"]["status"] == "needed"


def test_sync_board_metadata_assigns_stable_global_and_segment_labels():
    data = {
        "narration_segments": {
            "seg_1_2": {"boards": [{"page": 3, "board_id": "b0003"}]},
            "seg_1_1": {"boards": [{"page": 1, "board_id": "b0001"}, {"page": 2, "board_id": "b0002"}]},
        }
    }

    sync_board_metadata(data)

    first = data["narration_segments"]["seg_1_1"]["boards"][0]
    second_segment = data["narration_segments"]["seg_1_2"]["boards"][0]
    assert first["display_id"] == "P001"
    assert first["segment_display_id"] == "seg_1_1 P01"
    assert second_segment["display_id"] == "P003"
    assert second_segment["segment_board_index"] == 1
    assert second_segment["total_pages"] == 3


def test_apply_default_visual_macros_backfills_old_board_style_fields():
    data = {
        "narration_segments": {
            "seg_1_7": {
                "boards": [{
                    "video_goal": "深夜床沿出现异常，沈砚被奇异触感惊醒",
                    "asset_refs": {"characters": ["沈砚", "玄墨"], "scene": "卧室", "props": []},
                    "shot_timeline": [
                        {
                            "shot_id": "s01",
                            "visual": "低角度看向床沿，一道黑色蛇身阴影缓慢靠近",
                            "camera": "近景",
                            "purpose": "表现危险逼近",
                        },
                        {
                            "shot_id": "s02",
                            "visual": "沈砚猛然抬头，眼睛瞪大",
                            "camera": "中景",
                            "purpose": "表现惊醒瞬间",
                        },
                        {
                            "shot_id": "s03",
                            "visual": "沈砚低头看向床沿",
                            "camera": "主观视角",
                            "purpose": "引导观众注意黑暗中的物体",
                        },
                    ],
                }, {
                    "video_goal": "展示实木地板上的滑行摩擦",
                    "asset_refs": {"characters": ["玄墨"], "scene": "卧室", "props": []},
                    "shot_timeline": [
                        {
                            "shot_id": "s01",
                            "visual": "实木地板特写，细腻木纹",
                            "camera": "特写",
                            "purpose": "建立环境质感",
                        },
                    ],
                }]
            }
        }
    }

    assert apply_default_visual_macros(data) is True

    board = data["narration_segments"]["seg_1_7"]["boards"][0]
    assert board["palette_id"] == "suspense_cold_blue"
    same_segment_board = data["narration_segments"]["seg_1_7"]["boards"][1]
    assert same_segment_board["palette_id"] == "suspense_cold_blue"
    techniques = [shot["technique_id"] for shot in board["shot_timeline"]]
    assert techniques == ["slow_push_in", "crash_zoom", ""]


def test_apply_default_visual_macros_preserves_existing_style_fields():
    data = {
        "narration_segments": {
            "seg_01": {
                "boards": [{
                    "palette_id": "family_warm_gray",
                    "video_goal": "深夜异常",
                    "shot_timeline": [
                        {"visual": "突然冲出", "camera": "特写", "purpose": "爆点", "technique_id": "pov_shot"},
                        {"visual": "缓慢靠近", "camera": "近景", "purpose": "压迫", "technique_id": ""},
                    ],
                }]
            }
        }
    }

    apply_default_visual_macros(data)

    board = data["narration_segments"]["seg_01"]["boards"][0]
    assert board["palette_id"] == "family_warm_gray"
    assert board["shot_timeline"][0]["technique_id"] == "pov_shot"
    assert board["shot_timeline"][1]["technique_id"] == "slow_push_in"


def test_storyboard_submit_requires_all_asset_reference_images(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事","assets":{"characters":{"玄墨":{"status":"needed"}},"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"]},'
        '"storyboard_image":{"status":"needed","prompt":"画分镜"},"video":{"status":"needed"}}]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "vidu-key")

    import asyncio
    from fastapi import HTTPException

    try:
        asyncio.run(submit_storyboard(SubmitStoryboardRequest(
            project_name="大蟒蛇",
            segment_key="seg_01",
            board_index=0,
        )))
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail["error_type"] == "missing_reference_images"
        assert exc.detail["missing_asset_refs"] == ["角色:玄墨"]
    else:
        raise AssertionError("submit_storyboard should reject missing asset refs")


def test_storyboard_submit_rejects_invalid_prompt_before_vidu(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    ref = project / "characters" / "玄墨.jpg"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"ref")
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(ref) + '","seed":"黑色巨蟒"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"]},'
        '"voice_timeline":[{"beat_id":"v01","text":"玄墨拒食。","type":"narration"}],'
        '"review":{"source_excerpt":"玄墨拒食。"},'
        '"storyboard_image":{"status":"needed","prompt":"当前板5格：\\n角色在关键情节中反应\\n玄墨围绕原文事件行动：拒食。"},'
        '"video":{"status":"needed"}}]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "vidu-key")

    import asyncio
    from fastapi import HTTPException

    try:
        asyncio.run(submit_storyboard(SubmitStoryboardRequest(
            project_name="大蟒蛇",
            segment_key="seg_01",
            board_index=0,
        )))
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail["error_type"] == "invalid_storyboard_prompt"
        joined_errors = "\n".join(exc.detail["errors"])
        assert "角色在关键情节中反应" in joined_errors
        assert "围绕原文事件行动" in joined_errors
    else:
        raise AssertionError("submit_storyboard should reject invalid prompt before Vidu")


def test_storyboard_submit_ignores_board_palette_reference(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    ref = project / "characters" / "玄墨.jpg"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"ref")
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(ref) + '","seed":"黑色巨蟒"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"]},'
        '"palette_id":"suspense_cold_blue",'
        '"review":{"source_excerpt":"玄墨拒食。"},'
        '"storyboard_image":{"status":"needed","prompt":"玄墨在饲养箱中拒食，主人站在旁边观察。"},'
        '"video":{"status":"needed"}}]}}}',
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "vidu-key")
    monkeypatch.setattr("api.vidu.submit_image_task", lambda **kwargs: captured.update(kwargs) or {"task_id": "storyboard-task"})

    import asyncio

    result = asyncio.run(submit_storyboard(SubmitStoryboardRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
    )))

    assert result["task_id"] == "storyboard-task"
    assert captured["image_paths"] == [str(ref)]
    assert "色板参考图" not in captured["prompt"]
    assert not (project / "palettes" / "suspense_cold_blue_palette.png").exists()


def test_submit_storyboards_uses_configured_concurrency(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    xuanmo = project / "characters" / "玄墨.jpg"
    xuanmo.parent.mkdir()
    xuanmo.write_bytes(b"image")
    boards = ",".join(
        '{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        f'"storyboard_image":{{"status":"needed","prompt":"画分镜{i}"}},'
        '"video":{"status":"needed"}}'
        for i in range(3)
    )
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(xuanmo) + '","seed":"黑色巨蟒"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[' + boards + ']}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "vidu-key")
    monkeypatch.setenv("STORYBOARD_SUBMIT_CONCURRENCY", "2")

    import asyncio
    import json
    import threading
    import time

    lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "calls": 0, "prompts": []}

    def fake_submit_image_task(api_key, prompt, image_paths, ratio):
        with lock:
            state["active"] += 1
            state["calls"] += 1
            state["prompts"].append(prompt)
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return {"task_id": f"task-{state['calls']}"}

    import api.vidu
    monkeypatch.setattr(api.vidu, "submit_image_task", fake_submit_image_task)

    result = asyncio.run(submit_storyboards(ProjectRequest(project_name="大蟒蛇")))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))

    assert len(result["submitted"]) == 3
    assert state["max_active"] == 2
    assert all("参考图1：角色参考图：玄墨" in prompt for prompt in state["prompts"])
    assert all("禁止画成蜥蜴" in prompt for prompt in state["prompts"])
    assert all(
        board["storyboard_image"]["status"] == "submitted"
        for board in saved["narration_segments"]["seg_01"]["boards"]
    )
    assert all(
        board["storyboard_image"]["reference_image_labels"][0]["asset_name"] == "玄墨"
        for board in saved["narration_segments"]["seg_01"]["boards"]
    )


def test_submit_storyboards_records_individual_failure_without_aborting_batch(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    xuanmo = project / "characters" / "玄墨.jpg"
    project.mkdir(parents=True)
    xuanmo.parent.mkdir()
    xuanmo.write_bytes(b"image")
    boards = ",".join(
        '{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        f'"storyboard_image":{{"status":"needed","prompt":"画分镜{i}"}},'
        '"video":{"status":"needed"}}'
        for i in range(2)
    )
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(xuanmo) + '","seed":"黑色巨蟒"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[' + boards + ']}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "vidu-key")
    calls = {"count": 0}

    def fake_submit_image_task(api_key, prompt, image_paths, ratio):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("vidu timeout")
        return {"task_id": "task-ok"}

    monkeypatch.setattr("api.vidu.submit_image_task", fake_submit_image_task)

    import asyncio
    import json

    result = asyncio.run(submit_storyboards(ProjectRequest(project_name="大蟒蛇")))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    boards_saved = saved["narration_segments"]["seg_01"]["boards"]

    assert len(result["submitted"]) == 1
    assert len(result["skipped"]) == 1
    assert result["submitted"][0]["duration_ms"] >= 0
    assert boards_saved[0]["storyboard_image"]["status"] == "submitted"
    assert boards_saved[1]["storyboard_image"]["status"] == "failed"
    assert "vidu timeout" in boards_saved[1]["storyboard_image"]["error"]
    assert boards_saved[0]["storyboard_image"]["history"][-1]["duration_ms"] >= 0
    assert boards_saved[1]["storyboard_image"]["history"][-1]["duration_ms"] >= 0


def test_video_submit_requires_storyboard_and_asset_reference_images(tmp_path, monkeypatch):
    storyboard = tmp_path / "storyboard.jpg"
    storyboard.write_bytes(b"image")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事","assets":{"characters":{"玄墨":{"status":"needed"}},"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"]},'
        '"page":1,"total_pages":1,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
        '"video_goal":"测试视频",'
        '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
        '"shot_timeline":['
        '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
        '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
        '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
        '"video":{"status":"needed","prompt":"做视频"}}]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")

    import asyncio
    from fastapi import HTTPException

    try:
        asyncio.run(submit_video(SubmitVideoRequest(
            project_name="大蟒蛇",
            segment_key="seg_01",
            board_index=0,
        )))
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail["error_type"] == "missing_reference_images"
        assert exc.detail["missing_asset_refs"] == ["角色:玄墨"]
    else:
        raise AssertionError("submit_video should reject missing asset refs")


def test_video_submit_only_requires_character_reference_images(tmp_path, monkeypatch):
    storyboard = tmp_path / "storyboard.jpg"
    character = tmp_path / "xuanmo.jpg"
    storyboard.write_bytes(b"image")
    character.write_bytes(b"image")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{"工作室":{"status":"needed"}},'
        '"props":{"饲养箱":{"status":"needed"}}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"工作室","props":["饲养箱"]},'
        '"page":1,"total_pages":1,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
        '"video_goal":"测试视频",'
        '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
        '"shot_timeline":['
        '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"工作室","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
        '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"工作室","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"工作室","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"工作室","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"工作室","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
        '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
        '"video":{"status":"needed","prompt":"做视频"}}]}}}',
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")
    monkeypatch.setattr("api.wetoken.submit_video_task", lambda **kwargs: captured.update(kwargs) or "task-1")

    import asyncio
    import json

    result = asyncio.run(submit_video(SubmitVideoRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
    )))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    refs = saved["narration_segments"]["seg_01"]["boards"][0]["video"]["reference_images"]

    assert result["ok"] is True
    assert captured["image_paths"][:2] == [str(storyboard), str(character)]
    assert captured["image_paths"][-1].endswith("suspense_cold_blue_palette.png")
    assert refs == captured["image_paths"]


def test_video_submit_appends_palette_reference_image_when_board_has_palette(tmp_path, monkeypatch):
    storyboard = tmp_path / "storyboard.jpg"
    character = tmp_path / "xuanmo.jpg"
    storyboard.write_bytes(b"image")
    character.write_bytes(b"image")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        '"palette_id":"suspense_cold_blue",'
        '"page":1,"total_pages":1,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
        '"video_goal":"测试视频",'
        '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
        '"shot_timeline":['
        '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
        '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
        '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
        '"video":{"status":"needed","prompt":"做视频"}}]}}}',
        encoding="utf-8",
    )
    captured = {}
    palette_dir = tmp_path / "static" / "palettes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.color_palettes.STATIC_PALETTE_DIR", palette_dir)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")
    monkeypatch.setattr("api.wetoken.submit_video_task", lambda **kwargs: captured.update(kwargs) or "task-1")

    import asyncio
    import json

    result = asyncio.run(submit_video(SubmitVideoRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
    )))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    refs = saved["narration_segments"]["seg_01"]["boards"][0]["video"]["reference_images"]
    palette_ref = palette_dir / "suspense_cold_blue_palette.png"

    assert result["ok"] is True
    assert captured["image_paths"] == [str(storyboard), str(character), str(palette_ref)]
    assert refs == [str(storyboard), str(character), str(palette_ref)]
    assert palette_ref.exists()
    assert palette_ref.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not (project / "palettes" / "suspense_cold_blue_palette.png").exists()
    assert "最后一张参考图为色板参考图" in captured["prompt"]
    assert "整体色调为冷青灰" in captured["prompt"]


def test_video_submit_appends_palette_note_to_custom_prompt(tmp_path, monkeypatch):
    storyboard = tmp_path / "storyboard.jpg"
    character = tmp_path / "xuanmo.jpg"
    storyboard.write_bytes(b"image")
    character.write_bytes(b"image")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        '"palette_id":"suspense_cold_blue",'
        '"page":1,"total_pages":1,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
        '"video_goal":"测试视频",'
        '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
        '"shot_timeline":['
        '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
        '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
        '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
        '"video":{"status":"needed","prompt":"旧视频提示词"}}]}}}',
        encoding="utf-8",
    )
    captured = {}
    palette_dir = tmp_path / "static" / "palettes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.color_palettes.STATIC_PALETTE_DIR", palette_dir)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")
    monkeypatch.setattr("api.wetoken.submit_video_task", lambda **kwargs: captured.update(kwargs) or "task-1")

    import asyncio

    asyncio.run(submit_video(SubmitVideoRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
        prompt="自定义视频提示词",
    )))

    assert captured["prompt"].startswith("自定义视频提示词")
    assert "最后一张参考图为色板参考图" in captured["prompt"]
    assert "整体色调为冷青灰" in captured["prompt"]


def test_video_submit_uses_distinct_static_palette_references_for_two_palettes(tmp_path, monkeypatch):
    storyboard1 = tmp_path / "storyboard1.jpg"
    storyboard2 = tmp_path / "storyboard2.jpg"
    character = tmp_path / "xuanmo.jpg"
    for path in (storyboard1, storyboard2, character):
        path.write_bytes(b"image")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)

    def board_json(storyboard, palette_id):
        return (
            '{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
            f'"palette_id":"{palette_id}",'
            '"page":1,"total_pages":2,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
            '"video_goal":"测试视频",'
            '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
            '"shot_timeline":['
            '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
            '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
            '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
            '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
            '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
            '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
            '"video":{"status":"needed","prompt":"做视频"}}'
        )

    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":['
        + board_json(storyboard1, "family_warm_gray")
        + ","
        + board_json(storyboard2, "hospital_cold_white")
        + "]}}}",
        encoding="utf-8",
    )
    calls = []
    palette_dir = tmp_path / "static" / "palettes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.color_palettes.STATIC_PALETTE_DIR", palette_dir)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")

    def fake_submit_video_task(**kwargs):
        calls.append({**kwargs, "image_paths": list(kwargs["image_paths"])})
        return f"task-{len(calls)}"

    monkeypatch.setattr("api.wetoken.submit_video_task", fake_submit_video_task)

    import asyncio
    import json

    asyncio.run(submit_video(SubmitVideoRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
    )))
    asyncio.run(submit_video(SubmitVideoRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=1,
    )))

    family_palette = palette_dir / "family_warm_gray_palette.png"
    hospital_palette = palette_dir / "hospital_cold_white_palette.png"
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    boards = saved["narration_segments"]["seg_01"]["boards"]

    assert calls[0]["image_paths"] == [str(storyboard1), str(character), str(family_palette)]
    assert calls[1]["image_paths"] == [str(storyboard2), str(character), str(hospital_palette)]
    assert "整体色调为暖灰米棕" in calls[0]["prompt"]
    assert "整体色调为冷白灰蓝" in calls[1]["prompt"]
    assert boards[0]["video"]["reference_images"][-1] == str(family_palette)
    assert boards[1]["video"]["reference_images"][-1] == str(hospital_palette)
    assert family_palette.exists()
    assert hospital_palette.exists()
    assert not (project / "palettes").exists()


def test_submit_video_preserves_pipeline_changes_written_during_external_submit(tmp_path, monkeypatch):
    storyboard = tmp_path / "storyboard.jpg"
    character = tmp_path / "xuanmo.jpg"
    storyboard.write_bytes(b"image")
    character.write_bytes(b"image")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        '"page":1,"total_pages":1,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
        '"video_goal":"测试视频",'
        '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
        '"shot_timeline":['
        '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
        '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
        '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
        '"video":{"status":"needed","prompt":"做视频"}}]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")

    def fake_submit_video_task(**kwargs):
        import json

        current = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
        current["concurrent_marker"] = "keep-me"
        (project / "pipeline.json").write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        return "video-task-1"

    monkeypatch.setattr("api.wetoken.submit_video_task", fake_submit_video_task)

    import asyncio
    import json

    result = asyncio.run(submit_video(SubmitVideoRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
    )))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    video = saved["narration_segments"]["seg_01"]["boards"][0]["video"]

    assert result["task_id"] == "video-task-1"
    assert saved["concurrent_marker"] == "keep-me"
    assert video["status"] == "submitted"
    assert video["task_id"] == "video-task-1"


def test_submit_videos_records_individual_failure_without_aborting_batch(tmp_path, monkeypatch):
    storyboard1 = tmp_path / "storyboard1.jpg"
    storyboard2 = tmp_path / "storyboard2.jpg"
    character = tmp_path / "xuanmo.jpg"
    for path in (storyboard1, storyboard2, character):
        path.write_bytes(b"image")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)

    def board_json(storyboard):
        return (
            '{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
            '"page":1,"total_pages":2,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
            '"video_goal":"测试视频",'
            '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
            '"shot_timeline":['
            '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
            '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
            '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
            '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
            '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
            '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
            '"video":{"status":"needed","prompt":"做视频"}}'
        )

    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[' + board_json(storyboard1) + "," + board_json(storyboard2) + ']}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")
    calls = {"count": 0}

    def fake_submit_video_task(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("seedance timeout")
        return "video-task-ok"

    monkeypatch.setattr("api.wetoken.submit_video_task", fake_submit_video_task)

    import asyncio
    import json

    result = asyncio.run(submit_videos(ProjectRequest(project_name="大蟒蛇")))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    boards_saved = saved["narration_segments"]["seg_01"]["boards"]

    assert len(result["submitted"]) == 1
    assert len(result["skipped"]) == 1
    assert result["submitted"][0]["duration_ms"] >= 0
    assert boards_saved[0]["video"]["status"] == "submitted"
    assert boards_saved[1]["video"]["status"] == "failed"
    assert "seedance timeout" in boards_saved[1]["video"]["error"]
    assert boards_saved[0]["video"]["history"][-1]["duration_ms"] >= 0
    assert boards_saved[1]["video"]["history"][-1]["duration_ms"] >= 0


def test_submit_video_reassembles_prompt_when_editor_sends_stored_prompt(tmp_path, monkeypatch):
    storyboard = tmp_path / "storyboard.jpg"
    character = tmp_path / "xuanmo.jpg"
    storyboard.write_bytes(b"image")
    character.write_bytes(b"image")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        '"page":1,"total_pages":1,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
        '"video_goal":"测试视频",'
        '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
        '"shot_timeline":['
        '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨转向新的饲养箱角落","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
        '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
        '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"新分镜提示词"},'
        '"video":{"status":"needed","prompt":"旧视频提示词"}}]}}}',
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")
    monkeypatch.setattr("api.wetoken.submit_video_task", lambda **kwargs: captured.update(kwargs) or "task-1")

    import asyncio

    asyncio.run(submit_video(SubmitVideoRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
        prompt="旧视频提示词",
        force=True,
    )))

    assert "玄墨转向新的饲养箱角落" in captured["prompt"]
    assert captured["prompt"] != "旧视频提示词"


def test_submit_video_force_uses_custom_prompt_and_clears_old_video(tmp_path, monkeypatch):
    storyboard = tmp_path / "storyboard.jpg"
    character = tmp_path / "xuanmo.jpg"
    old_video = tmp_path / "old.mp4"
    storyboard.write_bytes(b"image")
    character.write_bytes(b"image")
    old_video.write_bytes(b"old video")
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{"玄墨":{"status":"completed","local_path":"' + str(character) + '"}},'
        '"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"asset_refs":{"characters":["玄墨"],"scene":"","props":[]},'
        '"page":1,"total_pages":1,"compact_page":false,"voice_duration":5,"visual_duration":5,"board_duration":5,'
        '"video_goal":"测试视频",'
        '"voice_timeline":[{"beat_id":"v01","type":"narration","text":"玄墨异常。","speaker":"旁白","start":0,"end":5,"duration":5}],'
        '"shot_timeline":['
        '{"shot_id":"s01","start":0,"end":1,"duration":1,"voice_refs":["v01"],"visual":"玄墨不动","camera":"近景","characters":["玄墨"],"scene":"","match_strategy":"sync","purpose":"展示异常","audio_behavior":"narration_sync","continuity_from_previous":null,"transition_type":null},'
        '{"shot_id":"s02","start":1,"end":2,"duration":1,"voice_refs":[],"visual":"箱体特写","camera":"特写","characters":["玄墨"],"scene":"","match_strategy":"supplement","purpose":"补充环境","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s03","start":2,"end":3,"duration":1,"voice_refs":[],"visual":"蛇身绷直","camera":"中景","characters":["玄墨"],"scene":"","match_strategy":"foreshadow","purpose":"制造悬念","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s04","start":3,"end":4,"duration":1,"voice_refs":[],"visual":"主人反应","camera":"近景","characters":[],"scene":"","match_strategy":"reaction_first","purpose":"展示反应","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"},'
        '{"shot_id":"s05","start":4,"end":5,"duration":1,"voice_refs":[],"visual":"黑暗收束","camera":"远景","characters":["玄墨"],"scene":"","match_strategy":"emotional_landing","purpose":"落点","audio_behavior":"ambient_only","continuity_from_previous":"延续","transition_type":"cut"}],'
        '"storyboard_image":{"status":"completed","local_path":"' + str(storyboard) + '","prompt":"画分镜"},'
        '"video":{"status":"completed","task_id":"old-task","prompt":"旧视频提示词","local_path":"' + str(old_video) + '"}}]}}}',
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.routes.tasks.get_api_key", lambda key: "wetoken-key")
    monkeypatch.setattr("api.wetoken.submit_video_task", lambda **kwargs: captured.update(kwargs) or "new-video-task")

    import asyncio
    import json

    result = asyncio.run(submit_video(SubmitVideoRequest(
        project_name="大蟒蛇",
        segment_key="seg_01",
        board_index=0,
        prompt="新视频提示词",
        force=True,
    )))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))
    video = saved["narration_segments"]["seg_01"]["boards"][0]["video"]

    assert result["task_id"] == "new-video-task"
    assert captured["prompt"].startswith("新视频提示词")
    assert "色板参考" in captured["prompt"]
    assert video["status"] == "submitted"
    assert video["prompt"] == captured["prompt"]
    assert video["local_path"] is None
    assert video["previous_local_path"] == str(old_video)
    assert video["history"][-1]["event"] == "submit_video"
    assert video["history"][-1]["prompt"] == captured["prompt"]


def test_asset_output_path_sanitizes_names(tmp_path):
    path = asset_output_path(tmp_path, "props", "铜锁/门缝")

    assert path == tmp_path / "props" / "铜锁_门缝.jpg"


def test_project_list_and_preflight_report_key_blockers(tmp_path, monkeypatch):
    desktop = tmp_path / "Desktop"
    root = desktop / "narration_studio"
    project = root / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","narration_style":"third_person","source_text":"故事",'
        '"assets":{"characters":{},"scenes":{},"props":{}},"narration_segments":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.llm._read_settings", lambda: {
        "llm_provider": "deepseek",
        "deepseek_api_key": "*****f36b",
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-v4-flash",
    })
    monkeypatch.setattr("api.routes.settings.SETTINGS_PATH", tmp_path / "settings.json")
    (tmp_path / "settings.json").write_text(
        '{"vidu_api_key":"*****o0Yg","wetoken_api_key":"*****QmdF",'
        '"idealab_api_key":"","idealab_base_url":"https://api.idealab.com/v1",'
        '"llm_provider":"deepseek","deepseek_api_key":"*****f36b",'
        '"deepseek_base_url":"https://api.deepseek.com","deepseek_model":"deepseek-v4-flash",'
        '"gh_token":"","gh_owner":"","gh_repo":""}',
        encoding="utf-8",
    )

    import asyncio

    projects = asyncio.run(list_projects())
    preflight = asyncio.run(project_preflight("大蟒蛇"))

    assert projects["projects"][0]["name"] == "大蟒蛇"
    assert projects["projects"][0]["source_len"] == 2
    assert preflight["checks"]["llm"]["ok"] is False
    assert preflight["checks"]["vidu"]["ok"] is False
    assert preflight["checks"]["wetoken"]["ok"] is False
    assert any("DeepSeek" in blocker for blocker in preflight["blockers"])


def test_preflight_reports_missing_voice_and_pipeline_integrity_issues(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","source_text":"故事",'
        '"assets":{"characters":{},"scenes":{},"props":{}},'
        '"narration_segments":{"seg_01":{"boards":[{"board_id":"b0001",'
        '"asset_refs":{"characters":[],"scene":"","props":[]},'
        '"voice_timeline":[],"shot_timeline":[],'
        '"storyboard_image":{"status":"needed"},'
        '"video":{"status":"needed"}}]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("api.llm._read_settings", lambda: {
        "llm_provider": "deepseek",
        "deepseek_api_key": "real-key",
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-v4-flash",
    })
    monkeypatch.setattr("api.routes.settings.SETTINGS_PATH", tmp_path / "settings.json")
    (tmp_path / "settings.json").write_text(
        '{"vidu_api_key":"real-vidu","wetoken_api_key":"real-wetoken",'
        '"idealab_api_key":"","idealab_base_url":"https://api.idealab.com/v1",'
        '"llm_provider":"deepseek","deepseek_api_key":"real-key",'
        '"deepseek_base_url":"https://api.deepseek.com","deepseek_model":"deepseek-v4-flash",'
        '"gh_token":"","gh_owner":"","gh_repo":""}',
        encoding="utf-8",
    )

    import asyncio

    preflight = asyncio.run(project_preflight("大蟒蛇"))

    assert preflight["checks"]["audio_ref"]["ok"] is False
    assert preflight["checks"]["pipeline_integrity"]["ok"] is False
    assert any("旁白音色" in blocker for blocker in preflight["blockers"])
    assert any("缺失" in issue for issue in preflight["integrity_issues"])
    assert preflight["can_generate_videos"] is False


def test_plan_project_script_writes_deterministic_script_plan(tmp_path, monkeypatch):
    desktop = tmp_path / "Desktop"
    project = desktop / "narration_studio" / "大蟒蛇"
    project.mkdir(parents=True)
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","narration_style":"third_person",'
        '"source_text":"沈砚看着玄墨。秦越吼道：\\"立刻装箱送来！\\"",'
        '"assets":{"characters":{},"scenes":{},"props":{}},"narration_segments":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import asyncio
    import json

    result = asyncio.run(plan_project_script(DecomposeRequest(project_name="大蟒蛇")))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert saved["script_plan"]["stats"]["boards"] >= 1
    assert saved["script_plan"]["coverage_errors"] == []
    assert saved["script_plan"]["board_plan"][0]["voice_beat_ids"]


def test_reset_project_storyboards_clears_old_boards_and_archives_images(tmp_path, monkeypatch):
    project = tmp_path / "Desktop" / "narration_studio" / "大蟒蛇"
    storyboards = project / "storyboards"
    storyboards.mkdir(parents=True)
    (storyboards / "seg_01_p01.jpg").write_bytes(b"old")
    (project / "pipeline.json").write_text(
        '{"project":"大蟒蛇","narration_style":"third_person","source_text":"故事",'
        '"assets":{"characters":{"医生":{"seed":"医生","status":"completed"}},'
        '"scenes":{},"props":{}},'
        '"_decomposition_outline":{"segments":{"seg_01":{"num_boards":1}}},'
        '"narration_segments":{"seg_01":{"boards":[{"board_id":"old"}]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import asyncio
    import json

    result = asyncio.run(reset_project_storyboards(DecomposeRequest(project_name="大蟒蛇")))
    saved = json.loads((project / "pipeline.json").read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["archived"]
    assert saved["narration_segments"] == {}
    assert "_decomposition_outline" not in saved
    assert saved["assets"]["characters"]["医生"]["seed"] == "医生"
    assert not storyboards.exists()
    assert Path(result["archived"][0]).exists()
