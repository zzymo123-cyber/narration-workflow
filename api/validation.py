VALID_AUDIO_BEHAVIORS = {
    "narration_sync", "narration_over", "dialogue_sync", "dialogue_offscreen",
    "phone_dialogue", "ambient_only", "sound_lead_in", "dramatic_silence", "ambient_transition",
}

VALID_MATCH_STRATEGIES = {
    "sync", "supplement", "contrast", "foreshadow", "reaction_first",
    "reveal", "emotional_landing", "transition",
}

VALID_VOICE_TYPES = {"narration", "dialogue"}
GENERIC_DIALOGUE_SPEAKERS = {"角色", "电话那头", "恐惧语气", "前所未有", "朋友", "他", "她"}

REQUIRED_SHOT_FIELDS = {
    "shot_id", "start", "end", "duration", "voice_refs", "visual",
    "camera", "characters", "scene", "match_strategy", "purpose",
    "audio_behavior", "continuity_from_previous", "transition_type",
}


def _required_voice_duration(beat: dict) -> int:
    from api.duration import dialogue_duration, narration_duration

    if beat.get("type") == "dialogue":
        return dialogue_duration(beat.get("text", ""))
    return narration_duration(beat.get("text", ""))


def validate_board_page(page: dict) -> list[str]:
    errors = []

    # Integer seconds
    for i, beat in enumerate(page.get("voice_timeline", [])):
        beat_type = beat.get("type") or "narration"
        if beat_type not in VALID_VOICE_TYPES:
            errors.append(f"voice_timeline[{i}].type '{beat_type}' is not valid")
        if not str(beat.get("text") or "").strip():
            errors.append(f"voice_timeline[{i}].text must not be empty")
        speaker = str(beat.get("speaker") or "").strip()
        if beat_type == "dialogue":
            if not speaker or speaker in GENERIC_DIALOGUE_SPEAKERS or "语气" in speaker:
                errors.append(f"voice_timeline[{i}].speaker must be a concrete character name, got {speaker or None}")
        elif beat_type == "narration" and not speaker:
            errors.append(f"voice_timeline[{i}].speaker must not be empty")
        for field in ("start", "end", "duration"):
            if not isinstance(beat.get(field), int):
                errors.append(f"voice_timeline[{i}].{field} must be integer, got {beat.get(field)}")
        if all(isinstance(beat.get(field), int) for field in ("start", "end", "duration")):
            if beat["duration"] != beat["end"] - beat["start"]:
                errors.append(f"voice_timeline[{i}].duration {beat['duration']} != end-start {beat['end']-beat['start']}")
            required = _required_voice_duration(beat)
            if beat["duration"] < required:
                errors.append(
                    f"voice_timeline[{i}] duration {beat['duration']}s is too short for text; "
                    f"requires at least {required}s"
                )
    for i, shot in enumerate(page.get("shot_timeline", [])):
        for field in ("start", "end", "duration"):
            if not isinstance(shot.get(field), int):
                errors.append(f"shot_timeline[{i}].{field} must be integer, got {shot.get(field)}")
    for field in ("voice_duration", "visual_duration", "board_duration"):
        if not isinstance(page.get(field), int):
            errors.append(f"{field} must be integer, got {page.get(field)}")

    # Boundary
    if page.get("board_duration", 0) > 15:
        errors.append(f"board_duration {page.get('board_duration')} exceeds 15")
    if page.get("voice_duration", 0) > page.get("board_duration", 0):
        errors.append(f"voice_duration {page.get('voice_duration')} exceeds board_duration {page.get('board_duration')}")

    # video_goal
    if not page.get("video_goal"):
        errors.append("video_goal must not be empty")

    # Shot count
    shot_count = len(page.get("shot_timeline", []))
    compact = page.get("compact_page", False)
    if not compact and shot_count < 5:
        errors.append(f"Regular page must have 5-6 shots, got {shot_count}")
    if compact and shot_count < 3:
        errors.append(f"Compact page must have 3-4 shots, got {shot_count}")
    if shot_count > 6:
        errors.append(f"Page must not exceed 6 shots, got {shot_count}")

    # Shot coverage
    shots = page.get("shot_timeline", [])
    if shots:
        if shots[0].get("start") != 0:
            errors.append(f"First shot start must be 0, got {shots[0].get('start')}")
        if shots[-1].get("end") != page.get("board_duration"):
            errors.append(f"Last shot end must equal board_duration {page.get('board_duration')}, got {shots[-1].get('end')}")
        for i in range(len(shots) - 1):
            if shots[i].get("end") != shots[i + 1].get("start"):
                errors.append(f"Shot gap/overlap: shot_timeline[{i}].end={shots[i].get('end')} != shot_timeline[{i+1}].start={shots[i+1].get('start')}")
        for i, shot in enumerate(shots):
            if shot.get("duration") is not None and shot.get("start") is not None and shot.get("end") is not None:
                if shot["duration"] != shot["end"] - shot["start"]:
                    errors.append(f"shot_timeline[{i}].duration {shot['duration']} != end-start {shot['end']-shot['start']}")

    # voice_refs references
    beat_ids = {b.get("beat_id") for b in page.get("voice_timeline", [])}
    referenced_beats = set()
    for i, shot in enumerate(shots):
        for ref in shot.get("voice_refs", []):
            if ref not in beat_ids:
                errors.append(f"shot_timeline[{i}].voice_refs contains unknown beat_id '{ref}'")
            else:
                referenced_beats.add(ref)

    # Orphan beats
    for beat in page.get("voice_timeline", []):
        if beat.get("beat_id") not in referenced_beats:
            errors.append(f"Orphan beat: beat_id '{beat.get('beat_id')}' not referenced by any shot")

    # Shot field completeness
    for i, shot in enumerate(shots):
        missing = REQUIRED_SHOT_FIELDS - set(shot.keys())
        if missing:
            errors.append(f"shot_timeline[{i}] missing fields: {missing}")
        if shot.get("audio_behavior") not in VALID_AUDIO_BEHAVIORS:
            errors.append(f"shot_timeline[{i}].audio_behavior '{shot.get('audio_behavior')}' is not valid")
        if shot.get("match_strategy") not in VALID_MATCH_STRATEGIES:
            errors.append(f"shot_timeline[{i}].match_strategy '{shot.get('match_strategy')}' is not valid")

    # First shot constraints
    if shots:
        if shots[0].get("continuity_from_previous") is not None:
            errors.append("First shot continuity_from_previous must be null")
        if shots[0].get("transition_type") is not None:
            errors.append("First shot transition_type must be null")

    return errors
