"""NHTSA MMUCC 6th Edition crash taxonomy classification and backfill.

This module adds a video-observable taxonomy alongside the existing crash review.

The semantic category names follow the NHTSA Model Minimum Uniform Crash
Criteria (MMUCC), Sixth Edition, for:

* C7 First Harmful Event
* C9 Manner of Collision of the First Harmful Event
* V37 Sequence of Events

``event_kind`` is a dataset-specific extension because near collisions are not
reportable crashes and therefore fall outside MMUCC crash coding.

The ``*_code`` fields below are stable explicit identifiers made from the MMUCC
element identifier plus the canonical attribute name. They are deliberately not
presented as official MMUCC numeric codes.
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import settings
from .shared import (
    classify_download_failure,
    clean_text,
    log,
    normalise_string_list,
    recover_json,
    save_state,
    unload_model,
)


CRASH_TAXONOMY_VERSION = "nhtsa_mmucc6_video_v3"
CRASH_TAXONOMY_STANDARD = "NHTSA MMUCC 6th Edition (2024)"
TAXONOMY_REVIEW_ATTEMPTS = 2

# A failed download is skipped immediately; yt-dlp already retries internally.
# Retrying on later cycles kept records pending forever and, because pending
# taxonomy blocks discovery, stalled the whole pipeline. Clearing
# ``crash_taxonomy_version`` on such segments re-queues them.
DOWNLOAD_TERMINAL_STATUSES = {
    "video_unavailable",
    "age_restricted",
    "download_failed",
}


EVENT_KINDS = {
    "collision",
    "near_collision",
}

MANNER_OF_COLLISION_VALUES = {
    "not_collision_with_motor_vehicle_in_transport",
    "angle",
    "front_to_front",
    "front_to_rear_or_rear_to_front",
    "rear_to_rear",
    "rear_to_side_or_side_to_rear",
    "sideswipe_opposite_direction",
    "sideswipe_same_direction",
    "other",
    "unknown",
}

# Short MMUCC C9 definitions shown to the model. v1 showed no definitions and an
# example answer, and the model returned that example for 99% of collisions.
MANNER_OF_COLLISION_DEFINITIONS = {
    "front_to_rear_or_rear_to_front": (
        "the front of one vehicle strikes the rear of another (rear-end)"
    ),
    "front_to_front": "the fronts of two vehicles strike each other (head-on)",
    "angle": (
        "the front of one vehicle strikes the side of another at an angle "
        "(T-bone, broadside, most intersection crashes)"
    ),
    "sideswipe_same_direction": (
        "the sides of two vehicles moving in the same direction scrape or "
        "strike each other"
    ),
    "sideswipe_opposite_direction": (
        "the sides of two vehicles moving in opposite directions scrape or "
        "strike each other"
    ),
    "rear_to_side_or_side_to_rear": (
        "the rear of one vehicle strikes the side of another"
    ),
    "rear_to_rear": "the rears of two vehicles strike each other",
    "not_collision_with_motor_vehicle_in_transport": (
        "the first harmful event is not a collision with another moving motor "
        "vehicle (e.g. a pedestrian, cyclist, barrier, parked car, rollover)"
    ),
    "other": "a collision between moving motor vehicles not covered above",
    "unknown": "the orientation at impact cannot be seen",
}

CRASH_TYPE_VALUES = {
    "rear_end",
    "side_impact",
    "head_on",
    "rollover",
    "pedestrian",
    "cyclist",
    "multi_vehicle",
    "single_vehicle",
    "near_collision",
    "other",
    "unknown",
}

MANNER_OF_COLLISION_CODES = {
    value: "C9_" + value.upper()
    for value in MANNER_OF_COLLISION_VALUES
}

FIRST_HARMFUL_EVENT_VALUES = {
    # Group 1: Non-Collision Harmful Events
    "rollover_overturn",
    "cargo_equipment_loss_shift_or_damage_harmful",
    "fell_jumped_from_motor_vehicle",
    "fire_explosion",
    "immersion_full_or_partial",
    "jackknife_harmful",
    "thrown_or_falling_object",
    "pavement_surface_irregularity",
    "injured_in_vehicle_non_collision",
    "gas_inhalation",
    "other_non_collision",
    # Group 2: Collision with Motor Vehicle
    "motor_vehicle_in_transport",
    "parked_motor_vehicle",
    "working_motor_vehicle",
    # Group 3: Collision with Non-Fixed Object
    "non_motorist",
    "live_animal",
    "ridden_animal_or_animal_drawn_conveyance",
    "railroad_vehicle",
    "road_vehicle_on_rails",
    "strikes_object_at_rest_that_had_fallen_from_motor_vehicle_in_transport",
    "striking_struck_by_object_cargo_person_from_other_motor_vehicle_in_transport",
    "other_object_not_fixed",
    "unknown_object_not_fixed",
    # Group 4: Collision with Fixed Object
    "bridge_overhead_structure",
    "bridge_pier_or_support",
    "bridge_rail_includes_parapet",
    "building",
    "wall",
    "cable_barrier",
    "concrete_traffic_barrier",
    "guardrail_face",
    "guardrail_end",
    "guardrail_end_treatment",
    "impact_attenuator_crash_cushion",
    "other_traffic_barrier",
    "traffic_sign_support",
    "traffic_signal_support",
    "utility_pole_light_support",
    "other_post_pole_or_other_supports",
    "culvert",
    "curb",
    "ditch",
    "embankment",
    "boulder",
    "ground",
    "tree_standing_only",
    "shrubbery",
    "snowbank",
    "fence",
    "mailbox",
    "fire_hydrant",
    "other_fixed_object",
    "unknown_fixed_object",
    "unknown",
}

FIRST_HARMFUL_EVENT_CODES = {
    value: "C7_" + value.upper()
    for value in FIRST_HARMFUL_EVENT_VALUES
}

SEQUENCE_OF_EVENT_VALUES = {
    # Group 1: Non-Harmful Events
    "cross_centerline",
    "cross_median",
    "end_departure",
    "downhill_runaway",
    "equipment_failure",
    "ran_off_roadway_left",
    "ran_off_roadway_right",
    "ran_off_roadway_direction_unknown",
    "non_harmful_swaying_trailer_jackknife",
    "cargo_equipment_loss_or_shift_non_harmful",
    "reentering_roadway",
    "separation_of_units",
    "vehicle_went_airborne",
    # Group 2: Non-Collision Harmful Events
    "rollover_overturn",
    "cargo_equipment_loss_shift_or_damage_harmful",
    "fell_jumped_from_motor_vehicle",
    "fire_explosion",
    "immersion_full_or_partial",
    "jackknife_harmful",
    "thrown_or_falling_object",
    "pavement_surface_irregularity",
    "injured_in_vehicle_non_collision",
    "gas_inhalation",
    "other_non_collision",
    # Group 3: Collision with Motor Vehicle
    "motor_vehicle_in_transport",
    "parked_motor_vehicle",
    "working_motor_vehicle",
    # Group 4: Collision with Non-Fixed Object
    "non_motorist",
    "live_animal",
    "ridden_animal_or_animal_drawn_conveyance",
    "railroad_vehicle",
    "road_vehicle_on_rails",
    "strikes_object_at_rest_that_had_fallen_from_motor_vehicle_in_transport",
    "striking_struck_by_object_cargo_person_from_other_motor_vehicle_in_transport",
    "other_object_not_fixed",
    "unknown_object_not_fixed",
    # Group 5: Collision with Fixed Object
    "bridge_overhead_structure",
    "bridge_pier_or_support",
    "bridge_rail_includes_parapet",
    "building",
    "wall",
    "cable_barrier",
    "concrete_traffic_barrier",
    "guardrail_face",
    "guardrail_end",
    "guardrail_end_treatment",
    "impact_attenuator_crash_cushion",
    "other_traffic_barrier",
    "traffic_sign_support",
    "traffic_signal_support",
    "utility_pole_light_support",
    "other_post_pole_or_other_supports",
    "culvert",
    "curb",
    "ditch",
    "embankment",
    "boulder",
    "ground",
    "tree_standing_only",
    "shrubbery",
    "snowbank",
    "fence",
    "mailbox",
    "fire_hydrant",
    "other_fixed_object",
    "unknown_fixed_object",
    # Group 6
    "unknown",
}

NON_HARMFUL_SEQUENCE_VALUES = {
    "cross_centerline",
    "cross_median",
    "end_departure",
    "downhill_runaway",
    "equipment_failure",
    "ran_off_roadway_left",
    "ran_off_roadway_right",
    "ran_off_roadway_direction_unknown",
    "non_harmful_swaying_trailer_jackknife",
    "cargo_equipment_loss_or_shift_non_harmful",
    "reentering_roadway",
    "separation_of_units",
    "vehicle_went_airborne",
}


def _normalised_enum(value: Any) -> str:
    return clean_text(value).strip().lower().replace("-", "_").replace(" ", "_")


def _taxonomy_empty_fields(
    status: str,
    *,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "event_kind": None,
        "manner_of_collision_code": None,
        "manner_of_collision": None,
        "first_harmful_event_code": None,
        "first_harmful_event": None,
        "sequence_of_events": [],
        "crash_taxonomy_standard": CRASH_TAXONOMY_STANDARD,
        "crash_taxonomy_status": status,
        "crash_taxonomy_version": CRASH_TAXONOMY_VERSION,
        "crash_taxonomy_error": error,
    }


# Words the model writes in sequence_of_events for an allowed MMUCC value.
SEQUENCE_SYNONYMS = {
    "rollover": "rollover_overturn",
    "overturn": "rollover_overturn",
    "fire": "fire_explosion",
    "explosion": "fire_explosion",
    "pedestrian": "non_motorist",
    "cyclist": "non_motorist",
    "bicyclist": "non_motorist",
    "animal": "live_animal",
    "tree": "tree_standing_only",
    "pole": "utility_pole_light_support",
    "guardrail": "guardrail_face",
    "barrier": "other_traffic_barrier",
    "parked_vehicle": "parked_motor_vehicle",
    "parked_car": "parked_motor_vehicle",
    "airborne": "vehicle_went_airborne",
    "jackknife": "jackknife_harmful",
}

# First harmful events that say the impact itself was not seen.
UNSEEN_FIRST_HARMFUL_EVENTS = {"unknown", "unknown_object_not_fixed", "unknown_fixed_object"}

# Manners that describe two motor vehicles in transport meeting.
VEHICLE_MANNERS = MANNER_OF_COLLISION_VALUES - {
    "not_collision_with_motor_vehicle_in_transport",
    "unknown",
}


def repair_taxonomy_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix the model's answer where the intended MMUCC value is unambiguous.

    v2 rejected 10k of 24k answers, mostly for free text in sequence_of_events,
    and forced manner "not_collision_with_motor_vehicle_in_transport" whenever
    the first harmful event was unknown, so 94% of collisions got that manner.
    Anything that still breaks the MMUCC rules after this is rejected as before.
    """
    if not isinstance(data, dict):
        return data
    data = dict(data)
    event_kind = _normalised_enum(data.get("event_kind"))
    if event_kind == "near_collision":
        # Outside MMUCC crash coding: these fields are empty by definition.
        data["manner_of_collision"] = None
        data["first_harmful_event"] = None
        data["sequence_of_events"] = []
        return data
    if event_kind != "collision":
        return data

    if not clean_text(data.get("impact_description")):
        data["impact_description"] = "not described"

    manner = _normalised_enum(data.get("manner_of_collision"))
    first = _normalised_enum(data.get("first_harmful_event"))
    if first in UNSEEN_FIRST_HARMFUL_EVENTS:
        if manner in VEHICLE_MANNERS:
            # The orientation was seen, so the other party was a moving vehicle.
            first = "motor_vehicle_in_transport"
        elif manner == "not_collision_with_motor_vehicle_in_transport":
            # An unseen impact says nothing about what was hit.
            manner = "unknown"
    data["manner_of_collision"] = manner
    data["first_harmful_event"] = first

    raw = data.get("sequence_of_events")
    values = raw if isinstance(raw, list) else []
    sequence: List[str] = []
    for value in values:
        value = _normalised_enum(value)
        value = SEQUENCE_SYNONYMS.get(value, value)
        # Free text and values of other fields are dropped, not guessed at.
        if value in SEQUENCE_OF_EVENT_VALUES and value not in sequence:
            sequence.append(value)
    if first in SEQUENCE_OF_EVENT_VALUES:
        # The first harmful event must be in the sequence, after any
        # non-harmful precursors and before every other harmful event.
        precursors = [v for v in sequence if v in NON_HARMFUL_SEQUENCE_VALUES]
        later = [v for v in sequence if v not in NON_HARMFUL_SEQUENCE_VALUES and v != first]
        sequence = precursors + [first] + later
    data["sequence_of_events"] = sequence[:4]
    return data


def validate_taxonomy_response(data: Dict[str, Any]) -> Optional[str]:
    """Return a semantic validation error for a taxonomy-only model response."""
    event_kind = _normalised_enum(data.get("event_kind"))
    if event_kind not in EVENT_KINDS:
        return "event_kind_must_be_collision_or_near_collision"

    crash_type = _normalised_enum(data.get("crash_type"))
    if crash_type not in CRASH_TYPE_VALUES:
        return "crash_type_must_be_one_allowed_value"
    if (event_kind == "near_collision") != (crash_type == "near_collision"):
        return "crash_type_near_collision_must_match_event_kind"
    if not clean_text(data.get("impact_description")):
        return "impact_description_must_not_be_empty"

    manner = data.get("manner_of_collision")
    first_harmful = data.get("first_harmful_event")
    sequence = data.get("sequence_of_events")

    if not isinstance(sequence, list):
        return "sequence_of_events_must_be_a_list"
    if len(sequence) > 4:
        return "sequence_of_events_must_have_at_most_four_values"

    sequence_values = [_normalised_enum(value) for value in sequence]
    if len(sequence_values) != len(set(sequence_values)):
        return "sequence_of_events_must_not_contain_duplicates"
    if any(value not in SEQUENCE_OF_EVENT_VALUES for value in sequence_values):
        return "sequence_of_events_contains_unknown_value"

    if event_kind == "near_collision":
        if manner is not None or first_harmful is not None:
            return "near_collision_must_not_have_mmucc_harmful_event_fields"
        if sequence_values:
            return "near_collision_must_have_empty_sequence_of_events"
        return None

    manner_value = _normalised_enum(manner)
    first_value = _normalised_enum(first_harmful)

    if manner_value not in MANNER_OF_COLLISION_VALUES:
        return "manner_of_collision_must_be_one_allowed_value"
    if first_value not in FIRST_HARMFUL_EVENT_VALUES:
        return "first_harmful_event_must_be_one_allowed_value"

    if first_value == "motor_vehicle_in_transport":
        if manner_value == "not_collision_with_motor_vehicle_in_transport":
            return "motor_vehicle_in_transport_requires_collision_manner"
    elif first_value in UNSEEN_FIRST_HARMFUL_EVENTS:
        if manner_value != "unknown":
            return "unseen_first_harmful_event_requires_unknown_manner"
    elif manner_value != "not_collision_with_motor_vehicle_in_transport":
        return "non_motor_vehicle_first_harmful_event_requires_not_collision_manner"

    if not sequence_values:
        return "collision_requires_at_least_one_sequence_event"

    if first_value not in sequence_values:
        return "sequence_of_events_must_include_first_harmful_event"

    first_harmful_index = sequence_values.index(first_value)
    for position, value in enumerate(sequence_values):
        if position < first_harmful_index and value not in NON_HARMFUL_SEQUENCE_VALUES:
            return "harmful_sequence_event_cannot_precede_first_harmful_event"

    return None


def normalise_taxonomy_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a validated model response into stable state and CSV fields."""
    event_kind = _normalised_enum(data.get("event_kind"))

    checked = {
        "crash_type": _normalised_enum(data.get("crash_type")),
        "impact_description": clean_text(data.get("impact_description")),
    }

    if event_kind == "near_collision":
        return {
            **checked,
            "event_kind": "near_collision",
            "manner_of_collision_code": None,
            "manner_of_collision": None,
            "first_harmful_event_code": None,
            "first_harmful_event": None,
            "sequence_of_events": [],
            "crash_taxonomy_standard": CRASH_TAXONOMY_STANDARD,
            "crash_taxonomy_status": "classified",
            "crash_taxonomy_version": CRASH_TAXONOMY_VERSION,
            "crash_taxonomy_error": None,
        }

    manner = _normalised_enum(data.get("manner_of_collision"))
    first_harmful = _normalised_enum(data.get("first_harmful_event"))
    sequence = [
        _normalised_enum(value)
        for value in normalise_string_list(data.get("sequence_of_events"))
    ][:4]

    return {
        **checked,
        "event_kind": "collision",
        "manner_of_collision_code": MANNER_OF_COLLISION_CODES[manner],
        "manner_of_collision": manner,
        "first_harmful_event_code": FIRST_HARMFUL_EVENT_CODES[first_harmful],
        "first_harmful_event": first_harmful,
        "sequence_of_events": sequence,
        "crash_taxonomy_standard": CRASH_TAXONOMY_STANDARD,
        "crash_taxonomy_status": "classified",
        "crash_taxonomy_version": CRASH_TAXONOMY_VERSION,
        "crash_taxonomy_error": None,
    }


# v3 classifies in two steps, because v2's single prompt over 24 small frames of
# the whole clip missed the moment of contact: on clips two human reviewers
# labelled, it called most collisions near collisions. The model first reads the
# time printed on frames of the whole clip to find first contact; then it sees
# short windows around that moment (and around the largest frame-to-frame
# changes) and answers plain questions. Fixed rules turn the answers into MMUCC.
LOCATE_PROMPT = """
This is a road traffic video clip. Every frame shows its time in seconds at the top left as t=...
Find the first moment of physical contact: two road users touching, or a vehicle touching an object,
a person or an animal, or a vehicle starting to roll over. If nothing touches, choose the moment the
road users come closest to each other.
Return one JSON object only: {"contact_time": <the t value of that moment, a number>,
"what_happens": "<one sentence>"}
""".strip()

DETAIL_PROMPT = """
This short road traffic clip (about 4 seconds) is centred on the moment two road users come into
contact or come closest. Answer from what you can see. Always choose one of the given values.
Return one JSON object only, with these keys:
{
 "contact": one of "vehicle_vehicle" (two moving motor vehicles touch), "vehicle_object" (a vehicle
   hits a barrier, pole, tree, wall, parked vehicle, debris or other object), "vehicle_person" (a
   vehicle hits a pedestrian, cyclist or animal), "rollover" (a vehicle overturns without hitting
   anything first), "none" (nothing touches: a near miss);
 "hit": for vehicle_object or vehicle_person, what was hit: "pedestrian", "cyclist", "animal",
   "parked_vehicle", "barrier_or_guardrail", "pole_or_sign", "tree", "wall_or_building" or "other";
 "part_a": for vehicle_vehicle, the part of the first vehicle (the camera vehicle if it is involved)
   that touches first: "front", "rear", "left_side" or "right_side";
 "part_b": the part of the other vehicle that touches first: "front", "rear", "left_side" or
   "right_side";
 "directions": just before contact the two vehicles were moving "same" way, "opposite" ways, or
   "crossing" paths;
 "description": one sentence describing the contact.
}
Fill every key with your best choice; keys that do not apply to the contact are ignored.
""".strip()

TAXONOMY_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TAXONOMY_WINDOW_SECONDS = 2.0
TAXONOMY_MAX_WINDOWS = 3
CONTACT_VALUES = {"vehicle_vehicle", "vehicle_object", "vehicle_person", "rollover", "none"}
HIT_FIRST_HARMFUL_EVENTS = {
    "pedestrian": "non_motorist",
    "cyclist": "non_motorist",
    "animal": "live_animal",
    "parked_vehicle": "parked_motor_vehicle",
    "barrier_or_guardrail": "other_traffic_barrier",
    "pole_or_sign": "other_post_pole_or_other_supports",
    "tree": "tree_standing_only",
    "wall_or_building": "wall",
    "other": "other_object_not_fixed",
}
MANNER_CRASH_TYPES = {
    "front_to_rear_or_rear_to_front": "rear_end",
    "front_to_front": "head_on",
    "angle": "side_impact",
    "sideswipe_same_direction": "side_impact",
    "sideswipe_opposite_direction": "side_impact",
    "rear_to_side_or_side_to_rear": "side_impact",
    "rear_to_rear": "other",
    "other": "other",
}


def manner_from_parts(part_a: str, part_b: str, directions: str) -> str:
    """MMUCC C9 manner of collision from the parts that touched first."""
    side = {"left_side", "right_side"}
    parts = {part_a, part_b}
    if parts == {"front", "rear"}:
        return "front_to_rear_or_rear_to_front"
    if parts == {"front"}:
        return "angle" if directions == "crossing" else "front_to_front"
    if "front" in parts and parts & side:
        return "angle"
    if parts <= side and parts:
        if directions == "opposite":
            return "sideswipe_opposite_direction"
        return "angle" if directions == "crossing" else "sideswipe_same_direction"
    if "rear" in parts and parts & side:
        return "rear_to_side_or_side_to_rear"
    if parts == {"rear"}:
        return "rear_to_rear"
    return "other"


def taxonomy_from_contact(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the plain answers of the detail step into the MMUCC response fields."""
    contact = _normalised_enum(detail.get("contact"))
    # An empty description must not throw away an otherwise usable answer.
    description = clean_text(detail.get("description")) or "not described"
    if contact == "none":
        return {"impact_description": description, "event_kind": "near_collision",
                "crash_type": "near_collision", "manner_of_collision": None,
                "first_harmful_event": None, "sequence_of_events": []}
    if contact == "vehicle_vehicle":
        manner = manner_from_parts(
            _normalised_enum(detail.get("part_a")),
            _normalised_enum(detail.get("part_b")),
            _normalised_enum(detail.get("directions")),
        )
        first, crash_type = "motor_vehicle_in_transport", MANNER_CRASH_TYPES[manner]
    elif contact == "rollover":
        manner, first, crash_type = (
            "not_collision_with_motor_vehicle_in_transport", "rollover_overturn", "rollover")
    else:
        hit = _normalised_enum(detail.get("hit"))
        manner = "not_collision_with_motor_vehicle_in_transport"
        first = HIT_FIRST_HARMFUL_EVENTS.get(hit, "other_object_not_fixed")
        crash_type = {"pedestrian": "pedestrian", "cyclist": "cyclist"}.get(hit, "single_vehicle")
    return {"impact_description": description, "event_kind": "collision", "crash_type": crash_type,
            "manner_of_collision": manner, "first_harmful_event": first, "sequence_of_events": [first]}


# Segments skipped because their source could not be downloaded. They are
# classified after all once a copy of the video appears in SHARED_VIDEO_DIRS.
SKIPPED_FOR_SOURCE = {"download_failed", "video_unavailable", "age_restricted"}


def _record_video_id(record: Dict[str, Any]) -> Optional[str]:
    metadata = record.get("metadata")
    url = metadata.get("youtube_url", "") if isinstance(metadata, dict) else ""
    match = re.search(r"[?&]v=([\w-]{11})", str(url or ""))
    return match.group(1) if match else None


def _shared_copy_signature(record: Dict[str, Any]) -> Optional[str]:
    """Path and size of a shared copy of the record's video, if there is one.

    Only a cheap existence check: this runs for every skipped record on every
    cycle. The copy is validated when the taxonomy stage opens it.
    """
    video_id = _record_video_id(record)
    if not video_id:
        return None
    for folder in settings.SHARED_VIDEO_DIRS:
        for suffix in (".mp4", ".mkv", ".webm", ".mov"):
            path = folder / f"{video_id}{suffix}"
            if path.is_file():
                return f"{path}:{path.stat().st_size}"
    return None


def _retry_from_shared_copy(record: Dict[str, Any]) -> bool:
    """A skipped record whose video has since become available locally.

    A copy that already failed once is not tried again until it changes.
    """
    if record.get("crash_taxonomy_status") not in SKIPPED_FOR_SOURCE:
        return False
    signature = _shared_copy_signature(record)
    return signature is not None and signature != record.get(
        "crash_taxonomy_failed_source"
    )


def _segment_pending(segment: Any, retry_skipped: bool) -> bool:
    if not isinstance(segment, dict):
        return False
    if segment.get("crash_taxonomy_version") != CRASH_TAXONOMY_VERSION:
        return True
    return retry_skipped and segment.get("crash_taxonomy_status") in SKIPPED_FOR_SOURCE


def crash_taxonomy_pending(record: Any) -> bool:
    """Whether a completed record still needs taxonomy backfill."""
    if not isinstance(record, dict) or record.get("status") != "complete":
        return False

    segments = record.get("segments", [])
    if not isinstance(segments, list):
        return False

    retry_skipped = _retry_from_shared_copy(record)
    return any(_segment_pending(segment, retry_skipped) for segment in segments)


def _pending_segments(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    retry_skipped = _retry_from_shared_copy(record)
    return [
        segment
        for segment in record.get("segments", [])
        if _segment_pending(segment, retry_skipped)
    ]


def _mark_video_unavailable(
    record: Dict[str, Any],
    segments: List[Dict[str, Any]],
    error: str,
    status: str = "video_unavailable",
) -> None:
    for segment in segments:
        # A re-classification cannot undo the earlier collision/near-collision
        # decision just because the video is gone; keep it.
        previous_event_kind = segment.get("event_kind")
        segment.update(
            _taxonomy_empty_fields(
                status,
                error=error,
            )
        )
        if previous_event_kind in EVENT_KINDS:
            segment["event_kind"] = previous_event_kind

    record["crash_taxonomy_status"] = status
    record["crash_taxonomy_error"] = error
    record["crash_taxonomy_version"] = CRASH_TAXONOMY_VERSION
    # Fields written by the earlier cooldown-based retry logic.
    record.pop("crash_taxonomy_download_retry_after", None)
    record.pop("crash_taxonomy_download_attempts", None)


def _existing_video_path(video_id: str, record: Dict[str, Any]) -> Optional[Path]:
    from .crash_review import _valid_video

    candidates: List[Path] = []

    stored = record.get("downloaded_path")
    if stored:
        candidates.append(Path(str(stored)))

    if settings.VIDEO_DIR.exists():
        candidates.extend(
            path
            for path in settings.VIDEO_DIR.glob(f"{video_id}.*")
            if path.is_file()
        )

    seen = set()
    for path in candidates:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file() and _valid_video(path):
            return path

    return None


def _obtain_video(video_id: str, record: Dict[str, Any]) -> Path:
    from .crash_review import download_video

    existing = _existing_video_path(video_id, record)
    if existing is not None:
        record["downloaded_path"] = str(existing)
        log(f"Reusing downloaded video for taxonomy backfill: {video_id}")
        return existing

    path = download_video(video_id)
    record["downloaded_path"] = str(path)
    return path


def _taxonomy_sample(
    video_path: Path, start: float, duration: float, destination: Path,
    *, fps: float, width: int, stamp_time: bool,
) -> Path:
    from .crash_review import _run_sample_command

    filters = f"fps={fps:.4g},scale={width}:-2,setpts=PTS-STARTPTS"
    if stamp_time:
        filters += (
            f",drawtext=fontfile={TAXONOMY_FONT}:text='t=%{{pts\\:flt}}':x=8:y=8:"
            "fontsize=26:fontcolor=white:box=1:boxcolor=black@0.7"
        )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{start:.6f}", "-i", str(video_path), "-t", f"{duration:.6f}",
        "-vf", filters, "-an", "-c:v", "libx264", "-bf", "0", "-preset", "veryfast",
        "-crf", "22", "-y", str(destination),
    ]
    _run_sample_command(command, destination, timeout=600, label="failed to create taxonomy sample")
    return destination


def _motion_peaks(video_path: Path, start: float, duration: float, count: int = 2) -> List[float]:
    """Clip-relative times of the largest frame-to-frame changes, 1.5 s apart.

    A crash jolts the camera or changes the picture abruptly, so these are
    candidate contact moments next to the model's own estimate.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    capture = cv2.VideoCapture(str(video_path))
    changes, previous, offset = [], None, 0.0
    while offset <= duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, (start + offset) * 1000)
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            changes.append((float(np.mean(np.abs(gray - previous))), offset))
        previous, offset = gray, offset + 0.1
    capture.release()
    peaks: List[float] = []
    for _, when in sorted(changes, reverse=True):
        if all(abs(when - peak) >= 1.5 for peak in peaks):
            peaks.append(when)
        if len(peaks) == count:
            break
    return peaks


def _ask_json(judge: Any, path: Path, prompt: str, fps: float) -> tuple:
    answer = ""
    for _ in range(TAXONOMY_REVIEW_ATTEMPTS):
        answer = judge._generate(path, prompt, fps)
        data = recover_json(answer)
        if isinstance(data, dict):
            return data, answer
    return None, answer


def _review_one_segment(
    judge: Any,
    video_path: Path,
    video_id: str,
    segment: Dict[str, Any],
    metadata: Dict[str, Any],
    work: Path,
) -> Dict[str, Any]:
    start = float(segment.get("start_time"))
    duration = max(float(segment.get("end_time")) - start, 0.5)
    index = int(segment.get("segment_index", 0))

    # Step 1: candidate moments of first contact.
    candidates: List[tuple] = []
    overview_fps = min(4.0, max(32 / duration, 1.0))
    try:
        overview = _taxonomy_sample(
            video_path, start, duration, work / f"taxonomy-{index:05d}-overview.mp4",
            fps=overview_fps, width=480, stamp_time=Path(TAXONOMY_FONT).is_file(),
        )
        located, _ = _ask_json(judge, overview, LOCATE_PROMPT, overview_fps)
        contact_time = float((located or {}).get("contact_time"))
        if 0 <= contact_time <= duration:
            candidates.append(("model", contact_time))
    except (TypeError, ValueError):
        pass
    for peak in _motion_peaks(video_path, start, duration):
        if all(abs(peak - when) >= 1.0 for _, when in candidates):
            candidates.append(("motion", peak))
    if not candidates:
        candidates.append(("middle", duration / 2))

    # Step 2: plain questions on a short window around each candidate. The
    # earliest window in which contact is seen decides; a near miss otherwise.
    windows = []
    for source, when in candidates[:TAXONOMY_MAX_WINDOWS]:
        w0 = max(0.0, when - TAXONOMY_WINDOW_SECONDS)
        w1 = min(duration, when + TAXONOMY_WINDOW_SECONDS)
        sample = _taxonomy_sample(
            video_path, start + w0, max(w1 - w0, 0.5),
            work / f"taxonomy-{index:05d}-window.mp4", fps=10.0, width=640, stamp_time=False,
        )
        detail, answer = _ask_json(judge, sample, DETAIL_PROMPT, 10.0)
        if detail is not None and _normalised_enum(detail.get("contact")) in CONTACT_VALUES:
            windows.append((detail, answer, source, when))
    if not windows:
        return {"status": "model_error", "error": "model_did_not_return_usable_taxonomy", "raw_response": ""}
    detail, answer, source, when = min(
        windows,
        key=lambda item: (_normalised_enum(item[0].get("contact")) == "none", item[3]),
    )

    # Step 3: fixed rules, then the same MMUCC checks as before.
    data = repair_taxonomy_response(taxonomy_from_contact(detail))
    error = validate_taxonomy_response(data)
    if error is not None:
        return {"status": "model_error", "error": f"semantic_error: {error}", "raw_response": answer}
    result = normalise_taxonomy_response(data)
    result["crash_taxonomy_raw_response"] = answer
    result["crash_taxonomy_contact"] = {
        key: detail.get(key) for key in ("contact", "hit", "part_a", "part_b", "directions")
    }
    result["crash_taxonomy_impact_time"] = round(start + when, 3)
    result["crash_taxonomy_impact_source"] = source
    return result


def _store_model_error(
    segment: Dict[str, Any],
    result: Dict[str, Any],
) -> bool:
    """Store retry state. Return True when the failure became terminal."""
    try:
        previous_cycles = max(
            0,
            int(segment.get("crash_taxonomy_review_cycles", 0)),
        )
    except (TypeError, ValueError):
        previous_cycles = 0

    cycles = previous_cycles + 1
    segment["crash_taxonomy_review_cycles"] = cycles
    segment["crash_taxonomy_status"] = "model_error"
    segment["crash_taxonomy_error"] = str(result.get("error") or "model_error")
    segment["crash_taxonomy_raw_response"] = str(
        result.get("raw_response") or ""
    )
    segment["crash_taxonomy_standard"] = CRASH_TAXONOMY_STANDARD

    if cycles < settings.MAX_REVIEW_CYCLES:
        return False

    segment.update(
        _taxonomy_empty_fields(
            "model_error_terminal",
            error=segment["crash_taxonomy_error"],
        )
    )
    segment["crash_taxonomy_review_cycles"] = cycles
    return True


def _cleanup_taxonomy_video_if_safe(
    record: Dict[str, Any],
    video_path: Path,
) -> None:
    """
    Delete a retained/downloaded source only when no location review still
    needs the video. This preserves the existing pipeline's deletion policy.
    """
    if not settings.DELETE_VIDEO_AFTER_PROCESSING:
        return

    # A retryable taxonomy result still needs this exact source on a later
    # cycle. Retaining it avoids an unnecessary YouTube re-download.
    if crash_taxonomy_pending(record):
        return

    from .crash_review import (
        _needs_location_visual_review,
        has_location_visual_review_errors,
    )

    if has_location_visual_review_errors(record):
        return

    if any(
        isinstance(segment, dict)
        and _needs_location_visual_review(segment)
        for segment in record.get("segments", [])
    ):
        return

    video_path.unlink(missing_ok=True)
    record["downloaded_path"] = None
    log(f"Deleted taxonomy-reviewed video: {video_path}")


def run_crash_taxonomy_stage(
    state: Dict[str, Any],
    max_videos: Optional[int] = None,
) -> int:
    """
    Backfill MMUCC taxonomy for completed accepted segments.

    Old state is upgraded on restart. Clearly unavailable YouTube sources are
    terminally marked ``video_unavailable``. Operational download failures
    remain pending with a cooldown so another historical video can progress.
    """
    all_pending_records = [
        (video_id, record)
        for video_id, record in state.get("videos", {}).items()
        if crash_taxonomy_pending(record)
    ]

    if max_videos is None:
        video_limit = settings.MAX_VIDEOS_PER_RUN
    else:
        try:
            video_limit = max(1, int(max_videos))
        except (TypeError, ValueError):
            raise ValueError("max_videos must be a positive integer")

    pending_records = all_pending_records[:video_limit]

    if not pending_records:
        return 0

    judge: Any = None
    processed = 0

    try:
        for video_id, record in pending_records:
            segments = _pending_segments(record)
            if not segments:
                continue

            try:
                video_path = _obtain_video(video_id, record)
            except KeyboardInterrupt:
                save_state(settings.STATE_JSON, state)
                raise
            except Exception as exc:
                error = clean_text(exc) or "video_download_failed"
                status = classify_download_failure(error) or "download_failed"
                _mark_video_unavailable(record, segments, error, status)
                # Do not retry the same shared copy until it changes.
                record["crash_taxonomy_failed_source"] = _shared_copy_signature(record)
                log(f"Skipping taxonomy backfill for {video_id} ({status}): {error}")
                processed += len(segments)
                save_state(settings.STATE_JSON, state)
                continue

            # Clear leftovers from the earlier cooldown-based retry logic.
            record.pop("crash_taxonomy_download_retry_after", None)
            record.pop("crash_taxonomy_download_attempts", None)
            if record.get("crash_taxonomy_status") == "download_error":
                record["crash_taxonomy_status"] = None
                record["crash_taxonomy_error"] = None
            record.pop("crash_taxonomy_failed_source", None)

            if judge is None:
                from .crash_review import CosmosCrashJudge

                judge = CosmosCrashJudge()

            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            classified = 0
            retryable_errors = 0
            terminal_errors = 0

            with tempfile.TemporaryDirectory(
                prefix="crash-taxonomy-",
                dir=settings.DATA_DIR,
            ) as temporary:
                work = Path(temporary)

                for position, segment in enumerate(segments, start=1):
                    try:
                        result = _review_one_segment(
                            judge,
                            video_path,
                            video_id,
                            segment,
                            metadata,
                            work,
                        )
                    except KeyboardInterrupt:
                        save_state(settings.STATE_JSON, state)
                        raise
                    except Exception as exc:
                        result = {
                            "status": "model_error",
                            "error": f"sample_error: {exc}",
                            "raw_response": "",
                        }

                    if result.get("status") == "model_error":
                        terminal = _store_model_error(segment, result)
                        if terminal:
                            terminal_errors += 1
                        else:
                            retryable_errors += 1
                    else:
                        segment.update(result)
                        segment["crash_taxonomy_review_cycles"] = 0
                        classified += 1

                    if (
                        position == 1
                        or position % 10 == 0
                        or position == len(segments)
                    ):
                        log(
                            f"Taxonomy progress for {video_id}: "
                            f"{position}/{len(segments)}"
                        )

            if retryable_errors:
                record["crash_taxonomy_status"] = "model_error"
                record["crash_taxonomy_error"] = (
                    f"{retryable_errors} segment taxonomy reviews need retry"
                )
                record.pop("crash_taxonomy_version", None)
            elif terminal_errors and not classified:
                record["crash_taxonomy_status"] = "model_error_terminal"
                record["crash_taxonomy_error"] = (
                    f"{terminal_errors} segment taxonomy reviews were "
                    "terminally skipped"
                )
                record["crash_taxonomy_version"] = CRASH_TAXONOMY_VERSION
            elif terminal_errors:
                record["crash_taxonomy_status"] = "partial"
                record["crash_taxonomy_error"] = (
                    f"{terminal_errors} segment taxonomy reviews were "
                    "terminally skipped"
                )
                record["crash_taxonomy_version"] = CRASH_TAXONOMY_VERSION
            else:
                record["crash_taxonomy_status"] = "classified"
                record["crash_taxonomy_error"] = None
                record["crash_taxonomy_version"] = CRASH_TAXONOMY_VERSION

            # The source is already available and Cosmos is already loaded.
            # Reuse both to fill any missing high-resolution visual location
            # evidence before deciding whether the source can be deleted.
            try:
                from .crash_review import review_missing_segment_locations

                location_reviewed = review_missing_segment_locations(
                    video_id,
                    record,
                    judge,
                    video_path,
                )
                if location_reviewed:
                    log(
                        f"Location visual backfill for {video_id}: "
                        f"{location_reviewed} segment(s)"
                    )
            except KeyboardInterrupt:
                save_state(settings.STATE_JSON, state)
                raise
            except Exception as exc:
                # Location recovery is useful but must not roll back a valid
                # taxonomy classification. The retained source can be retried
                # by the normal location visual stage when appropriate.
                log(
                    f"Location visual backfill failed for {video_id}: {exc}"
                )

            save_state(settings.STATE_JSON, state)
            _cleanup_taxonomy_video_if_safe(record, video_path)
            save_state(settings.STATE_JSON, state)
            processed += len(segments)

    finally:
        if judge is not None:
            unload_model(judge)

    return processed
