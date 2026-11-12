"""Regression tests for NHTSA MMUCC crash taxonomy backfill."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from car_crash_pipeline.crash_taxonomy import (
    CRASH_TAXONOMY_STANDARD,
    CRASH_TAXONOMY_VERSION,
    crash_taxonomy_pending,
    normalise_taxonomy_response,
    repair_taxonomy_response,
    run_crash_taxonomy_stage,
    DETAIL_PROMPT,
    LOCATE_PROMPT,
    _review_one_segment,
    manner_from_parts,
    taxonomy_from_contact,
    validate_taxonomy_response,
)
from car_crash_pipeline.output_writer import iter_mapping_rows, iter_rows


class CrashTaxonomyTests(unittest.TestCase):
    def test_motor_vehicle_collision_uses_mmucc_c7_and_c9(self) -> None:
        data = {
            "impact_description": "The front of a car strikes the rear of a van.",
            "event_kind": "collision",
            "crash_type": "rear_end",
            "manner_of_collision": "front_to_rear_or_rear_to_front",
            "first_harmful_event": "motor_vehicle_in_transport",
            "sequence_of_events": ["motor_vehicle_in_transport"],
        }

        self.assertIsNone(validate_taxonomy_response(data))

        result = normalise_taxonomy_response(data)

        self.assertEqual(result["event_kind"], "collision")
        self.assertEqual(
            result["manner_of_collision_code"],
            "C9_FRONT_TO_REAR_OR_REAR_TO_FRONT",
        )
        self.assertEqual(
            result["first_harmful_event_code"],
            "C7_MOTOR_VEHICLE_IN_TRANSPORT",
        )
        self.assertEqual(
            result["sequence_of_events"],
            ["motor_vehicle_in_transport"],
        )
        self.assertEqual(
            result["crash_taxonomy_standard"],
            CRASH_TAXONOMY_STANDARD,
        )
        self.assertEqual(
            result["crash_taxonomy_version"],
            CRASH_TAXONOMY_VERSION,
        )

    def test_non_motor_vehicle_first_event_requires_not_collision_manner(self) -> None:
        valid = {
            "impact_description": "A car strikes a pedestrian crossing the road.",
            "event_kind": "collision",
            "crash_type": "pedestrian",
            "manner_of_collision": (
                "not_collision_with_motor_vehicle_in_transport"
            ),
            "first_harmful_event": "non_motorist",
            "sequence_of_events": ["non_motorist"],
        }
        invalid = dict(valid)
        invalid["manner_of_collision"] = "angle"

        self.assertIsNone(validate_taxonomy_response(valid))
        self.assertEqual(
            validate_taxonomy_response(invalid),
            "non_motor_vehicle_first_harmful_event_requires_not_collision_manner",
        )

    def test_crash_type_must_agree_with_event_kind(self) -> None:
        data = {
            "impact_description": "The front of a car strikes the rear of a van.",
            "event_kind": "collision",
            "crash_type": "near_collision",
            "manner_of_collision": "front_to_rear_or_rear_to_front",
            "first_harmful_event": "motor_vehicle_in_transport",
            "sequence_of_events": ["motor_vehicle_in_transport"],
        }

        self.assertEqual(
            validate_taxonomy_response(data),
            "crash_type_near_collision_must_match_event_kind",
        )

    def test_prompts_ask_plain_questions_not_mmucc_codes(self) -> None:
        self.assertIn("contact_time", LOCATE_PROMPT)
        for key in ("contact", "part_a", "part_b", "directions"):
            self.assertIn(f'"{key}"', DETAIL_PROMPT)
        self.assertNotIn("front_to_rear_or_rear_to_front", DETAIL_PROMPT)

    def test_manner_follows_from_the_parts_that_touched(self) -> None:
        cases = {
            ("front", "rear", "same"): "front_to_rear_or_rear_to_front",
            ("front", "front", "opposite"): "front_to_front",
            ("front", "front", "crossing"): "angle",
            ("front", "left_side", "crossing"): "angle",
            ("left_side", "right_side", "same"): "sideswipe_same_direction",
            ("left_side", "left_side", "opposite"): "sideswipe_opposite_direction",
            ("rear", "right_side", "crossing"): "rear_to_side_or_side_to_rear",
            ("rear", "rear", "opposite"): "rear_to_rear",
        }
        for (a, b, directions), manner in cases.items():
            self.assertEqual(manner_from_parts(a, b, directions), manner, (a, b, directions))

    def test_contact_answers_become_valid_mmucc_fields(self) -> None:
        answers = [
            {"contact": "vehicle_vehicle", "part_a": "front", "part_b": "rear", "directions": "same"},
            {"contact": "vehicle_object", "hit": "tree"},
            {"contact": "vehicle_person", "hit": "pedestrian"},
            {"contact": "rollover"},
            {"contact": "none"},
        ]
        for answer in answers:
            data = repair_taxonomy_response(taxonomy_from_contact({**answer, "description": "x"}))
            self.assertIsNone(validate_taxonomy_response(data), answer)
        self.assertEqual(
            taxonomy_from_contact({"contact": "vehicle_object", "hit": "tree"})["first_harmful_event"],
            "tree_standing_only",
        )
        # The model sometimes leaves the description out.
        for answer in ({"contact": "none"}, {"contact": "vehicle_object", "hit": "tree"}):
            self.assertIsNone(validate_taxonomy_response(repair_taxonomy_response(taxonomy_from_contact(answer))))

    def test_earliest_window_with_contact_decides(self) -> None:
        answers = iter([
            '{"contact_time": 4.0, "what_happens": "x"}',
            '{"contact": "none", "description": "a close pass"}',
            '{"contact": "vehicle_vehicle", "part_a": "front", "part_b": "left_side", '
            '"directions": "crossing", "description": "a car is hit on its side"}',
        ])

        class Judge:
            def _generate(self, path, prompt, fps):
                return next(answers)

        with (
            patch("car_crash_pipeline.crash_taxonomy._taxonomy_sample",
                  side_effect=lambda video, start, duration, out, **kw: out),
            patch("car_crash_pipeline.crash_taxonomy._motion_peaks", return_value=[1.0]),
        ):
            result = _review_one_segment(
                Judge(), Path("video.mp4"), "vid",
                {"start_time": 10.0, "end_time": 20.0, "segment_index": 0}, {}, Path("/tmp"),
            )

        self.assertEqual(result["event_kind"], "collision")
        self.assertEqual(result["manner_of_collision"], "angle")
        self.assertEqual(result["crash_taxonomy_impact_source"], "motion")
        self.assertEqual(result["crash_taxonomy_impact_time"], 11.0)

    def test_near_collision_has_no_mmucc_harmful_event(self) -> None:
        data = {
            "impact_description": "A car brakes hard and stops short of a truck.",
            "event_kind": "near_collision",
            "crash_type": "near_collision",
            "manner_of_collision": None,
            "first_harmful_event": None,
            "sequence_of_events": [],
        }

        self.assertIsNone(validate_taxonomy_response(data))

        result = normalise_taxonomy_response(data)

        self.assertEqual(result["event_kind"], "near_collision")
        self.assertIsNone(result["manner_of_collision"])
        self.assertIsNone(result["first_harmful_event"])
        self.assertEqual(result["sequence_of_events"], [])

    def test_sequence_keeps_non_harmful_precursor_before_first_harmful_event(
        self,
    ) -> None:
        data = {
            "impact_description": "A car leaves the road and rolls over.",
            "event_kind": "collision",
            "crash_type": "rollover",
            "manner_of_collision": (
                "not_collision_with_motor_vehicle_in_transport"
            ),
            "first_harmful_event": "rollover_overturn",
            "sequence_of_events": [
                "ran_off_roadway_right",
                "rollover_overturn",
            ],
        }

        self.assertIsNone(validate_taxonomy_response(data))

    def test_repair_drops_free_text_and_maps_synonyms_in_sequence(self) -> None:
        data = repair_taxonomy_response({
            "impact_description": "A car leaves the road and rolls over.",
            "event_kind": "collision",
            "crash_type": "rollover",
            "manner_of_collision": "not_collision_with_motor_vehicle_in_transport",
            "first_harmful_event": "rollover_overturn",
            "sequence_of_events": [
                "the dashcam vehicle brakes hard to avoid a collision.",
                "ran_off_roadway_right",
                "rollover",
                "truck",
            ],
        })

        self.assertEqual(
            data["sequence_of_events"], ["ran_off_roadway_right", "rollover_overturn"]
        )
        self.assertIsNone(validate_taxonomy_response(data))

    def test_repair_clears_mmucc_fields_of_a_near_collision(self) -> None:
        data = repair_taxonomy_response({
            "impact_description": "A car swerves and avoids a truck.",
            "event_kind": "near_collision",
            "crash_type": "near_collision",
            "manner_of_collision": "unknown",
            "first_harmful_event": "unknown",
            "sequence_of_events": ["the car swerves to avoid a collision."],
        })

        self.assertIsNone(validate_taxonomy_response(data))
        self.assertEqual(data["sequence_of_events"], [])

    def test_unseen_impact_gives_unknown_manner_not_non_vehicle(self) -> None:
        data = repair_taxonomy_response({
            "impact_description": "The car's hood is crushed after an impact.",
            "event_kind": "collision",
            "crash_type": "unknown",
            "manner_of_collision": "not_collision_with_motor_vehicle_in_transport",
            "first_harmful_event": "unknown",
            "sequence_of_events": [],
        })

        self.assertIsNone(validate_taxonomy_response(data))
        result = normalise_taxonomy_response(data)
        self.assertEqual(result["manner_of_collision"], "unknown")
        self.assertEqual(result["sequence_of_events"], ["unknown"])

    def test_seen_vehicle_orientation_implies_vehicle_first_harmful_event(self) -> None:
        data = repair_taxonomy_response({
            "impact_description": "A car turning left is struck on its side.",
            "event_kind": "collision",
            "crash_type": "side_impact",
            "manner_of_collision": "angle",
            "first_harmful_event": "unknown",
            "sequence_of_events": ["cross_centerline"],
        })

        self.assertIsNone(validate_taxonomy_response(data))
        self.assertEqual(data["first_harmful_event"], "motor_vehicle_in_transport")
        self.assertEqual(
            data["sequence_of_events"], ["cross_centerline", "motor_vehicle_in_transport"]
        )

    def test_old_completed_segment_is_pending_until_backfilled(self) -> None:
        record = {
            "status": "complete",
            "segments": [{"segment_index": 0}],
        }

        self.assertTrue(crash_taxonomy_pending(record))

        record["segments"][0]["crash_taxonomy_version"] = (
            CRASH_TAXONOMY_VERSION
        )

        self.assertFalse(crash_taxonomy_pending(record))

    def test_unavailable_old_video_is_terminally_skipped(self) -> None:
        state = {
            "videos": {
                "missingVideo": {
                    "status": "complete",
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "duration_seconds": 1.0,
                        }
                    ],
                }
            }
        }

        with (
            patch(
                "car_crash_pipeline.crash_taxonomy._obtain_video",
                side_effect=RuntimeError("Video unavailable"),
            ),
            patch(
                "car_crash_pipeline.crash_taxonomy.save_state"
            ),
        ):
            processed = run_crash_taxonomy_stage(state)

        segment = state["videos"]["missingVideo"]["segments"][0]

        self.assertEqual(processed, 1)
        self.assertEqual(
            segment["crash_taxonomy_status"],
            "video_unavailable",
        )
        self.assertEqual(
            segment["crash_taxonomy_version"],
            CRASH_TAXONOMY_VERSION,
        )
        self.assertIsNone(segment["event_kind"])
        self.assertFalse(
            crash_taxonomy_pending(state["videos"]["missingVideo"])
        )

    def test_any_download_failure_is_skipped_immediately(self) -> None:
        state = {
            "videos": {
                "temporaryFailure": {
                    "status": "complete",
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "duration_seconds": 1.0,
                        }
                    ],
                }
            }
        }

        with (
            patch(
                "car_crash_pipeline.crash_taxonomy._obtain_video",
                side_effect=RuntimeError("HTTP Error 429: Too Many Requests"),
            ),
            patch("car_crash_pipeline.crash_taxonomy.save_state"),
        ):
            processed = run_crash_taxonomy_stage(state, max_videos=1)

        record = state["videos"]["temporaryFailure"]
        segment = record["segments"][0]

        self.assertEqual(processed, 1)
        self.assertEqual(record["crash_taxonomy_status"], "download_failed")
        self.assertEqual(segment["crash_taxonomy_status"], "download_failed")
        self.assertEqual(segment["crash_taxonomy_version"], CRASH_TAXONOMY_VERSION)
        self.assertFalse(crash_taxonomy_pending(record))

    def test_skipped_download_is_retried_once_a_shared_copy_exists(self) -> None:
        state = {
            "videos": {
                "abcdefghijk": {
                    "status": "complete",
                    "metadata": {"youtube_url": "https://www.youtube.com/watch?v=abcdefghijk"},
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "duration_seconds": 1.0,
                            "event_kind": "collision",
                        }
                    ],
                }
            }
        }
        record = state["videos"]["abcdefghijk"]

        with tempfile.TemporaryDirectory() as folder:
            shared = Path(folder)
            with (
                patch("car_crash_pipeline.crash_taxonomy.settings.SHARED_VIDEO_DIRS", [shared]),
                patch(
                    "car_crash_pipeline.crash_taxonomy._obtain_video",
                    side_effect=RuntimeError("Sign in to confirm you're not a bot"),
                ),
                patch("car_crash_pipeline.crash_taxonomy.save_state"),
            ):
                run_crash_taxonomy_stage(state, max_videos=1)
                self.assertEqual(record["crash_taxonomy_status"], "download_failed")
                # No copy anywhere: skipped, so it does not hold up discovery.
                self.assertFalse(crash_taxonomy_pending(record))

                (shared / "abcdefghijk.mp4").write_bytes(b"video")
                self.assertTrue(crash_taxonomy_pending(record))

                # The copy fails too: it is not retried until it changes.
                run_crash_taxonomy_stage(state, max_videos=1)
                self.assertFalse(crash_taxonomy_pending(record))
                self.assertEqual(record["segments"][0]["event_kind"], "collision")

                (shared / "abcdefghijk.mp4").write_bytes(b"a longer video")
                self.assertTrue(crash_taxonomy_pending(record))

    def test_legacy_cooldown_record_is_retried_once_then_skipped(self) -> None:
        state = {
            "videos": {
                "coolingDown": {
                    "status": "complete",
                    "crash_taxonomy_status": "download_error",
                    "crash_taxonomy_download_retry_after": 9e18,
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "duration_seconds": 1.0,
                        }
                    ],
                },
                "nextVideo": {
                    "status": "complete",
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "duration_seconds": 1.0,
                        }
                    ],
                },
            }
        }

        with (
            patch(
                "car_crash_pipeline.crash_taxonomy._obtain_video",
                side_effect=RuntimeError("Sorry, this content is age-restricted"),
            ) as obtain_video,
            patch("car_crash_pipeline.crash_taxonomy.save_state"),
        ):
            first = run_crash_taxonomy_stage(state, max_videos=1)
            second = run_crash_taxonomy_stage(state, max_videos=1)

        self.assertEqual((first, second), (1, 1))
        self.assertEqual(
            [call.args[0] for call in obtain_video.call_args_list],
            ["coolingDown", "nextVideo"],
        )
        cooling = state["videos"]["coolingDown"]
        self.assertEqual(cooling["crash_taxonomy_status"], "age_restricted")
        self.assertNotIn("crash_taxonomy_download_retry_after", cooling)
        self.assertFalse(crash_taxonomy_pending(cooling))
        self.assertFalse(crash_taxonomy_pending(state["videos"]["nextVideo"]))

    def test_csv_outputs_include_taxonomy(self) -> None:
        state = {
            "videos": {
                "videoA": {
                    "status": "complete",
                    "visual_review_version": "test",
                    "metadata": {},
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 10.0,
                            "end_time": 12.0,
                            "duration_seconds": 2.0,
                            "confidence": 0.99,
                            "short_description": "rear impact",
                            "crash_type": "rear_end",
                            "road_users": ["car"],
                            "time_of_day": "day",
                            "event_kind": "collision",
                            "manner_of_collision_code": (
                                "C9_FRONT_TO_REAR_OR_REAR_TO_FRONT"
                            ),
                            "manner_of_collision": (
                                "front_to_rear_or_rear_to_front"
                            ),
                            "first_harmful_event_code": (
                                "C7_MOTOR_VEHICLE_IN_TRANSPORT"
                            ),
                            "first_harmful_event": (
                                "motor_vehicle_in_transport"
                            ),
                            "sequence_of_events": [
                                "motor_vehicle_in_transport"
                            ],
                            "crash_taxonomy_standard": (
                                CRASH_TAXONOMY_STANDARD
                            ),
                            "crash_taxonomy_status": "classified",
                            "crash_taxonomy_version": (
                                CRASH_TAXONOMY_VERSION
                            ),
                            "location": {
                                "locality": "Test City",
                                "locality_aka": [],
                                "state": None,
                                "country": "Test Country",
                                "iso3": "TST",
                                "continent": "Test",
                                "lat": 1.0,
                                "lon": 2.0,
                                "osm_type": "relation",
                                "osm_id": 123,
                                "place_id": 456,
                                "geocode_status": "resolved",
                                "location_resolution_version": "segment_evidence_location_v7",
                            },
                        }
                    ],
                }
            }
        }

        segment_row = list(iter_rows(state))[0]
        mapping_row = list(iter_mapping_rows(state))[0]

        self.assertEqual(segment_row["event_kind"], "collision")
        self.assertEqual(
            segment_row["first_harmful_event"],
            "motor_vehicle_in_transport",
        )
        self.assertEqual(
            segment_row["sequence_of_events"],
            '["motor_vehicle_in_transport"]',
        )

        self.assertEqual(mapping_row["event_kind"], "[collision]")
        self.assertEqual(
            mapping_row["manner_of_collision"],
            "[front_to_rear_or_rear_to_front]",
        )
        self.assertEqual(
            mapping_row["sequence_of_events"],
            "[[motor_vehicle_in_transport]]",
        )
        self.assertEqual(
            mapping_row["crash_taxonomy_status"],
            "[classified]",
        )

    def test_taxonomy_backfill_can_be_limited_to_one_video(self) -> None:
        state = {
            "videos": {
                "first": {
                    "status": "complete",
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "duration_seconds": 1.0,
                        }
                    ],
                },
                "second": {
                    "status": "complete",
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 3.0,
                            "end_time": 4.0,
                            "duration_seconds": 1.0,
                        }
                    ],
                },
            }
        }

        with (
            patch(
                "car_crash_pipeline.crash_taxonomy._obtain_video",
                side_effect=RuntimeError("Video unavailable"),
            ),
            patch("car_crash_pipeline.crash_taxonomy.save_state"),
        ):
            processed = run_crash_taxonomy_stage(state, max_videos=1)

        self.assertEqual(processed, 1)
        self.assertEqual(
            state["videos"]["first"]["segments"][0][
                "crash_taxonomy_version"
            ],
            CRASH_TAXONOMY_VERSION,
        )
        self.assertNotIn(
            "crash_taxonomy_version",
            state["videos"]["second"]["segments"][0],
        )


if __name__ == "__main__":
    unittest.main()
