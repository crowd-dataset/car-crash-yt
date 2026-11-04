"""Tests for reusing videos downloaded elsewhere and for taxonomy re-runs."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from car_crash_pipeline import settings
from car_crash_pipeline.crash_review import download_video
from car_crash_pipeline.crash_taxonomy import run_crash_taxonomy_stage


AGE_ERROR = "ERROR: [youtube] abc: Sorry, this content is age-restricted"


class SharedVideoTests(unittest.TestCase):
    def test_shared_copy_is_used_instead_of_downloading(self) -> None:
        with TemporaryDirectory() as temporary:
            shared = Path(temporary) / "drive"
            shared.mkdir()
            (shared / "abc.mp4").write_bytes(b"video")
            videos = Path(temporary) / "videos"

            with (
                patch.object(settings, "SHARED_VIDEO_DIRS", [shared]),
                patch.object(settings, "VIDEO_DIR", videos),
                patch.object(settings, "TEMP_DIR", videos / ".tmp"),
                patch(
                    "car_crash_pipeline.crash_review._valid_video",
                    return_value=True,
                ),
                patch("car_crash_pipeline.crash_review.run_command") as command,
            ):
                path = download_video("abc")

            command.assert_not_called()
            self.assertEqual(path, videos / "abc.mp4")
            self.assertEqual(path.read_bytes(), b"video")
            # Deleting the processing copy leaves the shared file in place.
            path.unlink()
            self.assertTrue((shared / "abc.mp4").is_file())


class TaxonomyRerunTests(unittest.TestCase):
    def test_unavailable_video_keeps_earlier_event_kind(self) -> None:
        state = {
            "videos": {
                "gone": {
                    "status": "complete",
                    "text_decision": {"include": True},
                    "visual_review_version": "cosmos3_full_clip_crash_v3",
                    "segments": [
                        {
                            "segment_index": 0,
                            "start_time": 1.0,
                            "end_time": 2.0,
                            "event_kind": "near_collision",
                            "crash_taxonomy_version": "nhtsa_mmucc6_video_v1",
                        }
                    ],
                }
            }
        }

        with (
            patch(
                "car_crash_pipeline.crash_taxonomy._obtain_video",
                side_effect=RuntimeError(AGE_ERROR),
            ),
            patch("car_crash_pipeline.crash_taxonomy.save_state"),
        ):
            run_crash_taxonomy_stage(state, max_videos=1)

        segment = state["videos"]["gone"]["segments"][0]
        self.assertEqual(segment["crash_taxonomy_status"], "age_restricted")
        self.assertEqual(segment["event_kind"], "near_collision")
        self.assertIsNone(segment["manner_of_collision"])


if __name__ == "__main__":
    unittest.main()
