"""The VLM prompts must not show a filled-in answer the model could copy."""

import inspect
import unittest

from car_crash_pipeline import crash_review


class ExamplePromptTest(unittest.TestCase):
    def test_cut_check_prompt_has_no_example_answer(self) -> None:
        source = inspect.getsource(crash_review.CosmosCrashJudge.verify_boundary)
        self.assertNotIn('"confidence": 0.95', source)
        self.assertNotIn('"is_edit_boundary": true', source)
        self.assertIn('"confidence": <number between 0 and 1>', source)

    def test_segment_prompt_has_no_example_answer(self) -> None:
        prompt = crash_review.CosmosCrashJudge._segment_prompt({"title": "t"}, 10.0)
        self.assertNotIn('"confidence": 0.95', prompt)
        self.assertNotIn('"is_crash": true', prompt)

    def test_old_reviews_kept_unless_redo_is_switched_on(self) -> None:
        if crash_review.settings.REDO_EXAMPLE_PROMPT_REVIEWS:
            self.assertEqual(crash_review.UP_TO_DATE_REVIEW_VERSIONS, {crash_review.CRASH_REVIEW_VERSION})
        else:
            self.assertIn("cosmos3_full_clip_crash_v3", crash_review.UP_TO_DATE_REVIEW_VERSIONS)
            self.assertIn("cosmos3_full_clip_crash_v4", crash_review.UP_TO_DATE_REVIEW_VERSIONS)


if __name__ == "__main__":
    unittest.main()
