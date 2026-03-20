from __future__ import annotations

import unittest

import pandas as pd

from src.preprocessing import (
    PriorityPreprocessor,
    QueuePreprocessor,
    TextPreparationPipeline,
)


class TextPreparationPipelineTests(unittest.TestCase):
    def test_missing_body_column_raises_key_error(self) -> None:
        pipeline = TextPreparationPipeline()

        with self.assertRaisesRegex(KeyError, "Missing text column: body"):
            pipeline.transform(pd.DataFrame({"subject": ["printer issue"]}))

    def test_missing_subject_column_is_tolerated(self) -> None:
        pipeline = TextPreparationPipeline()

        transformed = pipeline.transform(
            pd.DataFrame({"body": ["Printer error on floor 2"]})
        )

        self.assertEqual(
            transformed[pipeline.combined_column].tolist(),
            ["Printer error on floor 2"],
        )

    def test_normalize_text_replaces_url_email_and_number_stably(self) -> None:
        normalized = TextPreparationPipeline._normalize_text(
            "Contact John_Doe@example.com at https://example.com about ticket #42.",
            language="unknown",
        )

        self.assertEqual(normalized, "contact email at url about ticket number")


class TargetPreprocessorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue_frame = pd.DataFrame(
            {
                "subject": ["printer issue", "billing help"],
                "body": ["printer error on floor 2", "need refund for invoice"],
                "queue": ["Technical Support", "Billing and Payments"],
            }
        )
        self.priority_frame = pd.DataFrame(
            {
                "subject": ["slow login", "invoice issue", "system outage"],
                "body": [
                    "users can still access the portal",
                    "customer needs a billing review",
                    "all customers are blocked",
                ],
                "priority": ["low", "medium", "high"],
                "language": ["unknown", "unknown", "unknown"],
            }
        )

    def test_queue_preprocessor_falls_back_when_language_column_is_missing(self) -> None:
        preprocessor = QueuePreprocessor()

        dataset = preprocessor.fit_transform(self.queue_frame)
        transformed = preprocessor.transform(self.queue_frame)

        self.assertEqual(dataset.X.shape[1], len(dataset.feature_names))
        self.assertEqual(transformed.shape[1], len(dataset.feature_names))

    def test_priority_preprocessor_uses_ordered_target_encoding(self) -> None:
        preprocessor = PriorityPreprocessor()

        dataset = preprocessor.fit_transform(self.priority_frame)

        self.assertEqual(dataset.y.tolist(), [0, 1, 2])
        self.assertEqual(dataset.target_mapping, {0: "low", 1: "medium", 2: "high"})

    def test_task_preprocessors_do_not_add_length_feature_by_default(self) -> None:
        queue_dataset = QueuePreprocessor().fit_transform(
            self.queue_frame.assign(language="unknown")
        )
        priority_dataset = PriorityPreprocessor().fit_transform(self.priority_frame)

        self.assertNotIn("ticket_text_length", queue_dataset.feature_names)
        self.assertNotIn("ticket_text_length", priority_dataset.feature_names)


if __name__ == "__main__":
    unittest.main()
