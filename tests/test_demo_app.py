from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.api import TicketRequest, create_app
from app.service import TicketRoutingService


MODEL_METADATA = {
    "run_id": "test-run",
    "algorithm": "linear_svc",
    "model_family": "LinearSVC",
    "c": 1.0,
    "feature_summary": "TF-IDF word 1-3 grams",
    "dataset_id": "test-dataset",
    "cv_macro_f1_mean": 0.7,
    "cv_accuracy_mean": 0.7,
}


class _FakeService:
    title = "Test Ticket Triage"

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "tasks": ["queue", "priority"],
            "models": {
                "queue": MODEL_METADATA,
                "priority": MODEL_METADATA,
            },
        }

    def predict_ticket(
        self, *, subject: str, body: str, language: str | None = None
    ) -> dict[str, Any]:
        return {
            "input": {"subject": subject, "body": body, "language": language},
            "predictions": {
                "queue": {
                    "label": "Technical Support",
                    "runner_up_label": "Product Support",
                    "margin_gap": 1.0,
                },
                "priority": {
                    "label": "high",
                    "runner_up_label": "medium",
                    "margin_gap": 0.5,
                },
            },
            "models": {
                "queue": MODEL_METADATA,
                "priority": MODEL_METADATA,
            },
        }


class DemoApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app = create_app(_FakeService())
        self.endpoints = {route.path: route.endpoint for route in app.routes}

    def test_health_endpoint_reports_loaded_models(self) -> None:
        payload = self.endpoints["/health"]()

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(set(payload["tasks"]), {"queue", "priority"})
        self.assertEqual(set(payload["models"]), {"queue", "priority"})

    def test_demo_ticket_endpoint_returns_usable_payload(self) -> None:
        payload = self.endpoints["/demo-ticket"](2)

        self.assertIn("title", payload)
        self.assertEqual(set(payload["ticket"]), {"subject", "body", "language"})
        self.assertGreater(payload["total"], 0)

    def test_predict_endpoint_returns_queue_and_priority_with_runner_up_and_gap(
        self,
    ) -> None:
        request = TicketRequest(
            subject="Critical security incident affecting the account portal",
            body="Users report suspicious activity and service disruption.",
            language="en",
        )
        payload = self.endpoints["/predict"](request)

        self.assertEqual(set(payload["predictions"]), {"queue", "priority"})
        self.assertEqual(set(payload["input"]), {"subject", "body", "language"})
        self.assertEqual(payload["input"]["language"], "en")

    def test_predict_endpoint_validates_missing_fields(self) -> None:
        with self.assertRaises(ValidationError):
            TicketRequest(subject="only subject")


class ServingModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = TicketRoutingService.from_config(
            Path("serving_assets/serving_config.json")
        )

    def test_checked_in_models_predict_and_report_metadata(self) -> None:
        payload = self.service.predict_ticket(
            subject="Need help with an invoice mismatch",
            body="The renewal invoice looks wrong and needs a billing review.",
            language="en",
        )

        self.assertEqual(set(payload["predictions"]), {"queue", "priority"})
        for task_name in ("queue", "priority"):
            prediction = payload["predictions"][task_name]
            self.assertTrue(prediction["label"])
            self.assertTrue(prediction["runner_up_label"])
            self.assertIsInstance(prediction["margin_gap"], float)
            self._assert_model_metadata_shape(payload["models"][task_name])

    def _assert_model_metadata_shape(self, metadata: dict[str, object]) -> None:
        self.assertEqual(
            set(metadata),
            {
                "run_id",
                "algorithm",
                "model_family",
                "c",
                "feature_summary",
                "dataset_id",
                "cv_macro_f1_mean",
                "cv_accuracy_mean",
            },
        )
        self.assertIsInstance(metadata["c"], float)
        self.assertIsInstance(metadata["cv_macro_f1_mean"], float)
        self.assertIsInstance(metadata["cv_accuracy_mean"], float)
        self.assertTrue(metadata["feature_summary"])
        self.assertTrue(metadata["dataset_id"])


class DemoUiHelperTests(unittest.TestCase):
    def test_ui_prediction_payload_requires_supported_language(self) -> None:
        import app.ui as ui

        payload = ui.build_prediction_payload(
            subject="Invoice issue",
            body="Please review this invoice.",
            language="en",
        )

        self.assertEqual(payload["language"], "en")
        for unsupported_language in (None, "fr"):
            with self.subTest(language=unsupported_language):
                with self.assertRaisesRegex(ValueError, "Select English or German"):
                    ui.build_prediction_payload(
                        subject="Invoice issue",
                        body="Please review this invoice.",
                        language=unsupported_language,
                    )


if __name__ == "__main__":
    unittest.main()
