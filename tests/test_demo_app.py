from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None
REQUESTS_AVAILABLE = importlib.util.find_spec("requests") is not None
SERVING_CONFIG_EXISTS = Path("serving_assets/serving_config.json").exists()

if FASTAPI_AVAILABLE and HTTPX_AVAILABLE and SERVING_CONFIG_EXISTS:
    from fastapi.testclient import TestClient

    from app.api import create_app
    from app.service import TicketRoutingService


@unittest.skipUnless(
    FASTAPI_AVAILABLE and HTTPX_AVAILABLE and SERVING_CONFIG_EXISTS,
    "fastapi/httpx or serving assets are not available",
)
class DemoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = TicketRoutingService.from_config(Path("serving_assets/serving_config.json"))
        cls.client = TestClient(create_app(cls.service))

    def test_health_endpoint_reports_loaded_models(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(set(payload["tasks"]), {"queue", "priority"})
        self.assertEqual(set(payload["models"]), {"queue", "priority"})

    def test_demo_ticket_endpoint_returns_usable_payload(self) -> None:
        response = self.client.get("/demo-ticket", params={"index": 2})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("title", payload)
        self.assertEqual(set(payload["ticket"]), {"subject", "body", "language"})
        self.assertGreater(payload["total"], 0)

    def test_predict_endpoint_returns_queue_and_priority_with_runner_up_and_gap(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "subject": "Critical security incident affecting the account portal",
                "body": "Users report suspicious activity and service disruption after repeated failed logins.",
                "language": "en",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload["predictions"]), {"queue", "priority"})
        self.assertEqual(set(payload["input"]), {"subject", "body", "language"})
        for task_name in ("queue", "priority"):
            task_payload = payload["predictions"][task_name]
            self.assertIn("runner_up_label", task_payload)
            self.assertIsInstance(task_payload["margin_gap"], float)
            self.assertTrue(task_payload["label"])
            self.assertTrue(task_payload["runner_up_label"])
            self.assertNotEqual(task_payload["label"], task_payload["runner_up_label"])

    def test_predict_endpoint_without_language_still_works(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "subject": "Need help with invoice mismatch",
                "body": "The renewal invoice looks wrong and we need a billing review.",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload["input"]), {"subject", "body"})

    def test_service_prediction_supports_german_language(self) -> None:
        payload = self.service.predict_ticket(
            subject="Dringender Sicherheitsvorfall im Kundenportal",
            body="Mehrere Nutzer melden verdächtige Anmeldeversuche und den Ausfall des Kundenportals.",
            language="de",
        )
        self.assertEqual(payload["input"]["language"], "de")
        self.assertEqual(set(payload["predictions"]), {"queue", "priority"})

    def test_predict_endpoint_validates_missing_fields(self) -> None:
        response = self.client.post("/predict", json={"subject": "only subject"})
        self.assertEqual(response.status_code, 422)

    def test_service_prediction_returns_model_metadata(self) -> None:
        payload = self.service.predict_ticket(
            subject="Need help with invoice mismatch",
            body="The renewal invoice looks wrong and we need a billing review.",
            language="en",
        )
        self.assertEqual(set(payload["models"]), {"queue", "priority"})
        self.assertEqual(set(payload["predictions"]), {"queue", "priority"})
        self.assertEqual(payload["input"]["language"], "en")
        for metadata in payload["models"].values():
            self.assertEqual(
                set(metadata),
                {"run_id", "algorithm", "model_family", "feature_families"},
            )


class ServingImportIsolationTests(unittest.TestCase):
    def test_app_service_import_and_model_load_do_not_require_mlflow(self) -> None:
        original_mlflow = sys.modules.pop("mlflow", None)
        try:
            from app.service import TicketRoutingService

            service = TicketRoutingService.from_config(Path("serving_assets/serving_config.json"))
            self.assertEqual(set(service.models), {"queue", "priority"})
        finally:
            if original_mlflow is not None:
                sys.modules["mlflow"] = original_mlflow


@unittest.skipUnless(REQUESTS_AVAILABLE, "requests is not installed")
class DemoUiHelperTests(unittest.TestCase):
    def test_ui_helpers_module_imports(self) -> None:
        import app.ui as ui

        self.assertTrue(ui.get_api_base_url())
        self.assertTrue(callable(ui.fetch_demo_ticket))
        self.assertTrue(callable(ui.request_prediction))


if __name__ == "__main__":
    unittest.main()
