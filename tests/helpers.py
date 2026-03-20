from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


QUEUE_LABELS = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]
PRIORITY_LABELS = ["low", "medium", "high"]



def build_smoke_dataset(csv_path: Path, repeats: int = 3) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "body", "queue", "priority", "language", "version"],
        )
        writer.writeheader()
        for index, queue_label in enumerate(QUEUE_LABELS):
            priority_label = PRIORITY_LABELS[index % len(PRIORITY_LABELS)]
            for repeat_index in range(repeats):
                writer.writerow(
                    {
                        "subject": f"subject {queue_label} {repeat_index}",
                        "body": f"body {queue_label} {priority_label} sample {repeat_index}",
                        "queue": queue_label,
                        "priority": priority_label,
                        "language": "en" if (index + repeat_index) % 2 == 0 else "de",
                        "version": "smoke",
                    }
                )



def run_training_smoke(
    repo_root: Path,
    data_path: Path,
    tracking_uri: str,
    experiment_name: str,
    algorithm: str = "logreg",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "train.py",
            "--data",
            str(data_path),
            "--cv-folds",
            "3",
            "--algorithm",
            algorithm,
            "--tracking-uri",
            tracking_uri,
            "--experiment-name",
            experiment_name,
            "--run-group",
            "smoke-test",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
