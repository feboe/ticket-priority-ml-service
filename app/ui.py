"""Streamlit UI for the ticket triage MVP."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def get_api_base_url() -> str:
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def fetch_demo_ticket(index: int, base_url: str | None = None) -> dict[str, Any]:
    response = requests.get(
        f"{base_url or get_api_base_url()}/demo-ticket",
        params={"index": index},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def request_prediction(
    payload: dict[str, str], base_url: str | None = None
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url or get_api_base_url()}/predict",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_health(base_url: str | None = None) -> dict[str, Any]:
    response = requests.get(f"{base_url or get_api_base_url()}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def _render_prediction_card(task_name: str, prediction: dict[str, Any]) -> None:
    title = task_name.replace("_", " ").title()
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.write(f"**Prediction:** {prediction['label']}")
        st.write(f"**Runner-up:** {prediction['runner_up_label']}")
        st.caption(f"Decision margin gap: {prediction['margin_gap']:.3f}")


def _render_model_metadata(models: dict[str, Any]) -> None:
    with st.expander("Model Details"):
        for task_name, metadata in models.items():
            with st.container(border=True):
                st.write(f"**{task_name.title()} model**")
                st.write(f"Run id: `{metadata['run_id']}`")
                st.write(
                    f"Algorithm: `{metadata['algorithm']}` ({metadata['model_family']})"
                )
                st.write("Feature families: " + ", ".join(metadata["feature_families"]))


def main() -> None:
    st.set_page_config(
        page_title="Ticket Triage Demo",
        layout="wide",
    )
    st.title("Ticket Triage Demo")
    st.caption(
        "End-to-end MVP: Streamlit UI calling a FastAPI service backed by fixed LinearSVC models."
    )

    if "demo_index" not in st.session_state:
        st.session_state.demo_index = 0
    if "subject" not in st.session_state:
        st.session_state.subject = ""
    if "body" not in st.session_state:
        st.session_state.body = ""
    if "language" not in st.session_state:
        st.session_state.language = ""
    if "prediction_response" not in st.session_state:
        st.session_state.prediction_response = None
    if "demo_title" not in st.session_state:
        st.session_state.demo_title = None

    base_url = get_api_base_url()

    with st.sidebar:
        st.header("Service")
        st.code(base_url)
        sidebar_models: dict[str, Any] | None = None
        try:
            health = fetch_health(base_url)
            st.success("API is reachable")
            st.write("Loaded tasks: " + ", ".join(health["tasks"]))
            sidebar_models = health.get("models")
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")
            health = None
        response = st.session_state.prediction_response
        if response and response.get("models"):
            sidebar_models = response["models"]
        if sidebar_models:
            _render_model_metadata(sidebar_models)

    left_col, right_col = st.columns([1.5, 1.0])

    with left_col:
        action_col, info_col = st.columns([0.4, 0.6])
        with action_col:
            if st.button("Generate Demo Ticket", use_container_width=True):
                try:
                    demo_payload = fetch_demo_ticket(
                        st.session_state.demo_index, base_url
                    )
                    st.session_state.demo_index = demo_payload["index"] + 1
                    st.session_state.subject = demo_payload["ticket"]["subject"]
                    st.session_state.body = demo_payload["ticket"]["body"]
                    st.session_state.language = demo_payload["ticket"].get("language", "")
                    st.session_state.demo_title = demo_payload["title"]
                except requests.RequestException as exc:
                    st.error(f"Could not fetch demo ticket: {exc}")
        with info_col:
            if st.session_state.demo_title:
                st.info(f"Loaded demo scenario: {st.session_state.demo_title}")

        with st.form("ticket-input"):
            st.text_input("Subject", key="subject")
            st.text_area("Body", key="body", height=260)
            st.selectbox(
                "Language",
                options=["", "en", "de"],
                format_func=lambda value: value or "Auto / not provided",
                key="language",
            )
            submitted = st.form_submit_button("Predict Ticket", use_container_width=True)

        if submitted:
            try:
                payload = {
                    "subject": st.session_state.subject,
                    "body": st.session_state.body,
                }
                if st.session_state.language:
                    payload["language"] = st.session_state.language
                st.session_state.prediction_response = request_prediction(
                    payload,
                    base_url,
                )
            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")

    with right_col:
        response = st.session_state.prediction_response
        if response:
            st.subheader("Prediction Results")
            queue_col, priority_col = st.columns(2)
            with queue_col:
                _render_prediction_card("queue", response["predictions"]["queue"])
            with priority_col:
                _render_prediction_card("priority", response["predictions"]["priority"])
        else:
            st.subheader("Prediction Results")
            st.write("Submit a ticket or generate a demo ticket to see predictions.")


if __name__ == "__main__":
    main()
