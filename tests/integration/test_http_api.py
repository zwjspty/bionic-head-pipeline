from __future__ import annotations

from fastapi.testclient import TestClient

from bionic_head.api.app import create_app


def test_health_is_independent_of_provider_status(app) -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_diagnostics_routes(app) -> None:
    client = TestClient(app)

    response = client.get("/diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"asr", "llm", "tts", "audio2face", "ue5"}
    assert body["asr"]["provider"] == "mock"
    assert body["asr"]["available"] is True
    assert client.get("/diagnostics/asr").json()["adapter"] == "asr"
    assert client.get("/diagnostics/not-real").status_code == 404


def test_latest_returns_404_before_first_run(app) -> None:
    client = TestClient(app)

    response = client.get("/pipeline/latest")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "invalid_request"


def test_offline_endpoint_and_latest(app, speech_wav) -> None:
    client = TestClient(app)

    response = _post_audio(client, speech_wav)

    assert response.status_code == 200
    assert response.json()["face"]["channel_count"] == 52
    assert client.get("/pipeline/latest").status_code == 200
    assert client.get("/ue5/latest").json()["format"] == "morpheus_52_raw"


def test_silence_maps_to_422(app, silence_wav) -> None:
    response = _post_audio(TestClient(app), silence_wav)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_speech_detected"


def test_provider_timeout_maps_to_504(mock_settings, tmp_path, speech_wav) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "api-data"
    settings.adapters.tts.timeout_seconds = 0.01
    settings.mock.timeout_stage = "tts"
    app = create_app(settings)

    response = _post_audio(TestClient(app), speech_wav)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "provider_timeout"


def test_failed_run_does_not_overwrite_latest(app, speech_wav) -> None:
    client = TestClient(app)
    first = _post_audio(client, speech_wav)
    first.raise_for_status()
    latest_before = client.get("/pipeline/latest").json()

    app.state.container.settings.mock.fail_stage = "tts"
    failed = _post_audio(client, speech_wav)

    assert failed.status_code == 502
    latest_after = client.get("/pipeline/latest").json()
    assert latest_after["turn_id"] == latest_before["turn_id"]


def _post_audio(client: TestClient, path):
    with path.open("rb") as handle:
        return client.post(
            "/pipeline/audio",
            files={"audio": (path.name, handle, "audio/wav")},
        )
