import io
import pytest

import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_recommend_requires_pdf_file(client):
    resp = client.post("/recommend", data={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing 'pdf' file"


def test_recommend_returns_videos_on_success(monkeypatch, client):
    monkeypatch.setattr(app_module, "recommend", lambda pdf_bytes: [{"title": "a video"}])

    resp = client.post("/recommend", data={"pdf": (io.BytesIO(b"%PDF-fake"), "doc.pdf")})

    assert resp.status_code == 200
    assert resp.get_json() == [{"title": "a video"}]


def test_recommend_returns_error_json_when_pipeline_raises(monkeypatch, client):
    def boom(pdf_bytes):
        raise RuntimeError("chunking/dedup failed: bad pdf")

    monkeypatch.setattr(app_module, "recommend", boom)

    resp = client.post("/recommend", data={"pdf": (io.BytesIO(b"%PDF-fake"), "doc.pdf")})

    body = resp.get_json()
    assert "chunking/dedup failed" in body["error"]


def test_cors_header_allowed_for_extension_origin(monkeypatch, client):
    monkeypatch.setattr(app_module, "recommend", lambda pdf_bytes: [])
    resp = client.post(
        "/recommend",
        data={"pdf": (io.BytesIO(b"%PDF-fake"), "doc.pdf")},
        headers={"Origin": "chrome-extension://abcdefg"},
    )
    assert resp.headers.get("Access-Control-Allow-Origin") == "chrome-extension://abcdefg"


def test_cors_header_blocked_for_arbitrary_website_origin(monkeypatch, client):
    monkeypatch.setattr(app_module, "recommend", lambda pdf_bytes: [])
    resp = client.post(
        "/recommend",
        data={"pdf": (io.BytesIO(b"%PDF-fake"), "doc.pdf")},
        headers={"Origin": "https://evil.example.com"},
    )
    assert "Access-Control-Allow-Origin" not in resp.headers


def test_oversized_upload_is_rejected(client):
    too_big = io.BytesIO(b"0" * (26 * 1024 * 1024))  # over the 25MB cap
    resp = client.post("/recommend", data={"pdf": (too_big, "doc.pdf")})
    assert resp.status_code == 413
