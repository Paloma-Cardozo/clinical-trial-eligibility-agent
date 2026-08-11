"""
Smoke test: Verify that the server starts and responds correctly.

This is a minimal test that ensures:
1. FastAPI app initializes without errors
2. GET / returns the HTML page
3. POST /chat accepts requests and returns valid responses
4. Session IDs are generated and persisted
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_server_starts():
    """Verify server responds on GET /."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Clinical Trial Eligibility Agent" in response.text


def test_chat_generates_session_id():
    """Verify POST /chat generates a session ID when none is provided."""
    response = client.post("/chat", json={
        "session_id": None,
        "message": "I have breast cancer"
    })
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] is not None
    assert "response" in data


def test_chat_persists_session_id():
    """Verify that subsequent requests with the same session_id use the same state."""
    # First request: generate session
    response1 = client.post("/chat", json={
        "session_id": None,
        "message": "First message"
    })
    session_id = response1.json()["session_id"]

    # Second request: reuse session
    response2 = client.post("/chat", json={
        "session_id": session_id,
        "message": "Second message"
    })
    assert response2.json()["session_id"] == session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
