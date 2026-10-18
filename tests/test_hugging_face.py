import requests
import hugging_face as hf


def _chunk():
    return {
        "chunk": "some context text",
        "keywords": [{"keyword": "docker", "score": 0.9, "freq": 2}, {"keyword": "kubernetes", "score": 0.8, "freq": 1}],
    }


def test_build_prompt_includes_context_and_keywords():
    prompt = hf._build_prompt(_chunk())
    assert "some context text" in prompt
    assert "docker" in prompt
    assert "kubernetes" in prompt


def test_generate_queries_uses_llm_response_when_available(monkeypatch):
    class FakeResp:
        def json(self):
            return {"choices": [{"message": {"content": "docker tutorial for beginners"}}]}

    monkeypatch.setattr(hf.requests, "post", lambda *a, **k: FakeResp())

    queries = hf.generate_queries([_chunk()])

    assert queries == ["docker tutorial for beginners"]


def test_generate_queries_falls_back_to_keywords_when_llm_response_malformed(monkeypatch):
    class FakeResp:
        def json(self):
            return {"error": "rate limited"}  # no "choices" key

    monkeypatch.setattr(hf.requests, "post", lambda *a, **k: FakeResp())

    queries = hf.generate_queries([_chunk()])

    assert queries == ["docker kubernetes"]


def test_generate_queries_falls_back_on_network_error(monkeypatch):
    def raise_timeout(*a, **k):
        raise requests.exceptions.Timeout("took too long")

    monkeypatch.setattr(hf.requests, "post", raise_timeout)

    queries = hf.generate_queries([_chunk()])

    assert queries == ["docker kubernetes"]


def test_generate_queries_falls_back_on_non_json_response(monkeypatch):
    class FakeResp:
        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(hf.requests, "post", lambda *a, **k: FakeResp())

    queries = hf.generate_queries([_chunk()])

    assert queries == ["docker kubernetes"]


def test_build_prompt_includes_document_topic_when_given():
    assert "load balancer" in hf._build_prompt(_chunk(), topic="multithreaded load balancer")
    assert "document about" not in hf._build_prompt(_chunk())


def test_document_topic_falls_back_to_keywords_on_failure(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(hf.requests, "post", boom)
    assert hf.document_topic("some title page", ["load", "balancer", "thread", "pool"]) == "load balancer thread"
