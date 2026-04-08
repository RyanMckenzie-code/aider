from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aider.models import Model, _normalize_model_name


class DummyBadRequestError(Exception):
    """Test double for litellm.BadRequestError."""


class FakeLiteLLM:
    BadRequestError = DummyBadRequestError

    def __init__(self, side_effects):
        self._lazy_module = None
        self._side_effects = list(side_effects)
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        result = self._side_effects.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_normalize_local_model_prefix_for_litellm_completion():
    assert _normalize_model_name("local/qwen3-coder:30b") == "ollama_chat/qwen3-coder:30b"


def test_keep_non_local_model_name_unchanged():
    assert _normalize_model_name("gpt-4o") == "gpt-4o"


def test_send_completion_fallback_on_missing_provider_error(monkeypatch):
    fallback_response = object()
    model = Model("local/qwen3-coder:30b")
    fake_litellm = FakeLiteLLM(
        [DummyBadRequestError("LLM Provider NOT provided"), fallback_response]
    )
    monkeypatch.setattr("aider.models.litellm", fake_litellm)
    messages = [{"role": "user", "content": "hello"}]

    _, response = model.send_completion(messages=messages, functions=None, stream=False)

    assert response is fallback_response
    assert len(fake_litellm.calls) == 2

    first_call = fake_litellm.calls[0]
    second_call = fake_litellm.calls[1]

    assert first_call["model"] == "ollama_chat/qwen3-coder:30b"
    assert second_call["model"] == "ollama_chat/qwen3-coder:30b"
    assert first_call["messages"] == messages
    assert second_call["messages"] == messages


def test_send_completion_fallback_switches_to_known_ollama_model(monkeypatch):
    fallback_response = object()
    model = Model("gpt-4o")
    fake_litellm = FakeLiteLLM(
        [DummyBadRequestError("LLM Provider NOT provided"), fallback_response]
    )
    monkeypatch.setattr("aider.models.litellm", fake_litellm)
    messages = [{"role": "user", "content": "hello"}]

    _, response = model.send_completion(messages=messages, functions=None, stream=False)

    assert response is fallback_response
    assert len(fake_litellm.calls) == 2
    assert fake_litellm.calls[0]["model"] == "gpt-4o"
    assert fake_litellm.calls[1]["model"] == "ollama_chat/qwen3-coder:30b"


def test_send_completion_reraises_unrelated_bad_request(monkeypatch):
    model = Model("gpt-4o")
    fake_litellm = FakeLiteLLM([DummyBadRequestError("Some other bad request")])
    monkeypatch.setattr("aider.models.litellm", fake_litellm)
    messages = [{"role": "user", "content": "hello"}]

    try:
        model.send_completion(messages=messages, functions=None, stream=False)
    except DummyBadRequestError as exc:
        assert "Some other bad request" in str(exc)
    else:
        raise AssertionError("Expected DummyBadRequestError to be raised")
