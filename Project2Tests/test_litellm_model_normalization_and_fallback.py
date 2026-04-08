import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aider.models import Model, _normalize_model_name


class DummyBadRequestError(Exception):
    """Test double for litellm.BadRequestError."""


class FakeLiteLLM:
    BadRequestError = DummyBadRequestError

    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        result = self._side_effects.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def build_minimal_model(name):
    """Create a Model instance without running __init__ or importing litellm deps."""
    model = Model.__new__(Model)
    model.name = name
    model.use_temperature = True
    model.extra_params = None
    model.verbose = False
    model.is_deepseek_r1 = lambda: False
    model.is_ollama = lambda: name.startswith("ollama/")
    model.token_count = lambda messages: 0
    model.github_copilot_token_to_open_ai_key = lambda _headers: None
    return model


def test_normalize_local_model_prefix_for_litellm_completion():
    assert (
        _normalize_model_name("local/qwen3-coder:30b") == "ollama_chat/qwen3-coder:30b"
    )


def test_keep_non_local_model_name_unchanged():
    assert _normalize_model_name("gpt-4o") == "gpt-4o"


def test_send_completion_fallback_on_missing_provider_error():
    fallback_response = object()
    model = build_minimal_model("local/qwen3-coder:30b")
    fake_litellm = FakeLiteLLM(
        [DummyBadRequestError("LLM Provider NOT provided"), fallback_response]
    )

    with patch("aider.models.litellm", new=fake_litellm):
        messages = [{"role": "user", "content": "hello"}]
        _, response = model.send_completion(
            messages=messages, functions=None, stream=False
        )

    assert response is fallback_response
    assert len(fake_litellm.calls) == 2

    first_call = fake_litellm.calls[0]
    second_call = fake_litellm.calls[1]

    assert first_call["model"] == "ollama_chat/qwen3-coder:30b"
    assert second_call["model"] == "ollama_chat/qwen3-coder:30b"
    assert first_call["messages"] == messages
    assert second_call["messages"] == messages


def test_send_completion_fallback_switches_to_known_ollama_model():
    fallback_response = object()
    model = build_minimal_model("gpt-4o")
    fake_litellm = FakeLiteLLM(
        [DummyBadRequestError("LLM Provider NOT provided"), fallback_response]
    )

    with patch("aider.models.litellm", new=fake_litellm):
        messages = [{"role": "user", "content": "hello"}]
        _, response = model.send_completion(
            messages=messages, functions=None, stream=False
        )

    assert response is fallback_response
    assert len(fake_litellm.calls) == 2
    assert fake_litellm.calls[0]["model"] == "gpt-4o"
    assert fake_litellm.calls[1]["model"] == "ollama_chat/qwen3-coder:30b"


def test_send_completion_reraises_unrelated_bad_request():
    model = build_minimal_model("gpt-4o")
    fake_litellm = FakeLiteLLM([DummyBadRequestError("Some other bad request")])

    with patch("aider.models.litellm", new=fake_litellm):
        messages = [{"role": "user", "content": "hello"}]
        with pytest.raises(DummyBadRequestError, match="Some other bad request"):
            model.send_completion(messages=messages, functions=None, stream=False)
