from unittest.mock import patch

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aider.models import Model, _normalize_model_name


class DummyBadRequestError(Exception):
    """Test double for litellm.BadRequestError."""


def test_normalize_local_model_prefix_for_litellm_completion():
    assert _normalize_model_name("local/qwen3-coder:30b") == "ollama_chat/qwen3-coder:30b"


def test_keep_non_local_model_name_unchanged():
    assert _normalize_model_name("gpt-4o") == "gpt-4o"


@patch("aider.models.litellm.BadRequestError", DummyBadRequestError)
@patch("aider.models.litellm.completion")
def test_send_completion_fallback_on_missing_provider_error(mock_completion):
    first_error = DummyBadRequestError("LLM Provider NOT provided")
    fallback_response = object()
    mock_completion.side_effect = [first_error, fallback_response]

    model = Model("local/qwen3-coder:30b")
    messages = [{"role": "user", "content": "hello"}]

    _, response = model.send_completion(messages=messages, functions=None, stream=False)

    assert response is fallback_response
    assert mock_completion.call_count == 2

    first_call = mock_completion.call_args_list[0].kwargs
    second_call = mock_completion.call_args_list[1].kwargs

    assert first_call["model"] == "ollama_chat/qwen3-coder:30b"
    assert second_call["model"] == "ollama_chat/qwen3-coder:30b"
    assert first_call["messages"] == messages
    assert second_call["messages"] == messages


@patch("aider.models.litellm.BadRequestError", DummyBadRequestError)
@patch("aider.models.litellm.completion")
def test_send_completion_fallback_switches_to_known_ollama_model(mock_completion):
    first_error = DummyBadRequestError("LLM Provider NOT provided")
    fallback_response = object()
    mock_completion.side_effect = [first_error, fallback_response]

    model = Model("gpt-4o")
    messages = [{"role": "user", "content": "hello"}]

    _, response = model.send_completion(messages=messages, functions=None, stream=False)

    assert response is fallback_response
    assert mock_completion.call_count == 2
    assert mock_completion.call_args_list[0].kwargs["model"] == "gpt-4o"
    assert mock_completion.call_args_list[1].kwargs["model"] == "ollama_chat/qwen3-coder:30b"


@patch("aider.models.litellm.BadRequestError", DummyBadRequestError)
@patch("aider.models.litellm.completion")
def test_send_completion_reraises_unrelated_bad_request(mock_completion):
    mock_completion.side_effect = DummyBadRequestError("Some other bad request")

    model = Model("gpt-4o")
    messages = [{"role": "user", "content": "hello"}]

    try:
        model.send_completion(messages=messages, functions=None, stream=False)
    except DummyBadRequestError as exc:
        assert "Some other bad request" in str(exc)
    else:
        raise AssertionError("Expected DummyBadRequestError to be raised")
