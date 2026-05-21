"""Tests for vendor-neutral OpenAI-API-compatible client configuration.

The `openai` SDK may not be installed in every environment; if missing, these
tests are skipped. When present, we patch the SDK's `OpenAI` and `AzureOpenAI`
classes with stubs that capture constructor kwargs so we can assert that the
environment-driven configuration is wired correctly.
"""

from __future__ import annotations

import pytest

openai = pytest.importorskip("openai")


from gpqa_cmab.llm.openai_compatible import (  # noqa: E402
    AzureOpenAIClient,
    OpenAICompatibleClient,
    _parse_headers,
)


class _StubOpenAI:
    instances: list[dict] = []

    def __init__(self, **kwargs):  # noqa: ANN003
        type(self).instances.append(kwargs)
        self.kwargs = kwargs


@pytest.fixture
def stub_openai(monkeypatch):
    _StubOpenAI.instances = []
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAI)
    monkeypatch.setattr(openai, "AzureOpenAI", _StubOpenAI)
    return _StubOpenAI


def test_openai_compatible_passes_base_url_and_headers(stub_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-123")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.together.xyz/v1")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-x")
    monkeypatch.setenv(
        "LLM_DEFAULT_HEADERS",
        "HTTP-Referer=https://example.com,X-Title=gpqa",
    )
    monkeypatch.setenv("LLM_TIMEOUT_S", "12.5")

    client = OpenAICompatibleClient()
    kwargs = stub_openai.instances[-1]
    assert kwargs["api_key"] == "key-123"
    assert kwargs["base_url"] == "https://api.together.xyz/v1"
    assert kwargs["organization"] == "org-x"
    assert kwargs["default_headers"] == {
        "HTTP-Referer": "https://example.com",
        "X-Title": "gpqa",
    }
    assert kwargs["timeout"] == 12.5
    assert client.base_url == "https://api.together.xyz/v1"


def test_openai_compatible_falls_back_to_sentinel_key(stub_openai, monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_API_KEYS", "LLM_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    OpenAICompatibleClient()
    kwargs = stub_openai.instances[-1]
    assert kwargs["api_key"] == "not-needed"
    assert kwargs["base_url"] == "http://localhost:8000/v1"


def test_openai_compatible_constructor_overrides_env(stub_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env-host/v1")
    OpenAICompatibleClient(api_key="explicit", base_url="https://override/v1")
    kwargs = stub_openai.instances[-1]
    assert kwargs["api_key"] == "explicit"
    assert kwargs["base_url"] == "https://override/v1"


def test_azure_client_requires_endpoint_and_version(stub_openai, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    with pytest.raises(ValueError):
        AzureOpenAIClient(api_key="k")
    with pytest.raises(ValueError):
        AzureOpenAIClient(api_key="k", endpoint="https://x.openai.azure.com")


def test_azure_client_wires_endpoint_and_version(stub_openai, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azk")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AzureOpenAIClient()
    kwargs = stub_openai.instances[-1]
    assert kwargs["api_key"] == "azk"
    assert kwargs["azure_endpoint"] == "https://x.openai.azure.com"
    assert kwargs["api_version"] == "2024-06-01"


def test_parse_headers_supports_json_and_kv():
    assert _parse_headers(None) is None
    assert _parse_headers("") is None
    assert _parse_headers('{"a": "1", "b": "2"}') == {"a": "1", "b": "2"}
    assert _parse_headers("a=1,b=2") == {"a": "1", "b": "2"}
    with pytest.raises(ValueError):
        _parse_headers("invalid")
    with pytest.raises(ValueError):
        _parse_headers('["not", "an", "object"]')


def test_make_client_routes_openai_compatible_aliases(stub_openai, monkeypatch):
    from gpqa_cmab.cli import make_client

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    for alias in ("openai", "openai_compatible", "vllm", "groq", "ollama"):
        client = make_client(alias)
        assert isinstance(client, OpenAICompatibleClient), alias


def test_make_client_routes_azure(stub_openai, monkeypatch):
    from gpqa_cmab.cli import make_client

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    client = make_client("azure_openai")
    assert isinstance(client, AzureOpenAIClient)


def test_make_client_rejects_unknown():
    from gpqa_cmab.cli import make_client

    with pytest.raises(ValueError):
        make_client("not-a-real-provider")


class _StubChatCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)

        class _Msg:
            content = '{"final_answer": "A"}'

        class _Choice:
            message = _Msg()

        class _Usage:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

            def model_dump(self, mode="json"):
                return {}

        return _Resp()


class _StubResponses:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)

        class _Details:
            reasoning_tokens = 7

        class _Usage:
            input_tokens = 11
            output_tokens = 22
            total_tokens = 33
            output_tokens_details = _Details()

        class _Resp:
            output_text = '{"final_answer": "B"}'
            usage = _Usage()

            def model_dump(self, mode="json"):
                return {}

        return _Resp()


class _StubChatClient:
    def __init__(self, **kwargs):  # noqa: ANN003
        self.chat_completions = _StubChatCompletions()
        self.responses = _StubResponses()

        class _Chat:
            completions = self.chat_completions

        self.chat = _Chat()


def test_reasoning_effort_chat_completions_sets_param_and_omits_temperature(
    monkeypatch,
):
    from gpqa_cmab.schemas import LLMRequest

    stub = _StubChatClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: stub)
    monkeypatch.setenv("REASONING_EFFORT", "high")
    monkeypatch.setenv("LLM_USE_RESPONSES_API", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    client = OpenAICompatibleClient()
    assert client.reasoning_effort == "high"
    assert client.use_responses_api is False
    client.complete(LLMRequest(prompt="hi", model="gpt-5", temperature=0.7))

    call = stub.chat_completions.calls[-1]
    assert call["reasoning_effort"] == "high"
    assert "temperature" not in call
    assert call["model"] == "gpt-5"


def test_reasoning_effort_auto_switches_to_responses_api_for_openai(monkeypatch):
    from gpqa_cmab.schemas import LLMRequest

    stub = _StubChatClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: stub)
    monkeypatch.delenv("LLM_USE_RESPONSES_API", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("REASONING_EFFORT", "medium")
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    client = OpenAICompatibleClient()
    assert client.use_responses_api is True
    response = client.complete(LLMRequest(prompt="hi", model="gpt-5.5"))

    call = stub.responses.calls[-1]
    assert call["model"] == "gpt-5.5"
    assert call["reasoning"] == {"effort": "medium"}
    assert call["input"] == [{"role": "user", "content": "hi"}]
    assert stub.chat_completions.calls == []  # didn't fall through to chat
    # Responses-API usage shape is normalized into Usage().
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 22
    assert response.usage.total_tokens == 33
    assert response.usage.reasoning_tokens == 7
    assert response.content == '{"final_answer": "B"}'


def test_reasoning_effort_stays_on_chat_for_non_openai_base_url(monkeypatch):
    from gpqa_cmab.schemas import LLMRequest

    stub = _StubChatClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: stub)
    monkeypatch.delenv("LLM_USE_RESPONSES_API", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    client = OpenAICompatibleClient()
    assert client.use_responses_api is False
    client.complete(LLMRequest(prompt="hi", model="x"))
    assert stub.chat_completions.calls
    assert stub.responses.calls == []


def test_no_reasoning_effort_passes_temperature(monkeypatch):
    from gpqa_cmab.schemas import LLMRequest

    stub = _StubChatClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: stub)
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    monkeypatch.delenv("LLM_USE_RESPONSES_API", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    OpenAICompatibleClient().complete(
        LLMRequest(prompt="hi", model="gpt-4o-mini", temperature=0.3)
    )
    call = stub.chat_completions.calls[-1]
    assert "reasoning_effort" not in call
    assert call["temperature"] == 0.3


def test_reasoning_effort_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("REASONING_EFFORT", "extreme")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(ValueError):
        OpenAICompatibleClient()


def test_reasoning_effort_accepts_extended_set(monkeypatch):
    stub = _StubChatClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: stub)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
        monkeypatch.setenv("REASONING_EFFORT", effort)
        client = OpenAICompatibleClient()
        assert client.reasoning_effort == effort


def test_reasoning_effort_empty_is_disabled(monkeypatch):
    stub = _StubChatClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: stub)
    monkeypatch.setenv("REASONING_EFFORT", "  ")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    client = OpenAICompatibleClient()
    assert client.reasoning_effort is None


def test_max_output_tokens_chat_completions(monkeypatch):
    from gpqa_cmab.schemas import LLMRequest

    stub = _StubChatClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: stub)
    monkeypatch.setenv("LLM_USE_RESPONSES_API", "false")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "1500")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    OpenAICompatibleClient().complete(LLMRequest(prompt="hi", model="gpt-4o-mini"))
    call = stub.chat_completions.calls[-1]
    assert call["max_tokens"] == 1500


def test_max_output_tokens_responses_api(monkeypatch):
    from gpqa_cmab.schemas import LLMRequest

    stub = _StubChatClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: stub)
    monkeypatch.setenv("LLM_USE_RESPONSES_API", "true")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "25000")
    monkeypatch.setenv("REASONING_EFFORT", "medium")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    OpenAICompatibleClient().complete(LLMRequest(prompt="hi", model="gpt-5.5"))
    call = stub.responses.calls[-1]
    assert call["max_output_tokens"] == 25000


def test_max_output_tokens_invalid_rejected(monkeypatch):
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "-3")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(ValueError):
        OpenAICompatibleClient()


# --- Multi-key load-balancing (API key pool) ---------------------------------


def _make_stub_client_factory():
    """Return (factory, instances) where factory builds a fresh stub per call."""
    instances: list[_StubChatClient] = []

    def factory(**kwargs):  # noqa: ANN003
        stub = _StubChatClient(**kwargs)
        stub.api_key = kwargs.get("api_key")
        instances.append(stub)
        return stub

    return factory, instances


def test_pool_built_from_openai_api_keys_env(monkeypatch):
    factory, instances = _make_stub_client_factory()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEYS", "k1, k2 , k3")
    monkeypatch.delenv("LLM_USE_RESPONSES_API", raising=False)

    client = OpenAICompatibleClient()
    assert client.api_key_count == 3
    assert [s.api_key for s in instances] == ["k1", "k2", "k3"]


def test_pool_deduplicates_keys(monkeypatch):
    factory, instances = _make_stub_client_factory()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEYS", "k1,k1,k2")

    client = OpenAICompatibleClient()
    assert client.api_key_count == 2
    assert [s.api_key for s in instances] == ["k1", "k2"]


def test_explicit_api_keys_constructor_overrides_env(monkeypatch):
    factory, instances = _make_stub_client_factory()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.setenv("OPENAI_API_KEY", "env-single")
    monkeypatch.setenv("OPENAI_API_KEYS", "envk1,envk2,envk3")

    client = OpenAICompatibleClient(api_keys=["explicit-a", "explicit-b"])
    assert client.api_key_count == 2
    assert [s.api_key for s in instances] == ["explicit-a", "explicit-b"]


def test_pool_round_robin_distributes_calls(monkeypatch):
    from gpqa_cmab.schemas import LLMRequest

    factory, instances = _make_stub_client_factory()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_USE_RESPONSES_API", raising=False)
    monkeypatch.setenv("OPENAI_API_KEYS", "k1,k2,k3")

    client = OpenAICompatibleClient()
    for _ in range(6):
        client.complete(LLMRequest(prompt="hi", model="gpt-4o-mini"))

    call_counts = [len(s.chat_completions.calls) for s in instances]
    assert call_counts == [2, 2, 2]


def test_pool_rotates_on_rate_limit_and_parks_offender(monkeypatch):
    from gpqa_cmab.schemas import LLMRequest

    factory, instances = _make_stub_client_factory()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_USE_RESPONSES_API", raising=False)
    monkeypatch.setenv("OPENAI_API_KEYS", "k1,k2")
    monkeypatch.setenv("OPENAI_KEY_COOLDOWN_S", "5")

    client = OpenAICompatibleClient()
    bad_stub = instances[0]

    # Patch the first stub's chat.completions.create to raise RateLimitError once.
    original = bad_stub.chat_completions.create
    raised = {"count": 0}

    class _FakeResponse:
        headers = {"retry-after": "0"}
        status_code = 429
        request = None

    def _raise_once(**kwargs):  # noqa: ANN003
        if raised["count"] == 0:
            raised["count"] += 1
            raise openai.RateLimitError(
                "rate limit", response=_FakeResponse(), body=None
            )
        return original(**kwargs)

    bad_stub.chat_completions.create = _raise_once  # type: ignore[assignment]

    # Single complete() call should succeed by rotating to the second key.
    response = client.complete(LLMRequest(prompt="hi", model="m"))
    assert response.content == '{"final_answer": "A"}'
    # k2 served this request.
    assert len(instances[1].chat_completions.calls) == 1
    # k1 was parked; subsequent calls should skip it until cooldown elapses.
    for _ in range(3):
        client.complete(LLMRequest(prompt="hi", model="m"))
    # k1 should still have zero successful calls (only the raised one).
    assert raised["count"] == 1
    # All 3 follow-ups landed on k2.
    assert len(instances[1].chat_completions.calls) == 4


def test_pool_reraises_when_all_keys_rate_limited(monkeypatch):
    from gpqa_cmab.schemas import LLMRequest

    factory, instances = _make_stub_client_factory()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_USE_RESPONSES_API", raising=False)
    monkeypatch.setenv("OPENAI_API_KEYS", "k1,k2")

    client = OpenAICompatibleClient()

    class _FakeResponse:
        headers = {}
        status_code = 429
        request = None

    def _always_raise(**kwargs):  # noqa: ANN003
        raise openai.RateLimitError("nope", response=_FakeResponse(), body=None)

    for stub in instances:
        stub.chat_completions.create = _always_raise  # type: ignore[assignment]

    with pytest.raises(openai.RateLimitError):
        client.complete(LLMRequest(prompt="hi", model="m"))


def test_invalid_cooldown_rejected(monkeypatch):
    factory, _ = _make_stub_client_factory()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_KEY_COOLDOWN_S", "not-a-number")
    with pytest.raises(ValueError):
        OpenAICompatibleClient()
