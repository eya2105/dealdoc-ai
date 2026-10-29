import config
from llm import GroqLLM
from tests.fakes import fake_client


def test_complete_text_sends_model_messages_and_temperature():
    client = fake_client(content="hello")
    llm = GroqLLM(api_key="k", model="m", client=client)
    assert llm.complete_text("sys", "usr", temperature=0.3) == "hello"
    call = client.calls[0]
    assert call["model"] == "m" and call["temperature"] == 0.3
    assert call["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert "response_format" not in call


def test_complete_json_asks_for_json_and_parses_it():
    client = fake_client(content='{"ok": true}')
    assert GroqLLM(api_key="k", client=client).complete_json("s", "u") == {"ok": True}
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_complete_json_returns_none_on_invalid_json():
    assert GroqLLM(api_key="k", client=fake_client(content="not json")).complete_json("s", "u") is None


def test_complete_json_returns_none_on_api_error():
    assert GroqLLM(api_key="k", client=fake_client(error=RuntimeError("boom"))).complete_json("s", "u") is None


def test_missing_key_raises_a_clear_error_on_first_use():
    llm = GroqLLM(api_key=None)
    try:
        llm.complete_text("s", "u")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "GROQ_API_KEY" in str(e)


def test_real_client_is_built_with_retries_for_rate_limits():
    assert GroqLLM(api_key="k")._get_client().max_retries == config.LLM_MAX_RETRIES
