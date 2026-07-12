"""Responses API client translation and Copilot fallback selection."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart import backend, gateway  # noqa: E402
from ludvart.llm import (  # noqa: E402
    LLMError,
    ProviderConfig,
    ResponsesClient,
    ToolSpec,
)


def _config():
    return ProviderConfig(
        name="custom",
        api_url="http://127.0.0.1:4000",
        api_key="test",
        model="github_copilot/gpt-5.6-terra",
        api_mode="responses",
    )


def _client_without_init():
    client = object.__new__(ResponsesClient)
    client.config = _config()
    client._detected_context_window = 0
    return client


def test_responses_input_translates_tool_round_trip():
    client = _client_without_init()
    items = client._responses_input(
        [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": "Do the task"},
            {
                "role": "assistant",
                "content": " ",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"x": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
    )
    assert items[0] == {
        "role": "developer",
        "content": [{"type": "input_text", "text": "System rules"}],
    }
    assert items[2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read",
        "arguments": '{"x": 1}',
    }
    assert items[3]["role"] == "assistant"
    assert items[4] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "result",
    }
    print("Responses input translates system, history, and tool round trip: OK")


def test_responses_request_uses_responses_tool_shape():
    client = _client_without_init()
    tool = ToolSpec("lookup", "Look something up", {"type": "object"})
    request = client._prepare_converse(
        [{"role": "user", "content": "hello"}], [tool], 42
    )
    assert request["model"] == "github_copilot/gpt-5.6-terra"
    assert request["max_output_tokens"] == 42
    assert request["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look something up",
            "parameters": {"type": "object"},
        }
    ]
    print("Responses request uses native function tool shape: OK")


def test_responses_turn_parses_text_and_function_calls():
    client = _client_without_init()
    response = NS(
        output=[
            NS(
                type="message",
                content=[NS(type="output_text", text="Working on it.")],
            ),
            NS(
                type="function_call",
                call_id="fc_1",
                name="lookup",
                arguments='{"query":"status"}',
            ),
        ],
        output_text="",
        usage=None,
    )
    turn = client._responses_turn(response)
    assert turn.text == "Working on it."
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "fc_1"
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].input == {"query": "status"}
    print("Responses output parses text and function calls: OK")


def test_responses_streams_text_and_function_calls():
    client = _client_without_init()
    completed = NS(
        output=[
            NS(
                type="function_call",
                call_id="fc_1",
                name="lookup",
                arguments='{"query":"status"}',
            )
        ],
        usage=None,
    )
    events = iter(
        [
            NS(type="response.output_text.delta", delta="I will "),
            NS(type="response.output_text.delta", delta="check that."),
            NS(
                type="response.output_item.done",
                item=completed.output[0],
            ),
            NS(type="response.completed", response=completed),
        ]
    )
    client._client = NS(responses=NS(create=lambda **kwargs: events))
    updates = []
    turn = client._stream_turn({"model": "test"}, updates.append)
    assert updates == ["I will ", "I will check that."]
    assert turn.text == "I will check that."
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "fc_1"
    assert turn.tool_calls[0].input == {"query": "status"}
    print("Responses stream emits text deltas and function calls: OK")


def test_responses_streams_reasoning_summary_as_narration():
    client = _client_without_init()
    completed = NS(output=[], usage=None)
    events = iter(
        [
            NS(type="response.reasoning_summary_text.delta", delta="Checking "),
            NS(type="response.reasoning_summary_text.delta", delta="the file."),
            NS(type="response.output_text.delta", delta="It is present."),
            # Later reasoning must not replace answer narration.
            NS(type="response.reasoning_summary_text.delta", delta="Ignored."),
            NS(type="response.completed", response=completed),
        ]
    )
    client._client = NS(responses=NS(create=lambda **kwargs: events))
    updates = []

    turn = client._stream_turn({"model": "test"}, updates.append)

    assert updates == ["Checking ", "Checking the file.", "It is present."]
    assert turn.text == "It is present."
    assert "Checking" not in turn.assistant_message["content"]
    print("Responses reasoning summaries stream as transient narration: OK")


def test_responses_reasoning_item_falls_back_to_narration():
    client = _client_without_init()
    reasoning = NS(
        type="reasoning",
        summary=[NS(type="summary_text", text="Considering options.")],
    )
    completed = NS(output=[], usage=None)
    events = iter(
        [
            NS(type="response.output_item.done", item=reasoning),
            NS(type="response.completed", response=completed),
        ]
    )
    client._client = NS(responses=NS(create=lambda **kwargs: events))
    updates = []

    turn = client._stream_turn({"model": "test"}, updates.append)

    assert updates == ["Considering options."]
    assert turn.text == ""
    assert turn.assistant_message["content"] == ""
    print("Responses reasoning-item fallback emits transient narration: OK")


def test_copilot_chat_rejection_falls_back_to_responses(monkeypatch):
    chat = NS(
        config=ProviderConfig("custom", "http://gateway", "key", "github_copilot/terra"),
        verify=lambda: (_ for _ in ()).throw(
            LLMError('model is not accessible via the /chat/completions endpoint')
        ),
    )
    responses = NS(config=NS(api_mode="responses"), verified=False)
    responses.verify = lambda: setattr(responses, "verified", True)
    calls = []
    monkeypatch.setattr(backend, "build_client", lambda config: calls.append(config) or responses)
    old_gateway = NS(model="gpt-5.6-terra", stop=lambda: None)

    class FakeGateway:
        base_url = "http://responses-gateway"
        litellm_model = "github_copilot/gpt-5.6-terra"

        def __init__(self, model, *, api_mode):
            assert model == "gpt-5.6-terra"
            assert api_mode == "responses"

        def start(self):
            pass

    monkeypatch.setattr(gateway, "CopilotGateway", FakeGateway)
    registration = {"provider": "copilot", "model": "gpt-5.6-terra"}
    live = backend.Backend(
        client=chat, gateway=old_gateway, registration=registration
    )
    backend.verify_backend(live)
    assert live.client is responses
    assert responses.verified
    assert calls[0].api_mode == "responses"
    assert old_gateway is not live.gateway
    assert registration["api_mode"] == "responses"
    print("Copilot chat rejection selects Responses client: OK")


def test_non_copilot_chat_rejection_does_not_fallback(monkeypatch):
    client = NS(
        config=ProviderConfig("custom", "http://gateway", "key", "model"),
        verify=lambda: (_ for _ in ()).throw(
            LLMError('model is not accessible via the /chat/completions endpoint')
        ),
    )
    monkeypatch.setattr(backend, "build_client", lambda config: None)
    try:
        backend.verify_backend(backend.Backend(client=client))
    except LLMError:
        pass
    else:
        raise AssertionError("non-Copilot backend unexpectedly fell back")
    print("non-Copilot chat rejection is not retried: OK")


def main():
    test_responses_input_translates_tool_round_trip()
    test_responses_request_uses_responses_tool_shape()
    test_responses_turn_parses_text_and_function_calls()
    test_responses_streams_text_and_function_calls()
    test_responses_streams_reasoning_summary_as_narration()
    test_responses_reasoning_item_falls_back_to_narration()
    test_copilot_chat_rejection_falls_back_to_responses()
    test_non_copilot_chat_rejection_does_not_fallback()
    print("\nALL Responses client tests passed.")


if __name__ == "__main__":
    main()
