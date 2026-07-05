"""Live end-to-end provider verification.

Given one or more ``~/.relai/llm.conf*`` files, for each: resolve the provider,
build the client (spawning the local LiteLLM gateway for Copilot), then exercise
the full agentic path the AI panel uses:

  1. ``verify()``  -- URL / key / model all work.
  2. a streamed tool-calling turn  -- the model asks for a tool, and text
     deltas are fed to ``on_text`` (the live "Thinking" narration).
  3. feed the tool result back and get a streamed final answer.

This talks to real APIs and costs tokens; it is a manual check, not part of the
unit suite. Usage:

    python tools/verify_providers.py <conf-file> [<conf-file> ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relai import gateway, llm  # noqa: E402


WEATHER = llm.ToolSpec(
    name="get_weather",
    description="Get the current weather for a city. Always use this for weather.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)


def _resolve(conf: dict) -> llm.ProviderConfig | None:
    for name in llm._PROVIDER_ORDER:
        cfg = llm._read_provider(name, conf)
        if cfg is not None:
            return cfg
    return None


def _build_client(path: str):
    """Return (client, label, gateway_or_None) for the given conf file."""
    conf = llm._load_conf(path)
    cfg = _resolve(conf)
    if cfg is not None:
        return llm.build_client(cfg), f"{cfg.name}:{cfg.model}", None

    model = llm._getvar(conf, "COPILOT_MODEL")
    if model:
        if not gateway.litellm_available():
            raise RuntimeError("litellm gateway not installed")
        if not gateway.copilot_authenticated():
            raise RuntimeError("Copilot not authenticated (run relai once to log in)")
        gw = gateway.CopilotGateway(model)
        gw.start()
        provider = llm.copilot_provider_config(
            gw.base_url, gw.litellm_model, gateway.GATEWAY_API_KEY
        )
        return llm.build_client(provider), f"copilot:{model}", gw
    raise RuntimeError("no provider configured in this file")


def verify_provider(path: str) -> bool:
    print(f"\n=== {Path(path).name} ===")
    client = None
    gw = None
    try:
        client, label, gw = _build_client(path)
        print(f"provider: {label}")

        # 1. verify
        client.verify()
        print(f"  [1] verify(): OK  (context_window={client.context_window})")

        # 2. streamed tool-calling turn
        messages = [
            {"role": "system", "content": "You are a terse assistant. Use tools when asked."},
            {"role": "user", "content": "What's the weather in Paris? Call get_weather."},
        ]
        chunks = []
        turn = client.converse(
            messages, tools=[WEATHER], max_tokens=512, on_text=chunks.append
        )
        if not turn.tool_calls:
            print(f"  [2] tool call: FAIL -- no tool call (text={turn.text!r})")
            return False
        call = turn.tool_calls[0]
        print(
            f"  [2] tool call: OK  -> {call.name}({call.input}) "
            f"[narration chunks={len(chunks)}, usage={turn.usage}]"
        )

        # 3. feed the tool result and stream the final answer
        history = [
            *messages,
            turn.assistant_message,
            client.tool_result_message(call.id, "Sunny, 22C"),
        ]
        chunks2 = []
        turn2 = client.converse(
            history, tools=[WEATHER], max_tokens=512, on_text=chunks2.append
        )
        print(
            f"  [3] follow-up: OK  text={turn2.text!r} "
            f"[narration chunks={len(chunks2)}, usage={turn2.usage}]"
        )
        if not turn2.text.strip():
            print("  [3] WARN: empty final answer")
        print("  RESULT: PASS")
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue
        import traceback

        print(f"  RESULT: FAIL -- {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    finally:
        if gw is not None:
            gw.stop()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = {Path(p).name: verify_provider(p) for p in argv}
    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
