"""LLM provider clients.

relai talks to one of three providers, selected purely by environment
variables. For each provider the user sets a triplet:

    OpenAI:     OPENAI_API_URL     OPENAI_API_KEY     OPENAI_MODEL
    Anthropic:  ANTHROPIC_API_URL  ANTHROPIC_API_KEY  ANTHROPIC_MODEL
    Google:     GOOGLE_API_URL     GOOGLE_API_KEY     GOOGLE_MODEL
    Custom:     CUSTOM_API_URL     CUSTOM_API_KEY     CUSTOM_MODEL

The "openai" and "custom" providers use the official ``openai`` SDK (custom just
points ``base_url`` at an OpenAI-compatible server: LM Studio, llama.cpp, vLLM,
Ollama's OpenAI shim, gateways, ...). The "anthropic" provider uses the official
``anthropic`` SDK, and "google" uses the ``google-genai`` (Gemini) SDK.

A provider is considered "configured" only when all three of its variables are
set; if several are configured, one is chosen by a fixed precedence
(custom > google > anthropic > openai).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

#: How long (seconds) to wait on any single LLM request.
DEFAULT_TIMEOUT = 30.0

#: A chat message. ``content`` is usually a string, but for tool use it may be a
#: list of provider-native content blocks (text / tool_use / tool_result).
Message = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """A tool advertised to the model (name + JSON-schema for its input)."""

    name: str
    description: str
    input_schema: dict


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation the model requested."""

    id: str
    name: str
    input: dict


@dataclass
class Turn:
    """One assistant response, which may request tool calls.

    ``assistant_message`` is the provider-native message to append back into the
    conversation history verbatim, so a subsequent request replays the exact
    tool_use blocks the model produced (mirroring how a client feeds an
    assistant turn back to the model).
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: Message = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when an LLM request fails (network, auth, bad response, ...)."""


class LLMNotConfigured(RuntimeError):
    """Raised when no provider is fully configured via environment variables."""


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved configuration for the selected provider."""

    name: str          # "openai" | "anthropic" | "google" | "custom"
    api_url: str
    api_key: str
    model: str


# Precedence when more than one provider is fully configured.
_PROVIDER_ORDER = ("custom", "google", "anthropic", "openai")

# Env-var prefixes per provider name.
_ENV_PREFIX = {
    "openai": "OPENAI",
    "anthropic": "ANTHROPIC",
    "google": "GOOGLE",
    "custom": "CUSTOM",
}


def _read_provider(name: str) -> ProviderConfig | None:
    """Return a ProviderConfig if all three env vars for ``name`` are set."""
    prefix = _ENV_PREFIX[name]
    url = os.environ.get(f"{prefix}_API_URL")
    key = os.environ.get(f"{prefix}_API_KEY")
    model = os.environ.get(f"{prefix}_MODEL")
    if url and key and model:
        return ProviderConfig(
            name=name, api_url=url.rstrip("/"), api_key=key, model=model
        )
    return None


def resolve_config() -> ProviderConfig | None:
    """Select a provider from the environment, or None if none is configured."""
    for name in _PROVIDER_ORDER:
        cfg = _read_provider(name)
        if cfg is not None:
            return cfg
    return None


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class LLMClient:
    """Base class: a client that can complete a chat and verify connectivity."""

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.config = config
        self.timeout = timeout

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model

    def complete(self, messages: Sequence[Message], max_tokens: int = 1024) -> str:
        """Return the assistant's reply text for ``messages``."""
        raise NotImplementedError

    def converse(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int = 1024,
    ) -> Turn:
        """One round-trip that may request tool calls.

        The default implementation has no tool support: it just wraps
        :meth:`complete` as a text-only turn. Providers that support tools
        override this.
        """
        text = self.complete(messages, max_tokens=max_tokens)
        return Turn(text=text, assistant_message={"role": "assistant", "content": text})

    def tool_result_message(self, tool_call_id: str, content: str) -> Message:
        """Build the message that reports a tool's output back to the model."""
        return {"role": "user", "content": content}

    def verify(self) -> None:
        """Make a minimal request to confirm URL, key, and model all work.

        Raises :class:`LLMError` on any failure.
        """
        self.complete([{"role": "user", "content": "ping"}], max_tokens=1)


class OpenAIClient(LLMClient):
    """OpenAI / OpenAI-compatible client via the ``openai`` SDK.

    Used for both the "openai" and "custom" providers; ``api_url`` becomes the
    SDK ``base_url``. A trailing ``/chat/completions`` is stripped since the SDK
    appends the path itself.
    """

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT) -> None:
        super().__init__(config, timeout)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError("the 'openai' package is required but not installed") from exc

        base_url = config.api_url
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        self._client = OpenAI(
            api_key=config.api_key, base_url=base_url, timeout=timeout
        )

    def complete(self, messages: Sequence[Message], max_tokens: int = 1024) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=list(messages),
                max_tokens=max_tokens,
            )
        except Exception as exc:  # SDK raises its own error hierarchy
            raise LLMError(f"{self.name} request failed: {exc}") from exc
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response from {self.name}: {resp!r}") from exc


class AnthropicClient(LLMClient):
    """Anthropic client via the ``anthropic`` SDK."""

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT) -> None:
        super().__init__(config, timeout)
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "the 'anthropic' package is required but not installed"
            ) from exc

        base_url = config.api_url
        # The SDK appends /v1/messages; strip a trailing endpoint path if given.
        for suffix in ("/v1/messages", "/messages"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        self._client = Anthropic(
            api_key=config.api_key, base_url=base_url, timeout=timeout
        )

    def complete(self, messages: Sequence[Message], max_tokens: int = 1024) -> str:
        # Anthropic takes the system prompt separately from the message list.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        turns = [m for m in messages if m.get("role") != "system"]
        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": turns,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc
        try:
            return "".join(
                block.text for block in resp.content if block.type == "text"
            )
        except (AttributeError, TypeError) as exc:
            raise LLMError(f"unexpected response from {self.name}: {resp!r}") from exc

    def converse(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int = 1024,
    ) -> Turn:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        turns = [m for m in messages if m.get("role") != "system"]
        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": turns,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc
        try:
            text_parts: list[str] = []
            blocks: list[dict] = []
            tool_calls: list[ToolCall] = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                    tool_calls.append(
                        ToolCall(
                            id=block.id, name=block.name, input=dict(block.input)
                        )
                    )
        except (AttributeError, TypeError) as exc:
            raise LLMError(f"unexpected response from {self.name}: {resp!r}") from exc
        return Turn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            assistant_message={"role": "assistant", "content": blocks},
        )

    def tool_result_message(self, tool_call_id: str, content: str) -> Message:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ],
        }


class GoogleClient(LLMClient):
    """Google Gemini client via the ``google-genai`` SDK.

    Gemini has no ``system`` role: any system messages are combined into the
    ``system_instruction`` config, and the remaining turns use Gemini's
    ``user`` / ``model`` roles. ``api_url`` sets the SDK ``base_url``.
    """

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT) -> None:
        super().__init__(config, timeout)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "the 'google-genai' package is required but not installed"
            ) from exc

        self._genai = genai
        self._types = types
        # google-genai expresses the timeout in milliseconds.
        http_options = types.HttpOptions(
            base_url=config.api_url, timeout=int(timeout * 1000)
        )
        self._client = genai.Client(api_key=config.api_key, http_options=http_options)

    def complete(self, messages: Sequence[Message], max_tokens: int = 1024) -> str:
        types = self._types
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        contents = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            # Gemini uses "model" for assistant turns.
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(role=gemini_role, parts=[types.Part(text=m["content"])])
            )

        config_kwargs: dict = {"max_output_tokens": max_tokens}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)

        try:
            resp = self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc
        try:
            text = resp.text
        except (AttributeError, TypeError) as exc:
            raise LLMError(f"unexpected response from {self.name}: {resp!r}") from exc
        return text or ""


def _client_for(config: ProviderConfig, timeout: float) -> LLMClient:
    if config.name == "anthropic":
        return AnthropicClient(config, timeout)
    if config.name == "google":
        return GoogleClient(config, timeout)
    # "openai" and "custom" both use the OpenAI SDK.
    return OpenAIClient(config, timeout)


def create_client(timeout: float = DEFAULT_TIMEOUT) -> LLMClient:
    """Resolve config from the environment and build the matching client.

    Raises :class:`LLMNotConfigured` if no provider is fully configured.
    """
    config = resolve_config()
    if config is None:
        raise LLMNotConfigured(
            "no LLM provider configured; set the API_URL, API_KEY and MODEL "
            "environment variables for OPENAI, ANTHROPIC, GOOGLE, or CUSTOM"
        )
    return _client_for(config, timeout)
