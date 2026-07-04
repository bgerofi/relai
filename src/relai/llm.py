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
class Usage:
    """Token accounting for one LLM response, normalized across providers.

    ``input_tokens`` / ``output_tokens`` are the prompt (context) and
    completion token counts. ``total_tokens`` is the provider-reported total
    when available, otherwise the sum. ``context_window`` is the model's
    maximum context size (0 / unknown -> percent is None).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    context_window: int = 0

    def context_percent(self) -> float | None:
        """Fraction of the context window consumed by the prompt, as a percent.

        Uses ``input_tokens`` (the prompt) against ``context_window``. Returns
        ``None`` when the window size is unknown (0). Result is clamped to
        [0, 100].
        """
        if self.context_window <= 0:
            return None
        pct = 100.0 * self.input_tokens / self.context_window
        return max(0.0, min(100.0, pct))


def _get(obj: Any, *names: str) -> int:
    """Fetch the first present attribute/key in ``names`` as an int (else 0)."""
    for name in names:
        val = None
        if isinstance(obj, dict):
            val = obj.get(name)
        else:
            val = getattr(obj, name, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return 0


def usage_from_response(resp: Any, context_window: int = 0) -> Usage | None:
    """Extract a :class:`Usage` from any provider's raw response object.

    Handles OpenAI (``resp.usage.prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens``), Anthropic (``resp.usage.input_tokens`` /
    ``output_tokens``), and Google (``resp.usage_metadata.prompt_token_count``
    / ``candidates_token_count`` / ``total_token_count``). Returns ``None`` when
    no usage block is present.
    """
    block = getattr(resp, "usage", None)
    if block is None and isinstance(resp, dict):
        block = resp.get("usage")
    if block is None:
        block = getattr(resp, "usage_metadata", None)
        if block is None and isinstance(resp, dict):
            block = resp.get("usage_metadata")
    if block is None:
        return None
    inp = _get(block, "input_tokens", "prompt_tokens", "prompt_token_count")
    out = _get(
        block, "output_tokens", "completion_tokens", "candidates_token_count"
    )
    total = _get(block, "total_tokens", "total_token_count")
    if total == 0:
        total = inp + out
    return Usage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=total,
        context_window=context_window,
    )


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
    usage: Usage | None = None


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
    # Model context window in tokens (0 = unknown; set via *_CONTEXT_WINDOW).
    context_window: int = 0


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
        ctx_raw = os.environ.get(f"{prefix}_CONTEXT_WINDOW", "")
        try:
            ctx = int(ctx_raw)
        except (TypeError, ValueError):
            ctx = 0
        return ProviderConfig(
            name=name,
            api_url=url.rstrip("/"),
            api_key=key,
            model=model,
            context_window=ctx,
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

# Fallback context windows for well-known models, used only when neither the
# *_CONTEXT_WINDOW env var nor the provider API supplies one. Matched as a
# case-insensitive substring of the model id, most specific entries first.
_KNOWN_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    # Anthropic (normally auto-detected via max_input_tokens).
    ("claude", 200_000),
    # OpenAI (the standard API does not report context size).
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("gpt-3.5", 16_385),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4", 200_000),
    # Google (normally auto-detected via input_token_limit).
    ("gemini-1.5-pro", 2_097_152),
    ("gemini-1.5", 1_048_576),
    ("gemini-2", 1_048_576),
    ("gemini", 1_048_576),
)


def _known_context_window(model: str) -> int:
    """Return a fallback context window for ``model`` (0 if not recognized)."""
    m = (model or "").lower()
    for needle, window in _KNOWN_CONTEXT_WINDOWS:
        if needle in m:
            return window
    return 0


def _first_positive_int(obj: Any, *names: str) -> int:
    """Return the first positive integer among the named attrs/extra fields."""
    extra = getattr(obj, "model_extra", None) or {}
    for name in names:
        val = getattr(obj, name, None)
        if val is None and isinstance(extra, dict):
            val = extra.get(name)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return 0


class LLMClient:
    """Base class: a client that can complete a chat and verify connectivity."""

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.config = config
        self.timeout = timeout
        # Usage from the most recent request (set by complete/converse).
        self._last_usage: Usage | None = None
        # Context window learned from the provider's models API (0 = not yet
        # detected / unavailable). See :meth:`detect_context_window`.
        self._detected_context_window: int = 0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def context_window(self) -> int:
        """The model's max input context window in tokens (0 = unknown).

        Precedence: an explicit ``*_CONTEXT_WINDOW`` env override, then a value
        auto-detected from the provider API, then a small table of well-known
        models. 0 means unknown, and the context-usage badge is hidden.
        """
        if self.config.context_window > 0:
            return self.config.context_window
        if self._detected_context_window > 0:
            return self._detected_context_window
        return _known_context_window(self.config.model)

    def detect_context_window(self) -> int:
        """Best-effort query of the model's max input context window.

        Returns 0 when the provider cannot report it. Overridden per provider;
        implementations must never raise (return 0 on any failure).
        """
        return 0

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
        return Turn(
            text=text,
            assistant_message={"role": "assistant", "content": text},
            usage=self._last_usage,
        )

    def tool_result_message(self, tool_call_id: str, content: str) -> Message:
        """Build the message that reports a tool's output back to the model."""
        return {"role": "user", "content": content}

    def verify(self) -> None:
        """Make a minimal request to confirm URL, key, and model all work.

        Raises :class:`LLMError` on any failure. On success, and only when the
        context window was not pinned via ``*_CONTEXT_WINDOW``, it also tries to
        auto-detect the model's context window (never fatal).
        """
        self.complete([{"role": "user", "content": "ping"}], max_tokens=1)
        if self.config.context_window <= 0 and self._detected_context_window <= 0:
            try:
                self._detected_context_window = self.detect_context_window() or 0
            except Exception:
                self._detected_context_window = 0


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

    def detect_context_window(self) -> int:
        # The standard OpenAI API doesn't report context size, but many
        # OpenAI-compatible servers do (e.g. vLLM's ``max_model_len``).
        try:
            info = self._client.models.retrieve(self.config.model)
        except Exception:
            return 0
        return _first_positive_int(
            info,
            "max_model_len",
            "max_context_length",
            "context_length",
            "context_window",
            "max_input_tokens",
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
        self._last_usage = usage_from_response(resp, self.context_window)
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

    def detect_context_window(self) -> int:
        # Anthropic's models API reports ``max_input_tokens`` (context window).
        try:
            info = self._client.models.retrieve(self.config.model)
        except Exception:
            return 0
        return _first_positive_int(info, "max_input_tokens")

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
        self._last_usage = usage_from_response(resp, self.context_window)
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
            usage=usage_from_response(resp, self.context_window),
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

    def detect_context_window(self) -> int:
        # Gemini's model metadata reports ``input_token_limit``.
        try:
            info = self._client.models.get(model=self.config.model)
        except Exception:
            return 0
        return _first_positive_int(info, "input_token_limit")

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
        self._last_usage = usage_from_response(resp, self.context_window)
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
