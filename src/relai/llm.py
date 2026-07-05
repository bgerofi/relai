"""LLM provider clients.

relai talks to one of three providers, selected from a triplet of variables per
provider:

    OpenAI:     OPENAI_API_URL     OPENAI_API_KEY     OPENAI_MODEL
    Anthropic:  ANTHROPIC_API_URL  ANTHROPIC_API_KEY  ANTHROPIC_MODEL
    Google:     GOOGLE_API_URL     GOOGLE_API_KEY     GOOGLE_MODEL
    Custom:     CUSTOM_API_URL     CUSTOM_API_KEY     CUSTOM_MODEL

These variables are read from the process environment and, as a fallback, from a
``~/.relai/llm.conf`` file (simple ``KEY=VALUE`` lines). Environment variables
take precedence over the file, so the file provides defaults that can still be
overridden per-invocation.

The "openai" and "custom" providers use the official ``openai`` SDK (custom just
points ``base_url`` at an OpenAI-compatible server: LM Studio, llama.cpp, vLLM,
Ollama's OpenAI shim, gateways, ...). The "anthropic" provider uses the official
``anthropic`` SDK, and "google" uses the ``google-genai`` (Gemini) SDK.

A provider is considered "configured" only when all three of its variables are
set; if several are configured, one is chosen by a fixed precedence
(custom > google > anthropic > openai).

Two optional settings tune request behaviour (env or ``~/.relai/llm.conf``):
``RELAI_LLM_TIMEOUT`` (seconds per request, default 30) and
``RELAI_LLM_MAX_RETRIES`` (retries on timeout / dropped connection / rate limit /
5xx, default 2). relai owns the retry loop so it can report each retry in the UI.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

#: How long (seconds) to wait on any single LLM request.
DEFAULT_TIMEOUT = 30.0

#: How many times to retry a transient LLM failure (timeout, dropped
#: connection, rate limit, 5xx) before giving up.
DEFAULT_MAX_RETRIES = 2

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
    """Raised when no provider is fully configured."""


def _root_cause(exc: BaseException) -> BaseException | None:
    """Return the deepest chained cause of ``exc`` (``None`` if it has none).

    SDK errors often wrap the real failure (e.g. an ``httpx.ReadTimeout``); the
    root cause usually names what actually went wrong.
    """
    seen = {id(exc)}
    cur = exc.__cause__ or exc.__context__
    root: BaseException | None = None
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        root = cur
        cur = cur.__cause__ or cur.__context__
    return root


def _describe_request_error(
    name: str, exc: BaseException, elapsed: float, timeout: float
) -> str:
    """Build a detailed one-line description of a failed provider request.

    Shown to the user in the AI panel, so it stays on a single line. It surfaces
    the exception type, how long the call ran versus the configured timeout, any
    HTTP status / request-id the SDK exposes, and the underlying cause (e.g. the
    ``httpx`` timeout hiding behind an SDK wrapper) -- enough to tell a genuine
    timeout apart from an auth error, a rate limit, or a bad endpoint.
    """
    header = f"{name} request failed after {elapsed:.1f}s (timeout {timeout:.0f}s)"

    cls = type(exc)
    type_name = cls.__name__
    module = (getattr(cls, "__module__", "") or "").split(".")[0]
    if module and module not in ("builtins", "__main__"):
        type_name = f"{module}.{type_name}"

    text = str(exc).strip()
    body = f"{type_name}: {text}" if text else type_name

    root = _root_cause(exc)
    if root is not None:
        root_text = str(root).strip()
        if root_text and root_text != text:
            body += f" (cause: {type(root).__name__}: {root_text})"

    meta = []
    status = getattr(exc, "status_code", None)
    if status is not None:
        meta.append(f"status={status}")
    request_id = getattr(exc, "request_id", None)
    if request_id:
        meta.append(f"request_id={request_id}")

    msg = f"{header}: {body}"
    if meta:
        msg += f" [{' '.join(meta)}]"
    return msg


#: Exception class names (openai / anthropic SDKs share these) and HTTP status
#: codes that indicate a transient failure worth retrying.
_RETRYABLE_TYPES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServerError",
        "ServiceUnavailableError",
    }
)
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transient failure worth retrying."""
    if type(exc).__name__ in _RETRYABLE_TYPES:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _RETRYABLE_STATUS


def _is_rate_limit(exc: BaseException) -> bool:
    """True if ``exc`` is a rate-limit failure (HTTP 429 / ``RateLimitError``)."""
    if type(exc).__name__ == "RateLimitError":
        return True
    return getattr(exc, "status_code", None) == 429


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Return the server-requested wait from a ``Retry-After`` response header.

    Rate-limit (429) and some 503 responses tell the client exactly how long to
    wait, either as an integer number of seconds or as an HTTP date. The SDKs
    expose the raw response headers (``exc.response.headers``); we also accept a
    plain ``retry_after`` attribute. Returns ``None`` when no usable value is
    present, and clamps the result to a sane [0, 300]s range.
    """
    raw = None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:
            raw = None
    if raw is None:
        raw = getattr(exc, "retry_after", None)
    if raw is None:
        return None

    # Numeric form: whole/fractional seconds.
    try:
        secs = float(raw)
    except (TypeError, ValueError):
        secs = None
    if secs is None:
        # HTTP-date form: parse and take the delta from now.
        try:
            from email.utils import parsedate_to_datetime

            when = parsedate_to_datetime(str(raw))
        except Exception:
            return None
        if when is None:
            return None
        import datetime as _dt

        now = _dt.datetime.now(when.tzinfo) if when.tzinfo else _dt.datetime.now()
        secs = (when - now).total_seconds()

    if secs < 0:
        secs = 0.0
    return min(secs, 300.0)


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

#: Location of the optional config file, read as a fallback for the provider
#: variables. Real environment variables always take precedence over it.
def _conf_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".relai", "llm.conf")


def _load_conf(path: str | None = None) -> dict[str, str]:
    """Parse ``~/.relai/llm.conf`` into a ``{KEY: VALUE}`` dict.

    The format is simple ``KEY=VALUE`` lines. Blank lines and lines starting
    with ``#`` are ignored, a leading ``export`` is allowed, and matching single
    or double quotes around the value are stripped. Returns an empty dict when
    the file is absent or cannot be read (this is only a convenience fallback).
    """
    if path is None:
        path = _conf_path()
    values: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def _getvar(conf: dict[str, str], name: str) -> str | None:
    """Return env var ``name`` if set, else its value from ``conf`` (or None)."""
    if name in os.environ:
        return os.environ[name]
    return conf.get(name)


def _read_provider(name: str, conf: dict[str, str]) -> ProviderConfig | None:
    """Return a ProviderConfig if all three variables for ``name`` are set.

    Each variable is taken from the environment, falling back to ``conf`` (the
    parsed ``~/.relai/llm.conf``); the environment always wins.
    """
    prefix = _ENV_PREFIX[name]
    url = _getvar(conf, f"{prefix}_API_URL")
    key = _getvar(conf, f"{prefix}_API_KEY")
    model = _getvar(conf, f"{prefix}_MODEL")
    if url and key and model:
        ctx_raw = _getvar(conf, f"{prefix}_CONTEXT_WINDOW") or ""
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
    """Select a provider from the environment / ``~/.relai/llm.conf``.

    Returns ``None`` if no provider is fully configured.
    """
    conf = _load_conf()
    for name in _PROVIDER_ORDER:
        cfg = _read_provider(name, conf)
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

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self.config = config
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        # Optional progress hook, called with a short human-readable note before
        # each retry so the UI can report what relai is doing while it waits.
        self.on_retry: Callable[[str], None] | None = None
        # Usage from the most recent request (set by complete/converse).
        self._last_usage: Usage | None = None
        # Context window learned from the provider's models API (0 = not yet
        # detected / unavailable). See :meth:`detect_context_window`.
        self._detected_context_window: int = 0

    def _request(self, call: Callable[[], Any], *, what: str = "request") -> Any:
        """Run one provider API ``call``, retrying transient failures.

        Retries up to ``self.max_retries`` times on timeouts, dropped
        connections, rate limits and 5xx responses, with exponential backoff
        plus jitter. A server-sent ``Retry-After`` header (rate limits / some
        503s) overrides the computed delay so we wait exactly as instructed.
        Before each retry the ``on_retry`` hook (if set) is called so the UI can
        report the wait. Non-retryable errors, and the final attempt's error,
        are wrapped in :class:`LLMError` with full diagnostics.
        """
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            start = time.monotonic()
            try:
                return call()
            except Exception as exc:  # SDK raises its own error hierarchy
                elapsed = time.monotonic() - start
                if attempt >= attempts or not _is_retryable(exc):
                    raise LLMError(
                        _describe_request_error(
                            self.name, exc, elapsed, self.timeout
                        )
                    ) from exc
                # Exponential backoff with jitter, but honor a server-sent
                # Retry-After header when present (rate limits / some 503s tell
                # us exactly how long to wait).
                backoff = min(0.5 * (2 ** (attempt - 1)), 8.0)
                delay = backoff + random.uniform(0.0, backoff / 2.0)
                rate_limited = _is_rate_limit(exc)
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None:
                    delay = retry_after
                if self.on_retry is not None:
                    if rate_limited:
                        wait_note = (
                            f"Retry-After {delay:.0f}s"
                            if retry_after is not None
                            else f"backing off {delay:.0f}s"
                        )
                        self.on_retry(
                            f"{self.name} {what} rate limited "
                            f"(HTTP 429 after {elapsed:.0f}s); waiting "
                            f"{wait_note}, retry {attempt}/{self.max_retries}"
                        )
                    else:
                        self.on_retry(
                            f"{self.name} {what} failed "
                            f"({type(exc).__name__} after {elapsed:.0f}s); "
                            f"retrying {attempt}/{self.max_retries} in {delay:.0f}s"
                        )
                time.sleep(delay)
        # Unreachable (the loop always returns or raises), but keeps type
        # checkers happy about the function always returning a value.
        raise LLMError(f"{self.name} request failed")

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
        on_text: Callable[[str], None] | None = None,
    ) -> Turn:
        """One round-trip that may request tool calls.

        The default implementation has no tool support: it just wraps
        :meth:`complete` as a text-only turn. Providers that support tools
        override this.

        ``on_text``, when given, is a progress hook fed the assistant's answer
        text as it is produced so the UI can show the model's narration live.
        Providers that stream call it repeatedly with the accumulated text; this
        text-only default has no stream, so it fires once with the final text.
        """
        text = self.complete(messages, max_tokens=max_tokens)
        if on_text is not None and text:
            on_text(text)
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

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        super().__init__(config, timeout, max_retries)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError("the 'openai' package is required but not installed") from exc

        base_url = config.api_url
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        # relai owns retries (see LLMClient._request) so it can report them; tell
        # the SDK not to retry on its own.
        self._client = OpenAI(
            api_key=config.api_key, base_url=base_url, timeout=timeout, max_retries=0
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
        resp = self._request(
            lambda: self._client.chat.completions.create(
                model=self.config.model,
                messages=list(messages),
                max_tokens=max_tokens,
            )
        )
        self._last_usage = usage_from_response(resp, self.context_window)
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response from {self.name}: {resp!r}") from exc


class AnthropicClient(LLMClient):
    """Anthropic client via the ``anthropic`` SDK."""

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        super().__init__(config, timeout, max_retries)
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
        # relai owns retries (see LLMClient._request) so it can report them; tell
        # the SDK not to retry on its own.
        self._client = Anthropic(
            api_key=config.api_key, base_url=base_url, timeout=timeout, max_retries=0
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
        resp = self._request(lambda: self._client.messages.create(**kwargs))
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
        on_text: Callable[[str], None] | None = None,
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
        if on_text is None:
            resp = self._request(lambda: self._client.messages.create(**kwargs))
            return self._turn_from_message(resp)

        # Streaming path: emit the assistant's text deltas as they arrive so the
        # UI can show the model's narration live, then assemble the final message
        # (which also carries any tool_use blocks and the usage totals).
        def _stream() -> Any:
            acc: list[str] = []
            with self._client.messages.stream(**kwargs) as stream:
                for delta in stream.text_stream:
                    acc.append(delta)
                    on_text("".join(acc))
                return stream.get_final_message()

        resp = self._request(_stream)
        return self._turn_from_message(resp)

    def _turn_from_message(self, resp: Any) -> Turn:
        """Build a :class:`Turn` from an Anthropic message (create or stream)."""
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

    def __init__(self, config: ProviderConfig, timeout: float = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        super().__init__(config, timeout, max_retries)
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

        resp = self._request(
            lambda: self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        )
        try:
            text = resp.text
        except (AttributeError, TypeError) as exc:
            raise LLMError(f"unexpected response from {self.name}: {resp!r}") from exc
        self._last_usage = usage_from_response(resp, self.context_window)
        return text or ""


def _client_for(
    config: ProviderConfig, timeout: float, max_retries: int
) -> LLMClient:
    if config.name == "anthropic":
        return AnthropicClient(config, timeout, max_retries)
    if config.name == "google":
        return GoogleClient(config, timeout, max_retries)
    # "openai" and "custom" both use the OpenAI SDK.
    return OpenAIClient(config, timeout, max_retries)


def _resolve_settings(conf: dict[str, str]) -> tuple[float, int]:
    """Read the request timeout and retry count from env / ``~/.relai/llm.conf``.

    ``RELAI_LLM_TIMEOUT`` is in seconds, ``RELAI_LLM_MAX_RETRIES`` a count; each
    falls back to its module default when unset or unparseable.
    """
    timeout = DEFAULT_TIMEOUT
    raw_timeout = _getvar(conf, "RELAI_LLM_TIMEOUT")
    if raw_timeout:
        try:
            parsed = float(raw_timeout)
            if parsed > 0:
                timeout = parsed
        except ValueError:
            pass

    max_retries = DEFAULT_MAX_RETRIES
    raw_retries = _getvar(conf, "RELAI_LLM_MAX_RETRIES")
    if raw_retries:
        try:
            parsed_int = int(raw_retries)
            if parsed_int >= 0:
                max_retries = parsed_int
        except ValueError:
            pass

    return timeout, max_retries


def create_client(
    timeout: float | None = None, max_retries: int | None = None
) -> LLMClient:
    """Resolve config from the environment / ``~/.relai/llm.conf`` and build the
    matching client.

    The request ``timeout`` (seconds) and ``max_retries`` come from
    ``RELAI_LLM_TIMEOUT`` / ``RELAI_LLM_MAX_RETRIES`` (env or ``~/.relai/llm.conf``)
    unless passed explicitly.

    Raises :class:`LLMNotConfigured` if no provider is fully configured.
    """
    config = resolve_config()
    if config is None:
        raise LLMNotConfigured(
            "no LLM provider configured; set the API_URL, API_KEY and MODEL "
            "variables for OPENAI, ANTHROPIC, GOOGLE, or CUSTOM (in the "
            "environment or in ~/.relai/llm.conf)"
        )
    conf_timeout, conf_retries = _resolve_settings(_load_conf())
    if timeout is None:
        timeout = conf_timeout
    if max_retries is None:
        max_retries = conf_retries
    return _client_for(config, timeout, max_retries)
