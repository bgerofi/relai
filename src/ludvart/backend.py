"""Build (and tear down) an LLM backend from a model registration.

A registration (see :mod:`ludvart.models`) is turned into a live
:class:`~ludvart.llm.LLMClient` here. Direct providers just build a client;
GitHub Copilot additionally spins up the local LiteLLM gateway and points the
client at it. The result is a :class:`Backend` bundling the client with any
gateway that must be stopped when the backend is discarded.

This module is UI-agnostic: progress is reported through an optional ``status``
callback so the same code serves the CLI startup path and the in-panel
``/model`` commands. Interactive GitHub device-flow authorization is *not* done
here -- Copilot must already be authorized (that one-time step lives in the CLI
setup wizard); building a Copilot backend when unauthorized raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from .llm import LLMClient, build_client, copilot_provider_config
from .models import (
    Registration,
    active_index,
    add_registration,
    is_copilot,
    label,
    registration_to_config,
    remove_registration,
    save_models,
    set_active,
)

if TYPE_CHECKING:
    from .gateway import CopilotGateway

StatusFn = Callable[[str], None]


@dataclass
class Backend:
    """A live client plus any gateway that owns its lifetime."""

    client: LLMClient
    gateway: "CopilotGateway | None" = None

    def stop(self) -> None:
        """Stop the gateway, if any. Safe to call more than once."""
        if self.gateway is not None:
            try:
                self.gateway.stop()
            except Exception:
                pass
            self.gateway = None


def build_backend(reg: Registration, *, status: StatusFn | None = None) -> Backend:
    """Build a :class:`Backend` for ``reg`` (starting the gateway for Copilot).

    Raises :class:`~ludvart.llm.LLMError` /
    :class:`~ludvart.gateway.GatewayError` on failure. The returned backend is
    *not* verified; callers run :meth:`Backend.client.verify` (or
    :func:`verify_backend`) themselves so the check can be reported/backgrounded.
    """
    if is_copilot(reg):
        return _build_copilot(reg, status)
    return Backend(build_client(registration_to_config(reg)), None)


def verify_backend(backend: Backend) -> None:
    """Confirm the backend's client works (raises on failure)."""
    backend.client.verify()


def _build_copilot(reg: Registration, status: StatusFn | None) -> Backend:
    from .gateway import (
        GATEWAY_API_KEY,
        CopilotGateway,
        GatewayError,
        copilot_authenticated,
        litellm_available,
    )

    model = reg["model"]
    if not litellm_available():
        raise GatewayError(
            "the LiteLLM gateway isn't installed; re-run ./setup.sh or "
            "uv pip install 'litellm[proxy]'"
        )
    if not copilot_authenticated():
        raise GatewayError(
            "GitHub Copilot isn't authorized yet; run `ludvart` in a terminal "
            "and add the Copilot model through the setup wizard once"
        )
    gateway = CopilotGateway(model)
    if status is not None:
        status(f"starting the GitHub Copilot gateway (model {model!r})...")
    gateway.start()
    config = copilot_provider_config(
        gateway.base_url, gateway.litellm_model, GATEWAY_API_KEY
    )
    return Backend(build_client(config), gateway)


class ModelManager:
    """Live view of the registered models: which is active, which are usable.

    Wraps the on-disk registry (:mod:`ludvart.models`) with the running client
    and any Copilot gateway, and drives the ``/model`` panel commands: switch
    the active model, add a new one (verified), or remove one. Changes are
    persisted back to ``models.json``. Network work (verify / gateway start) is
    synchronous here; callers run it on a background worker.
    """

    def __init__(
        self,
        models: list[Registration],
        available: list[bool],
        client: LLMClient,
        gateway: "CopilotGateway | None" = None,
    ) -> None:
        self.models = models
        self.available = available
        self.client = client
        self.gateway = gateway

    def active_index(self) -> int | None:
        return active_index(self.models)

    def _persist(self) -> None:
        try:
            save_models(self.models)
        except Exception:
            pass

    def describe(self) -> list[str]:
        """One display line per registered model for ``/model list``."""
        active = self.active_index()
        lines: list[str] = []
        for i, reg in enumerate(self.models):
            marks = []
            if i == active:
                marks.append("in use")
            marks.append("available" if self.available[i] else "unavailable")
            lines.append(f"  {i + 1}) {label(reg)}  [{', '.join(marks)}]")
        return lines

    def use(self, index: int, *, status: StatusFn | None = None) -> tuple[bool, str]:
        """Switch the active model to ``index`` (verifying it first).

        ``status`` (optional) receives short progress messages while the backend
        is built -- notably the Copilot gateway launch -- so callers can show it.
        """
        reg = self.models[index]
        if reg.get("active"):
            return True, f"Already using {label(reg)}."
        try:
            backend = build_backend(reg, status=status)
            verify_backend(backend)
        except Exception as exc:
            self.available[index] = False
            return False, f"Could not switch to {label(reg)}: {exc}"
        old = self.gateway
        self.client = backend.client
        self.gateway = backend.gateway
        if old is not None and old is not backend.gateway:
            try:
                old.stop()
            except Exception:
                pass
        self.models = set_active(self.models, index)
        self.available[index] = True
        self._persist()
        return True, f"Now using {label(reg)}."

    def add(self, reg: Registration, *, status: StatusFn | None = None) -> tuple[bool, str]:
        """Verify a new registration and append it (without switching to it).

        ``status`` (optional) receives short progress messages while the backend
        is built (e.g. the Copilot gateway launch used to verify it).
        """
        backend: Backend | None = None
        try:
            backend = build_backend(reg, status=status)
            verify_backend(backend)
        except Exception as exc:
            if backend is not None:
                backend.stop()
            return False, f"Verification failed: {exc}"
        # Verified only; we don't switch, so release any gateway we spun up.
        backend.stop()
        self.models = add_registration(self.models, reg, make_active=False)
        self.available.append(True)
        self._persist()
        return True, f"Added {label(reg)} (verified)."

    def remove(self, index: int) -> tuple[bool, str]:
        """Unregister model ``index`` (not allowed for the model in use)."""
        reg = self.models[index]
        if reg.get("active"):
            return False, "Cannot remove the model in use; /model use another first."
        self.models = remove_registration(self.models, index)
        del self.available[index]
        self._persist()
        return True, f"Removed {label(reg)}."

