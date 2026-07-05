"""Model clients: uniform interface over local llama.cpp and remote Fireworks AI."""

from routing_agent.clients.base import (
    GenerationError,
    LocalModelUnavailable,
    ModelClient,
    RemoteModelError,
)
from routing_agent.clients.local import LocalGemmaClient
from routing_agent.clients.remote import FireworksClient

__all__ = [
    "FireworksClient",
    "GenerationError",
    "LocalGemmaClient",
    "LocalModelUnavailable",
    "ModelClient",
    "RemoteModelError",
]
