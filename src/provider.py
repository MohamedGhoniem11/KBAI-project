import os
from dataclasses import dataclass
from enum import Enum


class Provider(str, Enum):
    GROQ = "groq"
    XAI = "xai"
    OLLAMA = "ollama"
    OPENAI = "openai"


@dataclass
class ProviderConfig:
    api_key_env: str
    api_url: str
    default_model: str


PROVIDER_REGISTRY: dict[Provider, ProviderConfig] = {
    Provider.GROQ: ProviderConfig(
        api_key_env="GROQ_API_KEY",
        api_url="https://api.groq.com/openai/v1/chat/completions",
        default_model="llama-3.3-70b-versatile",
    ),
    Provider.XAI: ProviderConfig(
        api_key_env="XAI_API_KEY",
        api_url="https://api.x.ai/v1/chat/completions",
        default_model="grok-3-latest",
    ),
    Provider.OLLAMA: ProviderConfig(
        api_key_env="",  # no key needed
        api_url="http://localhost:11434/v1/chat/completions",
        default_model="llama3.2:3b",
    ),
}


def get_provider() -> Provider:
    raw = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    try:
        return Provider(raw)
    except ValueError:
        return Provider.GROQ


def get_provider_config(provider: Provider | None = None) -> ProviderConfig:
    if provider is None:
        provider = get_provider()
    return PROVIDER_REGISTRY[provider]


def get_api_key() -> str | None:
    cfg = get_provider_config()
    if not cfg.api_key_env:
        return None
    key = os.getenv(cfg.api_key_env)
    if not key or "your_" in key:
        return None
    return key


def get_default_model() -> str:
    raw = os.getenv("LLM_MODEL", "")
    if raw:
        return raw
    return get_provider_config().default_model
