"""LLM Client implementation.

Supports multiple providers: OpenAI, DeepSeek, Azure OpenAI, etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.core.config import LLMConfig


class LLMClient:
    """LLM client wrapper supporting multiple providers."""

    def __init__(self, config: LLMConfig):
        """Initialize LLM client with configuration.

        Args:
            config: LLM configuration.
        """
        self.config = config
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI-compatible client."""
        if self._client is None:
            client_kwargs: Dict[str, Any] = {}

            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key

            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url

            self._client = OpenAI(**client_kwargs)

        return self._client

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat message and get response.

        Args:
            message: User message.
            system_prompt: Optional system prompt.
            temperature: Override temperature.
            max_tokens: Override max tokens.

        Returns:
            LLM response text.
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )

        return response.choices[0].message.content or ""

    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send chat with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override temperature.
            max_tokens: Override max tokens.

        Returns:
            LLM response text.
        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )

        return response.choices[0].message.content or ""

    def __call__(self, message: str) -> str:
        """Make client callable for convenience.

        Args:
            message: User message.

        Returns:
            LLM response text.
        """
        return self.chat(message)


def create_llm_client(config: Optional[LLMConfig] = None) -> Optional[LLMClient]:
    """Create LLM client from configuration.

    Args:
        config: LLM configuration. If None, loads from settings.

    Returns:
        LLM client instance, or None if no valid config.
    """
    if config is None:
        from src.core.config import load_settings

        settings = load_settings()
        config = settings.llm

    if not config.api_key:
        return None

    return LLMClient(config)
