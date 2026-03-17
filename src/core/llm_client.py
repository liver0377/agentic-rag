"""LLM Client implementation.

Supports multiple providers: OpenAI, DeepSeek, Azure OpenAI, etc.
Integrates with Langfuse for token usage tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.config import LLMConfig

if TYPE_CHECKING:
    from langfuse import Langfuse


class LLMClient:
    """LLM client wrapper supporting multiple providers with Langfuse integration."""

    def __init__(
        self,
        config: LLMConfig,
        langfuse_client: Optional["Langfuse"] = None,
    ):
        """Initialize LLM client with configuration.

        Args:
            config: LLM configuration.
            langfuse_client: Optional Langfuse client for tracing.
        """
        self.config = config
        self.langfuse_client = langfuse_client
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        """Get or create OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI

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
        name: Optional[str] = None,
    ) -> str:
        """Send a chat message and get response.

        Args:
            message: User message.
            system_prompt: Optional system prompt.
            temperature: Override temperature.
            max_tokens: Override max tokens.
            name: Optional name for generation tracking.

        Returns:
            LLM response text.
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})

        generation_name = name or "llm-chat"

        if self.langfuse_client:
            return self._chat_with_tracing(messages, temperature, max_tokens, generation_name)
        else:
            return self._chat_without_tracing(messages, temperature, max_tokens)

    def _chat_without_tracing(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """Chat without Langfuse tracing."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _chat_with_tracing(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        name: str,
    ) -> str:
        """Chat with Langfuse tracing for token usage tracking."""
        lf_client = self.langfuse_client
        assert lf_client is not None

        with lf_client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=self.config.model,
            input={"messages": messages},
        ) as gen:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            )

            content = response.choices[0].message.content or ""
            usage = response.usage

            gen.update(
                output=content,
                usage_details={
                    "input": usage.prompt_tokens if usage else 0,
                    "output": usage.completion_tokens if usage else 0,
                    "total": usage.total_tokens if usage else 0,
                },
            )

            return content

    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        name: Optional[str] = None,
    ) -> str:
        """Send chat with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override temperature.
            max_tokens: Override max tokens.
            name: Optional name for generation tracking.

        Returns:
            LLM response text.
        """
        generation_name = name or "llm-chat-history"

        if self.langfuse_client:
            return self._chat_with_tracing(messages, temperature, max_tokens, generation_name)
        else:
            return self._chat_without_tracing(messages, temperature, max_tokens)

    def __call__(self, message: str) -> str:
        """Make client callable for convenience.

        Args:
            message: User message.

        Returns:
            LLM response text.
        """
        return self.chat(message)


def create_llm_client(
    config: Optional[LLMConfig] = None, langfuse_client: Optional["Langfuse"] = None
) -> Optional[LLMClient]:
    """Create LLM client from configuration.

    Args:
        config: LLM configuration. If None, loads from settings.
        langfuse_client: Optional Langfuse client for tracing.

    Returns:
        LLM client instance, or None if no valid config.
    """
    if config is None:
        from src.core.config import load_settings

        settings = load_settings()
        config = settings.llm

    if not config.api_key:
        return None

    return LLMClient(config, langfuse_client=langfuse_client)
