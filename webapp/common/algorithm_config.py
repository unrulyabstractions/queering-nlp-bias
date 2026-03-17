"""Shared configuration for all sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from webapp.app_settings import DEFAULT_SETTINGS


@dataclass
class AlgorithmEvent:
    """Event yielded by algorithms to communicate with handlers."""

    type: str
    data: dict


# Providers that don't require API keys (local models)
LOCAL_PROVIDERS = {"huggingface"}


@dataclass
class SamplingConfig:
    """Configuration for all sampling algorithms."""

    gen_api_key: str
    judge_api_key: str
    gen_provider: str
    gen_model: str
    judge_provider: str
    judge_model: str
    temperature: float
    max_tokens: int
    judge_prompt: str

    def validate_api_keys(self, need_gen: bool = True, need_judge: bool = True) -> str | None:
        """Validate that required API keys are present.

        Args:
            need_gen: Whether generation API key is required
            need_judge: Whether judge API key is required

        Returns:
            Error message if validation fails, None if valid
        """
        gen_needs_key = need_gen and self.gen_provider not in LOCAL_PROVIDERS
        judge_needs_key = need_judge and self.judge_provider not in LOCAL_PROVIDERS

        if gen_needs_key and not self.gen_api_key:
            return f"Configure {self.gen_provider} API key in Settings"
        if judge_needs_key and not self.judge_api_key:
            return f"Configure {self.judge_provider} API key in Settings"
        return None

    @classmethod
    def from_request(cls, data: dict) -> SamplingConfig:
        s = data.get("settings", {})
        # Get API keys - support both old api_key and new api_keys format
        api_keys = data.get("api_keys", {})
        gen_provider = s.get("gen_provider", DEFAULT_SETTINGS["gen_provider"])
        judge_provider = s.get("judge_provider", DEFAULT_SETTINGS["judge_provider"])
        # Pick the right key for each provider
        gen_key = api_keys.get(gen_provider, data.get("api_key", ""))
        judge_key = api_keys.get(judge_provider, data.get("api_key", ""))
        return cls(
            gen_api_key=gen_key,
            judge_api_key=judge_key,
            gen_provider=gen_provider,
            gen_model=s.get("gen_model", DEFAULT_SETTINGS["gen_model"]),
            judge_provider=judge_provider,
            judge_model=s.get("judge_model", DEFAULT_SETTINGS["judge_model"]),
            temperature=s.get("temperature", DEFAULT_SETTINGS["temperature"]),
            max_tokens=s.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]),
            judge_prompt=s.get("judge_prompt", DEFAULT_SETTINGS["judge_prompt"]),
        )
