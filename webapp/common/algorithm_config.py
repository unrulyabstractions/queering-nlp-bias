"""Shared configuration for all sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from webapp.app_settings import DEFAULT_SETTINGS, LOCAL_PROVIDERS


@dataclass
class AlgorithmEvent:
    """Event yielded by algorithms to communicate with handlers."""

    type: str
    data: dict


@dataclass
class JudgeModelSpec:
    """Specification for a judge model including its provider."""

    provider: str
    model: str

    def to_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model}

    @classmethod
    def from_dict(cls, d: dict) -> JudgeModelSpec:
        return cls(provider=d["provider"], model=d["model"])


@dataclass
class SamplingConfig:
    """Configuration for all sampling algorithms."""

    gen_api_key: str
    api_keys: dict[str, str]  # All API keys by provider
    gen_provider: str
    gen_model: str
    judge_models: list[JudgeModelSpec]  # Multiple judge models from different providers
    gen_temperature: float
    judge_temperature: float
    gen_max_tokens: int | None  # None = use model's max context
    judge_max_tokens: int | None  # None = no limit, 0 also means no limit
    judge_prompt: str

    def get_judge_providers(self) -> set[str]:
        """Get all unique providers used by judge models."""
        return {m.provider for m in self.judge_models}

    def validate_api_keys(self, need_gen: bool = True, need_judge: bool = True) -> str | None:
        """Validate that required API keys are present."""
        if need_gen and self.gen_provider not in LOCAL_PROVIDERS and not self.gen_api_key:
            return f"Configure {self.gen_provider} API key in Settings"

        if need_judge:
            for provider in self.get_judge_providers():
                if provider not in LOCAL_PROVIDERS and not self.api_keys.get(provider):
                    return f"Configure {provider} API key in Settings"
        return None

    @classmethod
    def from_request(cls, data: dict) -> SamplingConfig:
        s = data.get("settings", {})
        api_keys = data.get("api_keys", {})
        gen_provider = s.get("gen_provider", DEFAULT_SETTINGS["gen_provider"])
        gen_key = api_keys.get(gen_provider, "")

        judge_model_raw = s.get("judge_model", DEFAULT_SETTINGS["judge_model"])
        if isinstance(judge_model_raw, list) and judge_model_raw and isinstance(judge_model_raw[0], dict):
            judge_models = [JudgeModelSpec.from_dict(m) for m in judge_model_raw]
        else:
            judge_models = [JudgeModelSpec(provider="openai", model="gpt-4o-mini")]

        gen_max_tokens_raw = s.get("max_tokens", DEFAULT_SETTINGS["max_tokens"])
        gen_max_tokens = None if gen_max_tokens_raw in (None, 0, "") else int(gen_max_tokens_raw)

        judge_max_tokens_raw = s.get("judge_max_tokens", 32)
        judge_max_tokens = None if judge_max_tokens_raw in (None, 0, "") else int(judge_max_tokens_raw)

        return cls(
            gen_api_key=gen_key,
            api_keys=api_keys,
            gen_provider=gen_provider,
            gen_model=s.get("gen_model", DEFAULT_SETTINGS["gen_model"]),
            judge_models=judge_models,
            gen_temperature=float(s.get("gen_temperature", DEFAULT_SETTINGS["gen_temperature"])),
            judge_temperature=float(s.get("judge_temperature", DEFAULT_SETTINGS["judge_temperature"])),
            gen_max_tokens=gen_max_tokens,
            judge_max_tokens=judge_max_tokens,
            judge_prompt=s.get("judge_prompt", DEFAULT_SETTINGS["judge_prompt"]),
        )
