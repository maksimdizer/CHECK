import re
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from sqlalchemy import select, text

from . import settings
from .contracts import Contract
from .db import CloudCredential, EvaluationProfile

CloudProviderName = Literal["openai", "openrouter", "anthropic", "google"]


class LocalModel(Contract):
    model: str = Field(default="", max_length=200)
    output_tokens: int = Field(default=1200, ge=1, le=32768)
    context_tokens: int | None = Field(default=None, ge=512, le=262144)
    reasoning_tokens: int | None = Field(default=None, ge=1, le=65536)
    think: bool | None = None
    agent_mode: bool = False
    agent_quality: bool = False

    @model_validator(mode="after")
    def validate_budgets(self):
        if self.context_tokens and self.output_tokens >= self.context_tokens:
            raise ValueError("The context limit must exceed the output token limit.")
        if self.agent_quality and not self.agent_mode:
            raise ValueError("Quality mode requires agent mode.")
        return self


class CloudModel(Contract):
    provider: CloudProviderName = "openai"
    model: str = Field(default="", max_length=200)
    output_tokens: int = Field(default=2048, ge=16, le=32768)

    @model_validator(mode="after")
    def validate_google_model(self):
        if self.provider == "google" and self.model:
            model = self.model.removeprefix("models/")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
                raise ValueError("Enter a Gemini model ID from Google AI Studio.")
            self.model = model
        return self


class ModelProfile(Contract):
    mode: Literal["local", "cloud"] = "local"
    local: LocalModel = Field(default_factory=LocalModel)
    cloud: CloudModel = Field(default_factory=CloudModel)

    @property
    def model(self):
        return self.local.model if self.mode == "local" else self.cloud.model


class ModelProfileInput(ModelProfile):
    api_key: SecretStr | None = Field(default=None, max_length=4096)


def active_profile(session, org):
    row = session.get(EvaluationProfile, org)
    if row:
        return ModelProfile.model_validate(row.data)
    return ModelProfile(local=LocalModel(
        model=settings.MODEL,
        output_tokens=settings.OLLAMA_NUM_PREDICT,
        think=settings.OLLAMA_THINK,
        agent_mode=settings.QA_AGENT_MODE,
        agent_quality=settings.QA_AGENT_MODE and settings.QA_AGENT_QUALITY,
    ))


def public_profile(session, org, profile):
    providers = session.scalars(select(CloudCredential.provider).where(CloudCredential.org == org))
    return {**profile.model_dump(), "configured_cloud_providers": sorted(providers)}


def cloud_key(session, org, provider):
    row = session.get(CloudCredential, (org, provider))
    return row.api_key if row else None


def resolve_cloud_key(session, org, payload):
    value = payload.api_key.get_secret_value().strip() if payload.api_key else ""
    if value and (not value.isascii() or any(char.isspace() or not char.isprintable() for char in value)):
        raise ValueError("Enter a valid API key.")
    return value or cloud_key(session, org, payload.cloud.provider)


def save_profile(session, org, payload):
    session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": "model-profile:" + org})
    profile = ModelProfile.model_validate(payload.model_dump(exclude={"api_key"}))
    if profile.mode == "cloud":
        key = resolve_cloud_key(session, org, payload)
        if not profile.cloud.model or not key:
            raise ValueError("Enter a cloud model ID and API key.")
        session.merge(CloudCredential(org=org, provider=profile.cloud.provider, api_key=key))
    session.merge(EvaluationProfile(org=org, data=profile.model_dump()))
    return profile
