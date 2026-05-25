from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM — OpenRouter (OpenAI-compatible)
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    # Cheap model for bulk job-fit screening
    screen_model: str = Field(default="anthropic/claude-haiku-4-5")
    # Accurate model for multi-step form parsing
    form_model: str = Field(default="anthropic/claude-sonnet-4-6")

    # Browser
    user_data_dir: Path = Field(
        default=ROOT_DIR / "browser_profile",
        description="Persistent Playwright profile directory (cookies/session live here)",
    )
    headless: bool = Field(default=True)

    # Search parameters
    target_titles: list[str] = Field(
        default=["Software Engineer"],
        description="Job titles to search for",
    )
    target_locations: list[str] = Field(
        default=["Remote"],
        description="Locations to search in",
    )

    # Run limits
    max_applications_per_run: int = Field(default=10, ge=1, le=100)
    max_pages_per_search: int = Field(default=5, ge=1, le=20)

    # Platforms
    enabled_platforms: list[str] = Field(
        default=["indeed"],
        description="Which platforms to run: indeed, naukri, wellfound",
    )

    # LLM thresholds
    fit_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum fit score [0,1] required to attempt application",
    )

    # Dashboard sync
    dashboard_api_url: str = Field(
        default="",
        description="Vercel dashboard base URL, e.g. https://job-dashboard.vercel.app",
    )
    dashboard_api_key: str = Field(
        default="",
        description="Shared secret for dashboard API auth (x-api-key header)",
    )

    # India-only locations — roles in these are applied regardless of type
    # Roles outside this list are only applied if they are remote
    india_locations: list[str] = Field(
        default=["bangalore", "bengaluru", "mumbai", "delhi", "hyderabad",
                 "chennai", "pune", "india", "kolkata", "noida", "gurgaon",
                 "gurugram", "ahmedabad"],
        description="Lowercase location names considered 'within India'",
    )

    # File paths (derived, not from .env)
    @property
    def db_path(self) -> Path:
        return ROOT_DIR / "data" / "jobs.db"

    @property
    def log_path(self) -> Path:
        return ROOT_DIR / "logs" / "agent.log"

    @property
    def sample_profile_path(self) -> Path:
        return ROOT_DIR / "data" / "sample_profile.json"

    @field_validator("target_titles", "target_locations", "enabled_platforms", mode="before")
    @classmethod
    def _split_csv(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("enabled_platforms", mode="after")
    @classmethod
    def _validate_platforms(cls, v: list[str]) -> list[str]:
        allowed = {"indeed", "naukri", "wellfound"}
        for platform in v:
            if platform not in allowed:
                raise ValueError(f"Unknown platform '{platform}'. Allowed: {allowed}")
        return v


settings = Settings()
