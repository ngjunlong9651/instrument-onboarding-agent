from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("AGENT_DATA_DIR", PROJECT_ROOT / ".agent_data")))
    release: str = field(default_factory=lambda: os.getenv("AGENT_RELEASE", "dev"))

    # LLM planner. Without an API key (or with AGENT_OFFLINE=1) a deterministic planner is used.
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "claude-sonnet-5-5"))
    cost_per_mtok_in: float = field(default_factory=lambda: _env_float("AGENT_COST_PER_MTOK_IN", 3.0))
    cost_per_mtok_out: float = field(default_factory=lambda: _env_float("AGENT_COST_PER_MTOK_OUT", 15.0))
    planner_max_attempts: int = 2

    # Tools
    parallel_tools: bool = field(default_factory=lambda: os.getenv("AGENT_PARALLEL_TOOLS", "1") == "1")
    tool_max_retries: int = 2
    tool_backoff_s: float = 0.05

    # Guards
    price_min: float = 1.0
    price_max: float = 300.0
    price_tolerance_pct: float = 0.5
    max_price_age_hours: float = 24.0

    # Circuit breaker on vendor feeds
    breaker_window: int = 10
    breaker_min_events: int = 3
    breaker_failure_rate: float = 0.5

    # Alerts (evaluated over the last `alert_window` runs)
    alert_window: int = 50
    alert_min_completion_rate: float = 0.6
    alert_max_tool_error_rate: float = 0.2
    alert_max_p95_ms: float = 5000.0

    @property
    def use_llm(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("AGENT_OFFLINE") != "1"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def observability_db(self) -> Path:
        return self.data_dir / "observability.sqlite"


def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
