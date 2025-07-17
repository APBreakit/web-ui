
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    agent_type: str = "custom"
    llm_provider: str = "ollama"
    llm_model_name: str = "llama3"
    llm_temperature: float = 0.0
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    use_own_browser: bool = False
    keep_browser_open: bool = True
    headless: bool = True
    disable_security: bool = False
    window_w: int = 1280
    window_h: int = 720
    save_recording_path: Optional[str] = "recordings"
    save_agent_history_path: str = "history"
    save_trace_path: str = "traces"
    enable_recording: bool = False
    task: str = "What is the latest news about AI?"
    add_infos: list[str] = field(default_factory=list)
    max_steps: int = 20
    use_vision: bool = True
    max_actions_per_step: int = 5
    tool_calling_method: str = "tool_calling"
