"""Central configuration, driven entirely by environment variables.

Copy .env.example to .env and edit, or just export the variables in your shell.
Everything has a sensible default so the agent runs out of the box.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency).

    Only sets variables that are not already present in the environment, so an
    explicit `export FOO=bar` always wins over the file.
    """
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass
class DeepSeekConfig:
    """The planner brain: DeepSeek's OpenAI-compatible chat API."""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class VisionConfig:
    """The inference backend that judges camera photos.

    A cloud multimodal (vision) LLM served over an OpenAI-compatible API. The
    captured photo is sent directly to the model as a base64 image.

    Defaults to Kimi (Moonshot) vision. DeepSeek's own API is text-only, so the
    photo-judging step uses Kimi. temperature < 0 means "don't send it" (Kimi
    docs recommend not manually setting it for some models).
    """

    base_url: str = "https://api.moonshot.cn/v1"
    api_key: str = ""
    model: str = "moonshot-v1-8k-vision-preview"
    temperature: float = -1.0
    max_tokens: int = 384


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    warmup_frames: int = 8
    save_dir: Path = field(default_factory=lambda: Path("captures"))


@dataclass
class FlashlightConfig:
    script_path: Path = field(default_factory=lambda: Path("/home/truthg/light/flashlight.py"))
    python_bin: str = "python3"


@dataclass
class AgentConfig:
    deepseek: DeepSeekConfig
    vision: VisionConfig
    camera: CameraConfig
    flashlight: FlashlightConfig
    max_steps: int = 12


def load_config() -> AgentConfig:
    _load_dotenv()

    deepseek = DeepSeekConfig(
        # Fall back to OPENAI_API_KEY so an already-exported key just works.
        api_key=_env("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=float(_env("DEEPSEEK_TEMPERATURE", "0.2")),
        max_tokens=int(_env("DEEPSEEK_MAX_TOKENS", "1024")),
    )

    # Vision judge: Kimi (Moonshot) by default. VISION_API_KEY wins; otherwise
    # fall back to MOONSHOT_API_KEY, then the DeepSeek/OpenAI key.
    _shared_key = deepseek.api_key or os.environ.get("OPENAI_API_KEY", "")
    vision = VisionConfig(
        base_url=_env("VISION_BASE_URL", "https://api.moonshot.cn/v1"),
        api_key=_env("VISION_API_KEY", _env("MOONSHOT_API_KEY", _shared_key)),
        model=_env("VISION_MODEL", "moonshot-v1-8k-vision-preview"),
        temperature=float(_env("VISION_TEMPERATURE", "-1")),
        max_tokens=int(_env("VISION_MAX_TOKENS", "384")),
    )

    camera = CameraConfig(
        index=int(_env("CAMERA_INDEX", "0")),
        width=int(_env("CAMERA_WIDTH", "1280")),
        height=int(_env("CAMERA_HEIGHT", "720")),
        warmup_frames=int(_env("CAMERA_WARMUP", "8")),
        save_dir=Path(_env("CAPTURE_DIR", "captures")),
    )

    flashlight = FlashlightConfig(
        script_path=Path(_env("FLASHLIGHT_SCRIPT", "/home/truthg/light/flashlight.py")),
        python_bin=_env("FLASHLIGHT_PYTHON", "python3"),
    )

    return AgentConfig(
        deepseek=deepseek,
        vision=vision,
        camera=camera,
        flashlight=flashlight,
        max_steps=int(_env("AGENT_MAX_STEPS", "12")),
    )
