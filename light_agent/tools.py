"""Tools the planner can call, plus their OpenAI tool schemas.

  * flashlight        -> drives flashlight.py (on/off/toggle/status) over ADB.
  * capture_and_check -> takes a webcam photo and asks the vLLM backend whether
                         the described goal is now visually satisfied.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

from . import camera
from .config import AgentConfig
from .vision import VisionJudge

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "flashlight",
            "description": (
                "Control the phone flashlight via ADB (flashlight.py). "
                "Use 'status' to read the current state, 'on'/'off' to set it, "
                "'toggle' to flip it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["on", "off", "toggle", "status"],
                        "description": "What to do with the flashlight.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_and_check",
            "description": (
                "Take a photo with the computer webcam and use the vision "
                "inference backend to verify whether a goal is visually achieved "
                "(e.g. 'the phone flashlight is emitting light'). Returns a "
                "verdict with achieved=true/false and a reason."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_description": {
                        "type": "string",
                        "description": (
                            "A concrete, visually checkable description of the "
                            "desired state, e.g. 'a bright flashlight beam is on'."
                        ),
                    }
                },
                "required": ["goal_description"],
            },
        },
    },
]


class Toolbox:
    def __init__(self, cfg: AgentConfig, fake_image: str | None = None) -> None:
        self.cfg = cfg
        self.judge = VisionJudge(cfg.vision)
        # When set, capture_and_check reads this file instead of the live camera.
        self.fake_image = fake_image

    def flashlight(self, action: str) -> dict[str, Any]:
        action = (action or "toggle").lower()
        if action not in {"on", "off", "toggle", "status"}:
            return {"ok": False, "error": f"invalid action '{action}'"}
        script = self.cfg.flashlight.script_path
        if not script.is_file():
            return {"ok": False, "error": f"flashlight script not found: {script}"}
        try:
            proc = subprocess.run(
                [self.cfg.flashlight.python_bin, str(script), action],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "flashlight.py timed out"}
        output = (proc.stdout + proc.stderr).strip()
        return {
            "ok": proc.returncode == 0,
            "action": action,
            "returncode": proc.returncode,
            "output": output,
        }

    def capture_and_check(self, goal_description: str) -> dict[str, Any]:
        try:
            if self.fake_image:
                frame = camera.load_image(self.cfg.camera, self.fake_image)
            else:
                frame = camera.capture(self.cfg.camera)
        except camera.CameraError as exc:
            return {"ok": False, "error": str(exc)}

        verdict = self.judge.judge(frame, goal_description)
        return {
            "ok": True,
            "image_path": str(frame.path),
            "features": frame.describe(),
            "achieved": verdict.achieved,
            "confidence": verdict.confidence,
            "reason": verdict.reason,
            "verdict_source": verdict.source,
        }

    def dispatch(self, name: str, arguments: str) -> str:
        """Run a tool by name with JSON-encoded arguments, return a JSON string."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}
        if name == "flashlight":
            result = self.flashlight(args.get("action", "toggle"))
        elif name == "capture_and_check":
            result = self.capture_and_check(args.get("goal_description", ""))
        else:
            result = {"ok": False, "error": f"unknown tool '{name}'"}
        return json.dumps(result, ensure_ascii=False)
