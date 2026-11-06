"""Inference backend that judges whether a goal is achieved from a photo.

Sends the captured image directly to a cloud multimodal (vision) LLM over an
OpenAI-compatible API. The model must answer with a small JSON object, which we
parse into a Verdict. OpenCV brightness features are attached as a hint and used
as a deterministic fallback if the API is unavailable or replies unparseably,
so the agent never gets stuck.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .camera import Frame
from .config import VisionConfig

# Brightness (0..255) above which the brightest region is treated as an active
# light source. Used to hint the model and as a standalone fallback.
_BRIGHT_REGION_THRESHOLD = 245.0
_BRIGHT_FRACTION_THRESHOLD = 0.02


@dataclass
class Verdict:
    achieved: bool
    confidence: float
    reason: str
    source: str  # "vision" (model) or "heuristic" (brightness fallback)

    def summary(self) -> str:
        flag = "达成 (ACHIEVED)" if self.achieved else "未达成 (NOT ACHIEVED)"
        return f"[{self.source}] {flag} (confidence={self.confidence:.2f}): {self.reason}"


_SYSTEM_PROMPT = (
    "You are a strict visual verification module. Look at the photo and decide "
    "whether the stated goal is satisfied. Reply with ONLY a compact JSON object: "
    '{"achieved": true|false, "confidence": 0.0-1.0, "reason": "short"}. '
    "No markdown, no extra text."
)


def _heuristic(frame: Frame) -> Verdict:
    """Model-free judgement based purely on brightness features."""
    achieved = (
        frame.max_region_brightness >= _BRIGHT_REGION_THRESHOLD
        and frame.bright_pixel_fraction >= _BRIGHT_FRACTION_THRESHOLD
    )
    conf = 0.6 if achieved else 0.55
    reason = (
        f"brightness heuristic: {frame.describe()} "
        f"(threshold region>={_BRIGHT_REGION_THRESHOLD}, "
        f"fraction>={_BRIGHT_FRACTION_THRESHOLD})"
    )
    return Verdict(achieved=achieved, confidence=conf, reason=reason, source="heuristic")


def _parse_json_verdict(text: str) -> Verdict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if "achieved" not in data:
        return None
    return Verdict(
        achieved=bool(data.get("achieved")),
        confidence=float(data.get("confidence", 0.7)),
        reason=str(data.get("reason", "")).strip() or "(no reason given)",
        source="vision",
    )


def _image_data_uri(path) -> str:
    raw = open(path, "rb").read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class VisionJudge:
    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg
        self._client = OpenAI(api_key=cfg.api_key or "EMPTY", base_url=cfg.base_url)

    def _build_messages(self, frame: Frame, goal: str) -> list[dict[str, Any]]:
        hint = (
            "Auto-detected brightness features (0-255 scale): "
            f"{frame.describe()}. A turned-on flashlight/lamp usually pushes "
            "brightest_region close to 255 with a non-trivial bright_pixel_fraction."
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Goal: {goal}\n{hint}\nLook at the image and decide.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_uri(frame.path)},
                    },
                ],
            },
        ]

    def judge(self, frame: Frame, goal: str) -> Verdict:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": self._build_messages(frame, goal),
            "max_tokens": self.cfg.max_tokens,
        }
        # Kimi recommends not manually setting temperature for some models;
        # a negative value means "leave it to the server default".
        if self.cfg.temperature >= 0:
            kwargs["temperature"] = self.cfg.temperature
        try:
            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            verdict = _parse_json_verdict(content)
            if verdict is not None:
                return verdict
            fallback = _heuristic(frame)
            fallback.reason = f"vision reply unparseable; {fallback.reason}"
            return fallback
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            fallback = _heuristic(frame)
            fallback.reason = f"vision API unavailable ({exc}); {fallback.reason}"
            return fallback
