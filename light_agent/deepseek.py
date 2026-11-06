"""Planner brain backed by the DeepSeek chat API (OpenAI-compatible).

Uses native OpenAI function/tool calling so the model decides which tool to
invoke at each step. The heavy lifting of executing tools lives in tools.py;
this module only speaks to the API.
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import DeepSeekConfig


class DeepSeekPlanner:
    def __init__(self, cfg: DeepSeekConfig) -> None:
        if not cfg.api_key:
            raise RuntimeError(
                "DeepSeek API key is missing. Set DEEPSEEK_API_KEY (or OPENAI_API_KEY)."
            )
        self.cfg = cfg
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """One round-trip to the model. Returns the raw message object."""
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message
