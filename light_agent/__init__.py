"""light_agent: a terminal agent.

- Planner brain  -> DeepSeek API (OpenAI-compatible chat).
- Vision / goal-check inference -> cloud multimodal LLM API (OpenAI-compatible).
- Tools -> phone flashlight (flashlight.py over ADB) + camera capture.
"""

__all__ = ["config", "deepseek", "vision", "camera", "tools", "agent"]
