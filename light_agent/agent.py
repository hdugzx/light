"""The agent loop: DeepSeek plans, tools act, vLLM verifies.

A single "run" takes a natural-language goal, then loops:
  model -> (optional tool calls) -> tool results -> model -> ... -> final answer.

The system prompt nudges the model to always visually verify success with the
camera before declaring the task done.
"""
from __future__ import annotations

from typing import Any

from .config import AgentConfig
from .deepseek import DeepSeekPlanner
from .tools import TOOL_SCHEMAS, Toolbox

SYSTEM_PROMPT = """你是一个运行在终端里的具身智能体（embodied agent）。

你可以使用以下工具：
1. flashlight(action): 通过 ADB 控制手机手电筒，action ∈ {on, off, toggle, status}。
2. capture_and_check(goal_description): 用电脑摄像头拍照，并由视觉推理后端判断某个
   “可视化目标”是否达成（例如“手电筒正在发出明亮的光”）。返回 achieved(true/false)。

工作原则：
- 先理解用户的最终目标，把它拆成动作 + 可视化验证。
- 执行改变物理世界的动作（如开灯）后，必须调用 capture_and_check 用摄像头确认目标是否真的达成，不要仅凭 flashlight 的返回就下结论。
- 如果验证结果为未达成，分析原因并重试（例如再 toggle 一次、检查 adb 状态），最多重试几次。
- 全部完成后，用简洁的中文向用户总结：做了什么、摄像头验证结论、最终是否成功。
- 不要臆造工具没有返回的信息。"""


class Agent:
    def __init__(self, cfg: AgentConfig, fake_image: str | None = None, verbose: bool = True) -> None:
        self.cfg = cfg
        self.planner = DeepSeekPlanner(cfg.deepseek)
        self.tools = Toolbox(cfg, fake_image=fake_image)
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _log(self, *args: Any) -> None:
        if self.verbose:
            print(*args, flush=True)

    def run(self, user_goal: str) -> str:
        self.messages.append({"role": "user", "content": user_goal})

        for step in range(1, self.cfg.max_steps + 1):
            message = self.planner.chat(self.messages, tools=TOOL_SCHEMAS)
            tool_calls = getattr(message, "tool_calls", None)

            # Record the assistant turn (must include tool_calls if present).
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_entry)

            if not tool_calls:
                final = message.content or "(模型没有返回内容)"
                self._log(f"\n[第 {step} 步] 最终回答:")
                return final

            for tc in tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                self._log(f"\n[第 {step} 步] 调用工具 {name}({raw_args})")
                result = self.tools.dispatch(name, raw_args)
                self._log(f"    -> {result}")
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result,
                    }
                )

        return "已到达最大步数上限，任务可能未完全完成。请检查上面的执行日志。"


def run_once(cfg: AgentConfig, goal: str, fake_image: str | None = None) -> str:
    agent = Agent(cfg, fake_image=fake_image)
    return agent.run(goal)
