"""Terminal entry point.

Usage:
    python -m light_agent                      # interactive REPL
    python -m light_agent "把手电筒打开并确认"   # one-shot goal
    python -m light_agent --fake-image test.jpg "确认手电筒是否亮着"
"""
from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .config import load_config


def _print_banner(cfg) -> None:
    print("=" * 64)
    print(" light_agent  ·  DeepSeek 规划 + 多模态 API 判图 + 手电筒/摄像头工具")
    print("=" * 64)
    print(f"  规划模型 (DeepSeek): {cfg.deepseek.model} @ {cfg.deepseek.base_url}")
    print(f"  视觉判图 (Vision)  : {cfg.vision.model} @ {cfg.vision.base_url}")
    print(f"  手电筒脚本         : {cfg.flashlight.script_path}")
    print(f"  摄像头索引         : {cfg.camera.index}")
    print("-" * 64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Terminal embodied agent.")
    parser.add_argument("goal", nargs="*", help="要完成的目标（留空进入交互模式）。")
    parser.add_argument(
        "--fake-image",
        metavar="PATH",
        default=None,
        help="用一张已有图片代替摄像头拍照（便于无摄像头时测试）。",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印中间执行日志。")
    args = parser.parse_args(argv)

    cfg = load_config()
    _print_banner(cfg)

    try:
        agent = Agent(cfg, fake_image=args.fake_image, verbose=not args.quiet)
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 2

    goal = " ".join(args.goal).strip()
    if goal:
        answer = agent.run(goal)
        print("\n" + "=" * 64)
        print(answer)
        return 0

    print("进入交互模式，输入目标后回车。输入 exit / quit 退出。")
    while True:
        try:
            line = input("\n目标> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            return 0
        answer = agent.run(line)
        print("\n" + "=" * 64)
        print(answer)


if __name__ == "__main__":
    raise SystemExit(main())
