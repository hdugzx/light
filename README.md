# light_agent

一个运行在终端里的具身智能体（embodied agent），全部走云端大模型 API：

- **规划大脑 / 推理**：DeepSeek API（OpenAI 兼容，文本），负责理解目标、决定调用哪个工具、重试与总结。
- **图片判定**：云端**多模态大模型 API**（OpenAI 兼容），把摄像头拍到的照片直接发给视觉模型，判断“目标是否达成”。
- **工具**：
  - `flashlight`：通过 `flashlight.py` + ADB 控制手机手电筒（`on/off/toggle/status`）。
  - `capture_and_check`：调用电脑摄像头拍照，交给视觉大模型判定目标是否达成。
- **入口**：终端（一次性命令 or 交互 REPL）。

> 说明：DeepSeek 官方 API 只有文本模型，没有视觉接口。因此“图片判定”走一个**可配置的多模态 API**（`VISION_*`），指向任意支持视觉模型的 OpenAI 兼容服务（gpt-4o、qwen-vl，或同时暴露 deepseek + 视觉模型的聚合网关）。视觉 API 不可用时会自动回退到亮度启发式（`source=heuristic`），不会卡住。

## 工作流

```
用户目标 ─▶ DeepSeek 规划 ─▶ flashlight(on) ─▶ capture_and_check ─▶ 多模态 API 判图
                    ▲                                                    │
                    └───────────────── 未达成则重试 ◀────────────────────┘
```

执行完“开灯”这种物理动作后，**必须**用摄像头拍照 + 视觉 API 验证才算数。

## 目录

```
light_agent/
  config.py       # 所有配置（读环境变量 / .env）
  deepseek.py     # DeepSeek 规划器（文本）
  vision.py       # 多模态 API 判图 + 亮度启发式兜底
  camera.py       # OpenCV 摄像头拍照 + 亮度特征
  tools.py        # flashlight / capture_and_check 工具及其 schema
  agent.py        # ReAct 主循环
  __main__.py     # 终端入口
flashlight.py     # 已有：ADB 手电筒控制
run.sh            # 运行 agent
.env.example      # 配置样例
```

## 使用步骤

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env：
#   DEEPSEEK_API_KEY  -> DeepSeek key（未填会回退到 OPENAI_API_KEY）
#   VISION_BASE_URL / VISION_MODEL / VISION_API_KEY -> 视觉多模态 API
#   （VISION_API_KEY 未填时回退到 DEEPSEEK_API_KEY / OPENAI_API_KEY）
```

### 2. 连接手机（手电筒工具需要）

```bash
adb devices   # 确认出现已授权设备
```

### 3. 运行 agent

```bash
# 一次性目标
./run.sh "打开手电筒，并用摄像头确认它确实亮了"

# 交互模式
./run.sh

# 没有摄像头时，用一张图片测试判定链路
./run.sh --fake-image /path/to/photo.jpg "确认画面里的手电筒是否亮着"
```

依赖：`openai`、`opencv-python`、`numpy`、`pillow`。
`run.sh` 默认用自带这些包的 `/mnt/orisd/shuaiman/.venv-vllm/bin/python`；也可 `pip install -r requirements.txt` 后用系统 Python：`python -m light_agent ...`。

## 配置项

见 `.env.example`，全部可用环境变量覆盖：DeepSeek（key/base_url/model）、Vision（base_url/api_key/model）、摄像头（index/分辨率）、手电筒脚本路径、最大步数等。

## 说明

- 拍摄的照片保存在 `CAPTURE_DIR`（默认 `captures/`）。
- 判定返回 `achieved / confidence / reason / source`，`source` 为 `vision`（模型判定）或 `heuristic`（亮度兜底）。
- 亮度阈值在 `vision.py` 顶部（`_BRIGHT_REGION_THRESHOLD` 等），可按你的环境微调。
