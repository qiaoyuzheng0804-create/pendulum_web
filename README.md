# Pendulum Motion Analysis Web App

Flask + YOLOv8 实验视频分析平台。上传摆的实验视频，自动追踪摆球轨迹、拟合运动方程、输出阻尼参数和周期。内置阻尼实验教学模块（理论、实验指导、交互模拟、AI 问答）。

## 功能特性

- **三种摆的实验分析**
  - 单摆（danbai）：3 点标定 + Savitzky-Golay 滤波 + 零交叉检测 + 周期拟合
  - 磁阻尼摆（cizuni）：高斯平滑 + 角度计算，支持欠阻尼 / 临界阻尼 / 过阻尼
  - 扭摆（niubai）：卡尔曼滤波（RTS 平滑）+ 角度提取
- **阻尼实验教学模块**
  - 理论知识：阻尼分类、微分方程、参数物理意义
  - 实验指导：器材清单、拍摄要点、标定说明、FAQ
  - 交互模拟：Canvas 实时绘制阻尼振荡波形
  - AI 问答：LLM 接入，Markdown + LaTeX 公式渲染
- **前端**：单页 HTML，6 种主题，SSE 流式进度，KaTeX 公式渲染

## 技术栈

- 后端：Flask + YOLOv8（ultralytics）
- 视频 / 数值：OpenCV、NumPy、SciPy、pandas、Matplotlib
- 滤波：Savitzky-Golay、高斯、卡尔曼（filterpy）
- 拟合：符号回归（纯 SciPy）+ 参数拟合
- 前端：原生 HTML / JS + marked.js + KaTeX
- AI：OpenAI 兼容 API

## 快速启动

### 1. 环境准备

```bash
conda create -n yolov8 python=3.8 -y
conda activate yolov8
pip install flask opencv-python numpy pandas scipy matplotlib ultralytics filterpy scikit-learn openai
```

### 2. 配置模型

将 YOLOv8 权重文件（`.pt`）放入项目 `models/` 目录，或在 `app.py` 中配置本地模型路径。三种摆各需要一个检测模型（权重文件默认不随仓库分发）。

### 3. 配置 AI 问答（可选）

设置环境变量：

- `MIMO_API_KEY` — API 密钥
- `MIMO_BASE_URL` — 默认 `https://token-plan-cn.xiaomimimo.com/v1`
- `MIMO_MODEL` — 默认 `mimo-v2.5`

Windows 下可在 `启动服务器.bat` 的 `set MIMO_API_KEY=` 后填入你的 key。

### 4. 启动

```bash
python app.py
# 或 Windows 双击 启动服务器.bat
```

服务运行在 `http://127.0.0.1:5000`，支持局域网访问。

## 项目结构

```
pendulum_web/
├── app.py                  # Flask 主入口：路由 + SSE 流式处理 + AI 问答
├── 启动服务器.bat           # Windows 启动脚本
├── 停止服务器.bat           # 停止服务
├── processors/
│   ├── __init__.py
│   ├── danbai_processor.py     # 单摆处理
│   ├── cizuni_processor.py     # 磁阻尼摆处理
│   ├── niubai_processor.py     # 扭摆处理
│   └── symbolic_regression.py  # 符号回归拟合
├── teaching/
│   ├── theory.json             # 阻尼振动理论
│   ├── guide_danbai.json       # 单摆实验指导
│   ├── guide_cizuni.json       # 磁阻尼摆实验指导
│   └── guide_niubai.json       # 扭摆实验指导
├── templates/
│   └── index.html              # 单页前端
└── uploads/                    # 上传视频临时存储（运行时创建，git 忽略）
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| POST | `/api/upload_video` | 上传视频，返回 session_id + 首帧 base64 |
| POST | `/api/process` | 同步处理（阻塞） |
| POST | `/api/process_stream` | SSE 流式处理（推荐） |
| GET | `/api/progress/<session_id>` | 查询处理进度 |
| POST | `/api/cleanup` | 清理 session 文件 |
| GET | `/api/teaching_content/<topic>` | 获取教学内容 JSON |
| POST | `/api/ai_chat` | AI 知识问答（SSE 流式） |

## 注意事项

- 服务器启动时预加载 YOLO 模型，首次启动较慢
- 同一时刻仅运行一个 YOLO 处理任务（`_process_lock`）
- Session 30 分钟自动清理，上传超过 4 GB 被拒绝
- 处理结果有缓存（视频 MD5 + 参数 hash）
