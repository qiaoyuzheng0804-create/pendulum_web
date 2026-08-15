# Pendulum Motion Analysis Web App

Flask + YOLOv8 实验视频分析平台。上传摆的实验视频，自动追踪摆球轨迹、拟合运动方程、输出阻尼参数和周期。内含阻尼实验教学模块（理论、实验指导、交互模拟、AI 问答）。

## 功能特性

- **四种摆的实验分析**
  - 单摆 (danbai)：3 点标定 + Savitzky-Golay 滤波 + 零交叉检测 + 周期拟合
  - 磁阻尼摆 (cizuni)：高斯平滑 + 角度计算，支持欠阻尼 / 临界阻尼 / 过阻尼
  - 扭摆 (niubai)：卡尔曼滤波（RTS 平滑）+ 角度提取
  - 磁力牛顿摆 (ciliniudun)：YOLOv8 + ByteTrack 多目标跟踪，2~5 个摆球同时提取角度-时间序列，轨迹绑定 + 水平几何约束修正，输出各摆 CSV / 汇总 CSV / 标注视频 / 时空数据图
- **阻尼实验教学模块**
  - 理论知识：阻尼分类、微分方程、参数物理意义
  - 实验指导：器材清单、拍摄要点、标定说明、FAQ
  - 交互模拟：Canvas 实时绘制阻尼振荡波形
  - AI 问答：LLM 接入，Markdown + LaTeX 公式渲染
- **前端**：单页 HTML，6 种主题，SSE 流式进度，KaTeX 公式渲染
- **实验获取**：视频获取方式双卡片（上传现成视频 / STM32+OpenMV 实验获取）；实验获取模式通过串口控制 STM32 电磁铁释放小球（'1' 吸合 / '0' 释放），OpenMV 拍摄模块预留位置

## 技术栈

- 后端：Flask + YOLOv8 (ultralytics)
- 串口：pyserial（STM32 电磁铁释放控制）
- 视频 / 数值：OpenCV、NumPy、SciPy、pandas、matplotlib
- 滤波：Savitzky-Golay、高斯、卡尔曼 (filterpy)
- 拟合：符号回归 (纯 SciPy) 参数拟合
- 前端：原生 HTML / JS + marked.js + KaTeX
- AI：OpenAI 兼容 API

## 快速启动

### 1. 环境准备

```bash
conda create -n yolov8 python=3.8 -y
conda activate yolov8
pip install -r requirements.txt
```

### 2. 配置模型

四种摆的 YOLOv8 权重已随仓库提供，位于 `models/` 目录：

| 实验 | 模型文件 |
|------|---------|
| 单摆 (danbai) | `models/danbai_best.pt` |
| 磁阻尼摆 (cizuni) | `models/cizuni_best.pt` |
| 扭摆 (niubai) | `models/niubai_best.pt` |
| 磁力牛顿摆 (ciliniudun) | `models/ciliniudun_best.pt` |

`app.py` 启动时会优先使用本地训练路径，找不到则回退到 `models/` 目录。若要换成自己训练的权重，直接替换对应 `.pt` 文件即可。

### 3. 配置 AI 问答（可选）

编辑项目根目录的 `.env` 文件，填入你的 API Key：

```env
MIMO_API_KEY=your_api_key_here
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2.5
```

> **安全提示**：`.env` 文件已在 `.gitignore` 中排除，不会被上传到 GitHub。

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
├── .env                    # API 配置文件（不会上传到 GitHub）
├── requirements.txt        # Python 依赖清单
├── 启动服务器.bat           # Windows 启动脚本（自动读取 .env）
├── 停止服务器.bat           # 停止服务
├── models/                 # YOLOv8 权重（四种摆各一个 .pt）
├── processors/
│   ├── __init__.py
│   ├── danbai_processor.py     # 单摆处理
│   ├── cizuni_processor.py     # 磁阻尼摆处理
│   ├── niubai_processor.py     # 扭摆处理
│   ├── ciliniudun_processor.py # 磁力牛顿摆处理（YOLOv8 + ByteTrack 多目标跟踪）
│   └── symbolic_regression.py  # 符号回归拟合
├── teaching/
│   ├── theory.json             # 阻尼振动理论
│   ├── guide_danbai.json       # 单摆实验指导
│   ├── guide_cizuni.json       # 磁阻尼摆实验指导
│   ├── guide_niubai.json       # 扭摆实验指导
│   └── guide_ciliniudun.json   # 磁力牛顿摆实验指导
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
| GET | `/api/serial/ports` | 列出可用串口 |
| POST | `/api/serial/connect` | 连接串口（115200 8N1），连接后先发送 `0` 建立断电状态 |
| POST | `/api/serial/command` | 发送命令 `0`(释放) / `1`(吸合)，返回发送时刻时间戳 |
| POST | `/api/serial/disconnect` | 断开串口（若通电先断电保护） |

## 注意事项

- 服务器启动时预加载 YOLO 模型，首次启动较慢
- 同一时刻仅运行一个 YOLO 处理任务（`_process_lock`）
- Session 30 分钟自动清理
- 处理结果有缓存（视频 MD5 + 参数 hash）

## 安全说明

- 文件上传：仅接受 `.mp4/.avi/.mov/.mkv/.webm/.m4v` 视频格式，单文件上限 1 GB（超出返回 413）
- API 密钥：存储在 `.env` 文件中，已通过 `.gitignore` 排除，不会上传到 GitHub
- 生产部署：本项目默认用 Flask 内置服务器（`debug=False`），仅适合局域网/教学使用；若部署到公网，建议用 Gunicorn + Nginx 反向代理，并启用 HTTPS