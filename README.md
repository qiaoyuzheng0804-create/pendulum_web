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
- **前端**：单页 HTML，浅色/深色双主题（日/月按钮切换），SSE 流式进度，KaTeX 公式渲染；首页背景公式水印（KaTeX 排版）、渐变流光大标题、动画步骤条
- **实验获取**：视频获取方式双卡片（上传现成视频 / STM32+OpenMV 实验获取）；释放装置按实验类型自动切换（单摆→电磁铁吸合/释放，磁阻尼摆·扭摆→二维云台＋夹爪方向/开合，磁力牛顿摆→手动释放）；OpenMV 实时预览 + 录制 + 云台

## 技术栈

- 后端：Flask + YOLOv8 (ultralytics)
- 串口：pyserial（STM32 电磁铁 / 二维云台＋夹爪释放控制）
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

`app.py` 启动时按环境变量（`MODEL_W_BEST_{key}`，可选）→ `models/` 目录的顺序查找权重。若要换成自己训练的权重，直接替换对应 `.pt` 文件，或设置对应环境变量指定路径。

### 3. 配置 AI 问答（可选）

AI 问答支持**任意 OpenAI 兼容大模型服务**（OpenAI / DeepSeek / Moonshot / 智谱 / mimo 等）。复制 `.env.example` 为 `.env`，填入你所用服务商的配置：

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

常用服务商组合：

| 服务商 | LLM_BASE_URL | LLM_MODEL 示例 |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |

> **安全提示**：`.env` 文件已在 `.gitignore` 中排除，不会被上传到 GitHub。跨设备迁移时 `.env` 需手工复制，其余内容 `git clone` 即可。

### 4. 启动

```bash
python app.py
# 或 Windows 双击 启动服务器.bat
```

服务运行在 `http://127.0.0.1:5000`，支持局域网访问。

## 实验获取（STM32 释放装置 + OpenMV 拍摄，可选）

「实验获取」模式通过独立 USB 串口控制释放装置，配合 OpenMV 实时预览/录制。**释放装置按实验类型自动切换**：

| 实验类型 | 释放装置 | 串口控制 |
|---|---|---|
| 单摆 (danbai) | 电磁铁（STM32F103C8T6） | 吸合 / 释放 |
| 磁阻尼摆 (cizuni) | 二维云台＋夹爪（STM32F103C8T6） | 方向 U/D/L/R + 开夹 O / 关夹 C |
| 扭摆 (niubai) | 二维云台＋夹爪 | 同上 |
| 磁力牛顿摆 (ciliniudun) | 无（手动释放） | 无需串口 |

使用流程：

1. **烧录/部署固件**：电磁铁见 `firmware/README.md`；云台夹爪见 `firmware/云台夹爪/`（Keil 打开 `Keil工程/USER/GIMBAL.uvprojx` 编译烧录，或直接烧预编译的 `Keil工程/OBJ/GIMBAL.hex`；USART1 115200 8N1，单字节 ASCII 命令）；OpenMV 相机部署 `firmware/openmv/OpenMV端/main.py`（详见 `firmware/openmv/README.md`）
2. **连接硬件**：每个装置一个 USB-TTL，插入后设备管理器确认 COM 口（串口列表已自动过滤蓝牙设备，并内置 `COM9` 手动选项）
3. **网页操作**：数据分析 → 选实验类型 → 视频获取方式选「实验获取」→ 弹出对应释放装置卡片 → 刷新串口 → 选择 → 连接
4. **单摆**：吸合（准备）→ OpenMV 开始拍摄 → 释放小球
5. **磁阻尼摆 / 扭摆**：连接夹爪串口 → 方向按钮调整云台（每次约 2°）→ 开夹准备 → 关夹释放
6. **磁力牛顿摆**：无串口控制，OpenMV 开始拍摄后手动释放摆球
7. **视频**：OpenMV 录制完成后点「停止并上传」自动进入标定流程

> 电脑端只需 `pip install -r requirements.txt`（含 pyserial、Pillow）即可运行网页与串口控制；Keil/STM32 标准库仅在重新编译固件时需要。
> OpenMV 使用 USBDBG V1 协议（921600 波特率）读取帧缓冲，不依赖 OpenMV IDE。

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
│   ├── symbolic_regression.py  # 符号回归拟合
│   └── openmv_manager.py       # OpenMV 摄像头管理器（USBDBG V1 + 录制 + 云台）
├── teaching/
│   ├── theory.json             # 阻尼振动理论
│   ├── guide_danbai.json       # 单摆实验指导
│   ├── guide_cizuni.json       # 磁阻尼摆实验指导
│   ├── guide_niubai.json       # 扭摆实验指导
│   └── guide_ciliniudun.json   # 磁力牛顿摆实验指导
├── firmware/                   # 外设硬件资料总览（详见 firmware/README.md）
│   ├── stm32标准库/             # 电磁铁 Keil 固件工程（与原始工程结构一致）
│   │   ├── project.uvprojx     # Keil 工程（打开→编译→ST-Link 烧录）
│   │   ├── User/               # main.c + stm32f10x_it.c（串口中断释放逻辑）
│   │   ├── System/             # 延时模块
│   │   ├── start/              # 启动文件 + 寄存器定义 + 时钟配置
│   │   └── library/            # ST 标准外设库
│   ├── 电脑端代码/             # 电磁铁 controller.py 命令行测试工具 + 接线说明
│   ├── openmv/                 # OpenMV：相机端脚本 + USBDBG 电脑端工具 + USB 驱动
│   └── 云台夹爪/               # 二维云台＋夹爪：Keil 工程 + 预编译 HEX + 接线/烧录文档 + PC 工具
├── templates/
│   └── index.html              # 单页前端
└── uploads/                    # 上传视频临时存储（运行时创建，git 忽略）
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| GET | `/api/calibration_spec/<motion_type>` | 获取实验类型的标定点规格 |
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
| GET | `/api/gripper/ports` | 列出夹爪可用串口（过滤蓝牙，含 COM9） |
| POST | `/api/gripper/connect` | 连接夹爪串口（115200 8N1） |
| POST | `/api/gripper/command` | 发送命令 `U/D/L/R`(方向) / `O/C`(开合) |
| POST | `/api/gripper/disconnect` | 断开夹爪串口 |
| GET | `/api/openmv/ports` | 列出串口（标注 OpenMV 设备） |
| POST | `/api/openmv/connect` | 连接 OpenMV 摄像头（USBDBG V1，921600 波特率） |
| POST | `/api/openmv/disconnect` | 断开 OpenMV 摄像头 |
| GET | `/api/openmv/frame` | 单帧 JPEG（前端 canvas 轮询预览用） |
| GET | `/api/openmv/stream` | MJPEG 实时预览流 |
| POST | `/api/openmv/record/start` | 开始录制帧 |
| POST | `/api/openmv/record/stop` | 停止录制，保存 MP4 并自动创建处理 session |
| POST | `/api/openmv/gimbal` | 云台控制（U/D/L/R/C） |
| GET | `/api/openmv/status` | 查询 OpenMV 连接与录制状态 |

## 处理速度说明

处理耗时随视频时长线性增长，主要开销在逐帧解码 + YOLO 推理（RTX 4060 实测约 50 帧/秒）：

- **单摆**：默认输出 400 个周期，会持续处理到检测满 800 次零交叉；视频较短则自动用完全部帧
- **磁力牛顿摆**：跟踪 + 标注视频两遍处理，最慢；不需要标注视频时取消勾选「输出标注视频」可大幅提速
- **进度到 100% 后**：还有运动方程拟合阶段（约 10~60 秒，视数据量），期间界面无进度属正常现象
- 处理结果有缓存机制（视频 MD5 + 参数 hash），相同请求可秒回

## 注意事项

- YOLO 模型按需懒加载：启动秒开，首次用到某实验类型时才加载对应权重，之后常驻内存
- 叉掉网页后约 2~3 秒服务器自动停止（页面卸载发告别信标 + 心跳看门狗；退出前电磁铁自动断电保护），多标签页时最后一个关闭才停；`停止服务器.bat` 保留为手动兜底
- 桌面快捷方式「摆动实验分析平台」：双击启动服务并自动打开网页（等端口就绪后打开，不会报错）
- 同一时刻仅运行一个 YOLO 处理任务（`_process_lock`）
- Session 30 分钟自动清理

## 安全说明

- 文件上传：仅接受 `.mp4/.avi/.mov/.mkv/.webm/.m4v` 视频格式，单文件上限 1 GB（超出返回 413）
- API 密钥：存储在 `.env` 文件中，已通过 `.gitignore` 排除，不会上传到 GitHub
- 生产部署：本项目默认用 Flask 内置服务器（`debug=False`），仅适合局域网/教学使用；若部署到公网，建议用 Gunicorn + Nginx 反向代理，并启用 HTTPS