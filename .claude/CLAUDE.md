# Pendulum Motion Analysis Web App

Flask + YOLOv8 实验视频分析平台。上传摆的实验视频，自动追踪摆球轨迹、拟合运动方程、输出阻尼参数和周期。含阻尼实验教学模块（理论、实验指导、交互模拟、AI 问答）。

## 快速启动

```bash
# 激活 conda 环境
conda activate yolov8

# 安装依赖（如未安装）
pip install flask opencv-python numpy pandas scipy matplotlib ultralytics filterpy scikit-learn openai

# 启动
cd pendulum_web
python app.py
# 或双击 启动服务器.bat（自动激活 yolov8 环境 + MIMO API 环境变量）
```

服务运行在 `http://127.0.0.1:5000`，支持局域网访问。

## 项目结构

```
pendulum_web/
├── app.py                          # Flask 主入口，路由 + SSE 流式处理 + AI 问答
├── 启动服务器.bat                   # Windows 启动脚本（含 MIMO API 环境变量）
├── 停止服务器.bat                   # 停止服务
├── processors/
│   ├── __init__.py                 # 导出 process_danbai / process_cizuni / process_niubai
│   ├── danbai_processor.py         # 单摆处理
│   ├── cizuni_processor.py         # 磁阻尼摆处理
│   ├── niubai_processor.py         # 扭摆处理
│   └── symbolic_regression.py      # 符号回归拟合（纯 scipy）
├── teaching/
│   ├── theory.json                 # 阻尼振动理论知识
│   ├── guide_danbai.json           # 单摆实验指导
│   ├── guide_cizuni.json           # 磁阻尼摆实验指导
│   └── guide_niubai.json           # 扭摆实验指导
├── templates/
│   └── index.html                  # 单页前端（数据分析 + 教学双模式）
├── .gitignore                      # 忽略 uploads/ __pycache__/ *.pt 等
└── uploads/                        # 上传视频临时存储（运行时自动创建，git 忽略）
```

## 前端架构

单页 HTML，6 种主题（dark/ocean/sunset/forest/amethyst/light），顶部模式切换：**🔬 数据分析** / **📚 阻尼实验教学**。

### 数据分析模式

原有功能：选择实验类型 → 上传视频 → 标定坐标系 → 运行处理 → 查看结果。通过 SSE（`/api/process_stream`）接收处理进度。

### 教学模式（4 个 Tab）

| Tab | 内容 | 技术 |
|-----|------|------|
| 理论知识 | 阻尼分类、微分方程、参数物理意义、三种实验对比 | KaTeX 公式渲染 |
| 实验指导 | 器材清单、拍摄要点、标定说明、参数设置、FAQ | JSON 动态加载 |
| 交互模拟 | Canvas 实时绘制阻尼振荡波形，可调 β/ω₀/θ₀ | 固定比例尺 ±25° |
| AI 问答 | 接入 mimo v2.5，Markdown + LaTeX 公式渲染 | SSE 流式 + marked.js + KaTeX |

### 前端公式渲染系统

共享函数（在 `index.html` 中）：

- `unicodeToLatex(text)` — Unicode 数学符号（√ θ ω β ζ ₀₁₂₃ 等）转 LaTeX 命令
- `wrapBareLatex(text)` — 裸 LaTeX 命令自动包裹 `$...$`（调用 unicodeToLatex）
- `renderKaTeXinHTML(html)` — HTML 字符串中的 `$...$` / `$$...$$` 渲染为 KaTeX
- `convertNoteToLatex(text)` — 理论笔记专用，转 LaTeX 并包裹
- `renderChatMarkdown(text)` — AI 聊天完整管线：清理 → 解码实体 → 包裹 LaTeX → marked.js → 解码 → KaTeX

AI 聊天中：先解码 `&lt;` `&gt;`，再 Unicode→LaTeX，再包裹裸命令，再 marked.js 解析 Markdown，marked.js 会重新编码 `<` `>` 所以再解码一次，最后 KaTeX 渲染。

### AI 问答

使用 `openai` Python 包调用 mimo v2.5（兼容 OpenAI API 格式）。

环境变量（已配置在 `启动服务器.bat`）：
- `MIMO_API_KEY` — API 密钥
- `MIMO_BASE_URL` — `https://token-plan-cn.xiaomimimo.com/v1`
- `MIMO_MODEL` — `mimo-v2.5`

System prompt 要求 AI 用 Markdown 格式回答，公式用 `$...$` / `$$...$$` 包裹。

## 三种摆的处理流程

1. **单摆** (`danbai`) — 3 点标定（悬挂点 + 竖直参考），Savitzky-Golay 滤波 → 零交叉检测 → 等角度重采样 → 周期拟合
2. **磁阻尼摆** (`cizuni`) — 3 点标定（2 竖直参考 + 转轴），高斯平滑 → 角度计算 → 支持欠阻尼/临界阻尼/过阻尼
3. **扭摆** (`niubai`) — 5 点标定（原点 + XY 轴 + 标尺参考），卡尔曼滤波（RTS 平滑）→ 角度提取

## 交互模拟参数

- 阻尼系数 β：0 ~ 5 s⁻¹
- 固有频率 ω₀：0.5 ~ 20 rad/s
- 初始角度 θ₀：0 ~ 20°
- 画布固定比例尺：±25°
- 三种阻尼类型自动切换（ζ < 1 欠阻尼 / ζ ≈ 1 临界阻尼 / ζ > 1 过阻尼）

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

## 模型路径

YOLOv8 模型按优先级查找：本地路径 → 项目 `models/` 目录。

本地路径（开发环境）：
- `C:\Users\MECHREV\Desktop\yolov8\runs\detect\small_ball_yolov8_safe2\weights\best.pt`（单摆）
- `C:\Users\MECHREV\Desktop\yolov8\runs\detect\cizuni\weights\best.pt`（磁阻尼摆）
- `C:\Users\MECHREV\Desktop\yolov8\runs\detect\niubai\weights\best.pt`（扭摆）

## 处理器说明

### danbai_processor.py — 单摆

1. YOLO 逐帧检测（`conf=0.25, iou=0.45, imgsz=1280`），最近邻跟踪避免目标跳变
2. 角度计算 → 在线平滑 → 提前停止 → Savitzky-Golay 滤波
3. 零偏校正 → 零交叉检测 → 等角度重采样 → 包络提取
4. 输出 CSV + PNG

### cizuni_processor.py — 磁阻尼摆

1. YOLO 逐帧检测，`sample_step` 降采样（保证 ≥180 数据点）
2. 角度计算（支持 flip）→ 高斯平滑（σ=0.8）
3. 输出双面板图，支持中文字体回退
4. 内部每次重新创建 YOLO 模型（避免 CUDA 上下文冲突）

### niubai_processor.py — 扭摆

1. 像素坐标 → 物理坐标（米）→ 角度计算（atan2 + 相位展开）
2. 卡尔曼滤波（前向 + RTS 后向平滑，`q=1.0, r=0.1`）
3. 零基线 → 周期提取 → 提前停止
4. 输出 CSV + 双面板图

### symbolic_regression.py — 参数拟合

纯 scipy。三种阻尼拟合 + ODE 验证 + 物理项库拟合。

## 注意事项

- 服务器启动时预加载 YOLO 模型，首次启动较慢
- `_process_lock` 确保同一时刻只有一个 YOLO 处理任务
- Session 30 分钟自动清理，超过 1 GB 被 Flask 拒绝
- 处理结果有缓存（视频 MD5 + 参数 hash）
- `KMP_DUPLICATE_LIB_OK=TRUE` 解决 PyTorch 与 OpenCV 冲突
- 扭摆依赖 `filterpy`，AI 问答依赖 `openai`
- 新增实验：在 `teaching/` 添加 JSON 文件，在 `app.py` 的 `allowed` 集合注册
- 前端输出目录由用户指定，需确保可写
- 所有处理器 CSV 格式：`time_s, angle_deg`（前两列）

## Git 自动提交与推送规则

本项目由 agent 维护，**每完成一个有意义的最小改动单元后，必须自动 commit 并 push 到 GitHub**，保持远程与本地同步。

### 执行命令
```bash
git add -A
git commit -m "简洁描述改动内容"
git push
```

### commit message 规范
- 用中文写明改了什么
- 不超过 30 个字
- 示例："补充依赖与模型权重"、"修复上传类型校验"

### 提交时机
- 完成一个完整的功能/修复/文档改动后提交一次，**不要每保存一次就提交**
- 攒够一个「有意义的最小改动单元」（一个功能、一个 bug 修复、一段文档）再提交
- commit 后紧跟 push，保持 GitHub 同步

### 禁止提交的内容（提交前务必检查）
- API key、token、密码等任何敏感凭据
- `uploads/`、`__pycache__/`、`*.pyc`、`server.pid`、`.env`（已在 .gitignore 忽略）
- 本地调试用的临时文件

### push 凭据
- 远程地址：`https://github.com/qiaoyuzheng0804-create/pendulum_web.git`
- 凭据通过 git credential helper 提供，**严禁把 token 写进任何会提交到 git 的文件**
