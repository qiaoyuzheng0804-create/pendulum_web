# Pendulum Motion Analysis Web App

Flask + YOLOv8 实验视频分析平台。上传摆的实验视频，自动追踪摆球轨迹、拟合运动方程、输出阻尼参数和周期。含阻尼实验教学模块（理论、实验指导、交互模拟、AI 问答）。

## 快速启动

```bash
# 激活 conda 环境
conda activate yolov8

# 安装依赖（如未安装，自动包含 flask/cv2/ultralytics/filterpy/pyserial 等全部依赖）
pip install -r requirements.txt

# 配置 API Key（可选，用于 AI 问答）
# 复制 .env.example 为 .env，填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL

# 启动
cd pendulum_web
python app.py
# 或双击 启动服务器.bat（自动读取 .env 配置）
```

服务运行在 `http://127.0.0.1:5000`，支持局域网访问。

## 项目结构

```
pendulum_web/
├── app.py                          # Flask 主入口，路由 + SSE 流式处理 + AI 问答
├── .env                            # API 配置文件（不会上传到 GitHub）
├── 启动服务器.bat                   # Windows 启动脚本（自动读取 .env）
├── 停止服务器.bat                   # 停止服务
├── models/                         # YOLOv8 权重（四种摆各一个 .pt，已纳入 git 跟踪）
├── processors/
│   ├── __init__.py                 # 导出四个处理器
│   ├── danbai_processor.py         # 单摆处理
│   ├── cizuni_processor.py         # 磁阻尼摆处理
│   ├── niubai_processor.py         # 扭摆处理
│   ├── ciliniudun_processor.py     # 磁力牛顿摆处理（YOLOv8 + ByteTrack 多目标跟踪）
│   ├── symbolic_regression.py      # 符号回归拟合（纯 scipy）
│   └── openmv_manager.py           # OpenMV 摄像头管理器（USBDBG V1 帧采集 + 录制 + 云台控制）
├── teaching/
│   ├── theory.json                 # 阻尼振动理论知识
│   ├── guide_danbai.json           # 单摆实验指导
│   ├── guide_cizuni.json           # 磁阻尼摆实验指导
│   ├── guide_niubai.json           # 扭摆实验指导
│   └── guide_ciliniudun.json       # 磁力牛顿摆实验指导
├── firmware/                       # 外设硬件资料总览（详见 firmware/README.md）
│   ├── stm32标准库/ + 电脑端代码/   # 电磁铁固件 Keil 工程 + 测试工具
│   ├── openmv/                     # OpenMV：相机端脚本 + USBDBG 电脑端工具 + USB 驱动（openmv_manager.py 的参考实现）
│   └── 云台夹爪/                   # 二维云台＋夹爪：Keil 工程 + 预编译 GIMBAL.hex + 接线/烧录文档
├── templates/
│   └── index.html                  # 单页前端（数据分析 + 教学双模式）
├── .gitignore                      # 忽略 uploads/ __pycache__/ *.pt 等
└── uploads/                        # 上传视频临时存储（运行时自动创建，git 忽略）
```

## 前端架构

单页 HTML：浅色/深色双主题（`data-theme="light|dark"`，右上角日/月圆形按钮切换，localStorage 记忆，默认深色），顶部模式切换：**数据分析** / **阻尼实验教学**。

### 数据分析模式

原有功能：选择实验类型 → 上传视频 → 标定坐标系 → 运行处理 → 查看结果。通过 SSE（`/api/process_stream`）接收处理进度。

### 教学模式（4 个 Tab）

| Tab | 内容 | 技术 |
|-----|------|------|
| 理论知识 | 阻尼分类、微分方程、参数物理意义、三种实验对比 | KaTeX 公式渲染 |
| 实验指导 | 器材清单、拍摄要点、标定说明、参数设置、FAQ | JSON 动态加载 |
| 交互模拟 | Canvas 实时绘制阻尼振荡波形，可调 β/ω₀/θ₀ | 固定比例尺 ±25° |
| AI 问答 | 接入 OpenAI 兼容大模型（LLM_* 可配置），Markdown + LaTeX 公式渲染 | SSE 流式（~150ms 节流渲染 + 自动滚动 + 结束终绘）+ marked.js + KaTeX |

### 前端公式渲染系统

共享函数（在 `index.html` 中）：
- `unicodeToLatex(text)` — Unicode 数学符号（≈ π ω β θ ∝ 等）转 LaTeX 命令
- `wrapBareLatex(text)` — 裸 LaTeX 命令自动包裹 `$...$`（调用 unicodeToLatex）
- `renderKaTeXinHTML(html)` — HTML 字符串中的 `$...$` / `$$...$$` 渲染为 KaTeX
- `convertNoteToLatex(text)` — 理论笔记专用，转 LaTeX 并包裹
- `renderChatMarkdown(text)` — AI 聊天完整管线：清理 → 解码实体 → 包裹 LaTeX → marked.js → 解码 → KaTeX

AI 聊天渲染（`renderChatMarkdown`）实际管线：`convertBacktickMath`（反引号包裹且含 LaTeX 命令/希腊字母/上下标的数学式自动转 `$...$`，普通代码不转）→ marked.js 解析 Markdown → `renderKaTeXinHTML`（先剥真实 HTML 标签，再用 `decodeHtmlEntities` 解码 `&lt;` `&gt;` 等实体，最后 KaTeX 渲染——否则实体被当未知命令渲染成红色报错）。

流式策略：SSE 分词期间以 ~150ms 节流做 Markdown+KaTeX 渲染（边生成边呈现排版、自动跟随滚动），结束后终绘一次；渲染频率有上限，长回答不卡。

### 首页视觉组件

- **背景公式水印**：9 条 `.formula-text` 浮动水印，LaTeX 写在 `data-tex` 属性中，`DOMContentLoaded` 后用 KaTeX 渲染（KaTeX 为 defer 加载，此时必已就绪）；KaTeX 加载失败自动降级为纯文本 + 纯 CSS 根号补线（`.fsqrt/.frad`，用 border-top 画 vinculum）。**改水印公式要同时改 `data-tex` 和降级文本两处**。
- **步骤条**（1 选择类型 → 2 标定 → 3 运行 → 4 完成）：`.step-dot` 渐变圆点 + `stepPop` 弹性入场 + `stepHalo` 光环脉冲；连接线 `.step-line` 用 `lineGrow/lineShift` 做流光；当前步骤标签为胶囊徽章。JS 钩子不变：`sn1-4/sd1-4/sl1-3` 元素 + `done/current` 类，由 `updateStepBar(n)` 驱动。
- **大标题**：五段回文渐变（a2→a1→a3→a1→a2）+ `titleShimmer` 9s 流光 + `drop-shadow(var(--fr))` 柔光；首尾同色保证 background-position 循环无缝。
- `@keyframes dotPulse` 被 AI 聊天打字点共用，调整步骤条动画时勿删。

### AI 问答

使用 `openai` Python 包调用任意 OpenAI 兼容大模型服务（OpenAI / DeepSeek / Moonshot / 智谱 / mimo 等）。

配置方式（存储在 `.env` 文件中，不会上传到 GitHub，模板见 `.env.example`）：
- `LLM_API_KEY` — API 密钥（必需）
- `LLM_BASE_URL` — 服务商地址，默认 `https://api.openai.com/v1`
- `LLM_MODEL` — 模型名，默认 `gpt-4o-mini`
- 旧版 `MIMO_*` 变量名仍被识别（向后兼容）

System prompt 要求 AI 用 Markdown 格式回答，公式用 `$...$` / `$$...$$` 包裹。

## 四种摆的处理流程

1. **单摆** (danbai_processor.py)
   - YOLO 逐帧检测（`conf=0.25, iou=0.45, imgsz=1280`），最近邻跟踪避免目标跳变
   - 角度计算 → 在线平滑 → 提前停止 → Savitzky-Golay 滤波
   - 零偏校正 → 零交叉检测 → 等角度重采样 → 包络提取
   - 输出 CSV + PNG

2. **磁阻尼摆** (cizuni_processor.py)
   - YOLO 逐帧检测，`sample_step` 降采样（保证 ≥80 数据点）
   - 角度计算（支持 flip）→ 高斯平滑（σ=0.8）
   - 输出双面板图，支持中文字体回退
   - 内部每次重新创建 YOLO 模型（避免 CUDA 上下文冲突）

3. **扭摆** (niubai_processor.py)
   - 像素坐标 → 物理坐标（米）→ 角度计算（atan2 + 相位展开）
   - 卡尔曼滤波（前向 + RTS 后向平滑，`q=1.0, r=0.1`）
   - 零基线 → 周期提取 → 提前停止
   - 输出 CSV + 双面板图

4. **磁力牛顿摆** (ciliniudun_processor.py)
   - YOLOv8 检测 + ByteTrack 多目标跟踪（`conf=0.18, iou=0.5, imgsz=1536`，bytetrack.yaml）
   - 轨迹绑定：每条轨迹的质心离哪个悬挂点最近就归哪个摆（中断轨迹自动合并）
   - 水平几何约束修正：匈牙利算法全局最优匹配，解决碰撞时相邻摆球混淆
   - 角度计算：检测框中心与悬挂点连线相对竖直方向夹角（最低点 0°，向右为正）
   - 降噪：统一时间网格 + 短缺失线性插值 + Savitzky-Golay 滤波（sg 窗口 11，阶数 3）
   - 输出各摆 CSV（`{basename}_angle_N.csv`）+ 汇总 CSV（`{basename}_all_pendulums.csv`）+ 标注视频（可选）+ 多面板时空图
   - 不做符号回归（多摆碰撞/能量传递系统，`_run_processing` 中 `sr_result = None`）

5. **符号回归** (symbolic_regression.py)
   - 纯 scipy。三种阻尼拟合 + ODE 验证 + 物理项库拟合

## 磁力牛顿摆标定（分阶段，与底层 ManualMarker 一致）

前端标定弹窗分三阶段（状态 `S.calibPhase: pivots → vertical → horizontal`，`S._pivotCount` 记录确认的悬挂点数）：

1. **pivots**：自由点击每个摆球的**悬挂点**（2~5 个，数量不预设，标多少算多少），Enter/『下一步』确认
2. **vertical**：竖直方向 2 点（上→下）
3. **horizontal**：水平方向 2 点（左→右）

按键：Enter=下一阶段 / C=上一步 / R=撤销上一点 / Esc=关闭。关闭弹窗（X/Esc/遮罩）不设限制，标定是否完成由『下一步』校验与开始处理时把关。

后端 `_run_processing` 以实际标定点数推导摆球数量 `num_pivots = len(calibration_points) - 4`（**不信任前端传入值**），范围 2~5，超范围报错。标定点顺序：pivot_1..pivot_N, vertical_1, vertical_2, horizontal_1, horizontal_2（显示坐标除以缩放比转原始坐标）。

## 交互模拟参数

- 阻尼系数 β：0 ~ 5 s⁻¹
- 固有频率 ω₀：0.5 ~ 20 rad/s
- 初始角度 θ₀：0 ~ 20°
- 画布固定比例尺：±25°
- 三种阻尼类型自动切换（ζ < 1 欠阻尼 / ζ = 1 临界阻尼 / ζ > 1 过阻尼）

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
| POST | `/api/heartbeat` | 页面心跳（看门狗自动关停依据，前端 Web Worker 每 2s 发送） |
| GET | `/api/serial/ports` | 列出电磁铁可用串口（过滤蓝牙 + COM1~20 补齐） |
| POST | `/api/serial/connect` | 连接电磁铁串口（115200 8N1），连接后先发 `0` 建立断电状态 |
| POST | `/api/serial/command` | 电磁铁命令 `0`(断电释放) / `1`(通电吸合)，返回提交时间戳 |
| POST | `/api/serial/disconnect` | 断开（若通电先断电保护） |
| GET | `/api/gripper/ports` | 列出夹爪可用串口 |
| POST | `/api/gripper/connect` | 连接夹爪串口（115200 8N1） |
| POST | `/api/gripper/command` | 夹爪命令 `U/D/L/R`(方向) / `O/C`(开合) |
| POST | `/api/gripper/disconnect` | 断开夹爪串口 |
| GET | `/api/openmv/ports` | 列出串口（标注 OpenMV 设备） |
| POST | `/api/openmv/connect` | 连接 OpenMV（`camera_port` 必填，`control_port` 云台可选） |
| POST | `/api/openmv/disconnect` | 断开 OpenMV 与云台 |
| GET | `/api/openmv/frame` | 单帧 JPEG（前端 canvas 轮询预览用，约 12 FPS） |
| GET | `/api/openmv/stream` | MJPEG 实时预览流（备用） |
| POST | `/api/openmv/record/start` | 开始录制帧 |
| POST | `/api/openmv/record/stop` | 停止录制，存 MP4 并自动创建处理 session |
| POST | `/api/openmv/gimbal` | 云台命令 `U/D/L/R/C` |
| GET | `/api/openmv/status` | OpenMV 连接与录制状态 |

## 模型路径

YOLOv8 模型查找优先级（`app.py` 中 `MODEL_PATHS`，不再硬编码本机绝对路径）：
1. 环境变量 `MODEL_W_BEST_{key}`（key 为 danbai/cizuni/niubai/ciliniudun；可用分号或逗号分隔多个路径，取第一个存在的）
2. 项目 `models/` 目录下的 `{key}_best.pt`

示例：`MODEL_W_BEST_DANBAI=C:\path\to\best.pt` 或默认 `models/danbai_best.pt`。

## 处理器说明

### danbai_processor.py — 单摆

1. YOLO 逐帧检测（`conf=0.25, iou=0.45, imgsz=1280`），最近邻跟踪避免目标跳变
2. 角度计算 → 在线平滑 → 提前停止 → Savitzky-Golay 滤波
3. 零偏校正 → 零交叉检测 → 等角度重采样 → 包络提取
4. 输出 CSV + PNG

### cizuni_processor.py — 磁阻尼摆

1. YOLO 逐帧检测，`sample_step` 降采样（保证 ≥80 数据点）
2. 角度计算（支持 flip）→ 高斯平滑（σ=0.8）
3. 输出双面板图，支持中文字体回退
4. 内部每次重新创建 YOLO 模型（避免 CUDA 上下文冲突）

### niubai_processor.py — 扭摆

1. 像素坐标 → 物理坐标（米）→ 角度计算（atan2 + 相位展开）
2. 卡尔曼滤波（前向 + RTS 后向平滑，`q=1.0, r=0.1`）
3. 零基线 → 周期提取 → 提前停止
4. 输出 CSV + 双面板图

### ciliniudun_processor.py — 磁力牛顿摆

1. 网页分阶段标定（悬挂点 N 个 + 竖直 2 点 + 水平 2 点，无 GUI 弹窗，见上文"磁力牛顿摆标定"）
2. YOLOv8 检测 + ByteTrack 跟踪（`model.track(persist=True, tracker="bytetrack.yaml")`），统一时间网格采样
3. 轨迹绑定（质心最近悬挂点，仅统计参考）+ 水平几何约束修正（匈牙利算法，`linear_sum_assignment`）
4. 角度计算（`compute_angle`，最低点 0°、向右为正、`right_positive` 可翻转）→ 短缺失插值 + Savitzky-Golay 分段滤波
5. 输出 CSV（各摆 + 汇总）、标注视频（`write_annotated_video`，mp4v 编码、临时文件避免中文路径问题）、多面板时空图（`plot_all`）
6. 进度回调覆盖跟踪 + 视频两遍，参数 `gen_video` 控制是否输出标注视频

### symbolic_regression.py — 参数拟合

纯 scipy。三种阻尼拟合 + ODE 验证 + 物理项库拟合（磁力牛顿摆不做符号回归）。

## 性能要点与已知瓶颈

实测环境 RTX 4060 Laptop：单帧 YOLO 推理约 12~15ms（CPU/GPU 均可达 50~85 fps），推理本身不是瓶颈，慢在流程叠加：

1. **逐帧解码 + 推理随时长线性放大**：`cap.read()` 约 8ms/帧 + 推理约 12ms/帧，30fps 视频吞吐约 50 帧/秒
2. **单摆默认 `target_cycles=400`**：提前停止需检测到 800 次零交叉，长视频可达数万帧
3. **磁力牛顿摆两遍处理**：第一遍 ByteTrack 跟踪（`imgsz=1536`），第二遍重新解码写标注视频（mp4v 编码约 30ms/帧），全实验最慢
4. **符号回归发生在进度 100% 之后**：`progress_callback` 只覆盖帧处理阶段；随后 SR 执行 36 组 `least_squares` 网格搜索（每组最多 1 万次迭代）+ ODE 验证 + matplotlib 出图，约 10~60 秒无任何进度提示——前端体感"最后卡住"
5. **杂项开销**：cizuni 每次运行重新 `YOLO(model_path)`（忽略预加载模型，为规避 CUDA 上下文问题）；缓存键对整个视频文件算 MD5（1GB ≈ 2s）；服务启动后首次推理有 CUDA 上下文初始化

**结果缓存的真相**：`_run_processing` 把结果写进 `_progress[session_id]["cache"]`，但处理结束时 `finally` 会 `_progress.pop(session_id)`，缓存随之销毁——实际几乎不产生跨请求命中，仅在同一次请求内有意义。

**候选优化（尚未实施）**：SR 独立进度阶段上报 / 单摆周期数按视频长度自适应 / cizuni 复用预加载模型 / 牛顿摆默认关标注视频 / 扭摆 `find_peaks` 只查尾部窗口（现为每帧全量，O(n²)）。

## 注意事项

- YOLO 模型懒加载（`get_model(key)` 线程安全按需加载），启动秒开；处理器都接受 `model=None` 自行回退
- **自动关停**：前端 Web Worker 每 2s POST `/api/heartbeat`（带随机客户端 ID）；页面卸载时 `pagehide` + `sendBeacon` POST `/api/client_exit` 告别。`_watchdog_loop`（0.5s 周期）在客户端集合清空且无处理/录制 → 2s 宽限期 → `_shutdown_safety()`（电磁铁断电、关全部串口、断 OpenMV）→ `os._exit(0)`。**刷新保护**：GET `/` 会取消待关停并刷新 `_last_page_request`，距上次页面请求 6s 内一律不关停（新页面 Worker 首次心跳因 CDN 脚本加载会晚到）。心跳 6s 未刷新视为离线（浏览器崩溃兜底）；多标签页只有最后一个关闭才停；从未有页面连接过（命令行调试）不退出
- `_process_lock` 确保同一时刻只有一个 YOLO 处理任务
- Session 30 分钟自动清理，超过 1 GB 被 Flask 拒绝
- 处理结果有缓存机制（视频 MD5 + 参数 hash），但仅在一次处理请求内有效（见"性能要点"）
- `KMP_DUPLICATE_LIB_OK=TRUE` 解决 PyTorch 与 OpenCV 冲突
- 扭摆依赖 `filterpy`，AI 问答依赖 `openai`
- 新增实验：需完整注册——`app.py` 的 `MODEL_PATHS`（本地路径 + models/ 回退）、`CALIB_SPECS`、`_run_processing` 分发分支、`get_teaching_content` 的 `allowed` 集合；新增 `processors/xxx_processor.py` 并在 `processors/__init__.py` 导出；新增 `teaching/guide_xxx.json`；前端 `index.html` 添加类型卡片、标定逻辑与参数组（参考磁力牛顿摆 ciliniudun 的完整实现）
- 前端输出目录由用户指定，需确保可写
- 磁力牛顿摆输出文件以视频 basename 为前缀（`{basename}_angle_N.csv` 等）；其他处理器 CSV 格式：`time_s, angle_deg`（前两列）

## Git 自动提交与推送规则

本项目由 agent 维护：**每完成一个有意义的最小改动单元后，必须自动 commit 并 push 到 GitHub**，保持远程与本地同步。

### 执行命令

```bash
git add -A
git commit -m "简洁描述改动内容"
git push
```

### commit message 规范

- 用中文写明改了什么
- 不超过 30 个字
- 示例：补充依赖与模型权重、修复上传类型校验

### 提交时机

- 完成一个完整的功能/修复/文档改动后提交一次，**不要每保存一次就提交**
- 攒够一个「有意义的最小改动单元」（一个功能、一个 bug 修复、一段文档）再提交
- commit 后紧跟 push，保持 GitHub 同步

### 大改动先建备份分支

- 涉及大段 UI/样式重构前，先建备份分支（如 `git branch backup/<说明>`）；改崩可用 `git checkout backup/<说明> -- <文件>` 一键还原
- 已有备份分支：`backup/pre-stepbar-redesign`（步骤条重构前快照，仅本地）

### 禁止提交的内容（提交前务必检查）

- API key、token、密码等任何敏感凭证
- `uploads/`、`__pycache__/`、`*.pyc`、`server.pid`、`.env`（已在 .gitignore 忽略）
- 本地调试用的临时文件

### push 凭据

- 远程地址：`https://github.com/qiaoyuzheng0804-create/pendulum_web.git`
- 凭据通过 git credential helper 提供
- **严禁把 token 写进任何会提交到 git 的文件**