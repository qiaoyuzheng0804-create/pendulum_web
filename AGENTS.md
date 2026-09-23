# AGENTS.md — 大学物理实验平台

面向 AI 编码助手的项目指南（ZCode 工作区指令文件，本文件为唯一权威版本）。

## 项目是什么

大学生物理实验平台（由"单摆运动视频分析器"升级而来）。核心能力：

1. **视频分析实验**（4 种摆：单摆/磁阻尼摆/扭摆/磁力牛顿摆）：上传或树莓派实时取流 → YOLOv8 逐帧检测 → 提取角度-时间数据 → 拟合阻尼参数，输出 CSV/图/标注视频。**四种摆都有完整实验指导**（`teaching/guide_<id>.json`），入口：工作台"去预习"、分析向导第 1 步"实验指导"按钮、指导页流程条"打开视频分析"。
2. **指导型实验**（三线摆、杨氏模量、牛顿环、迈克尔逊、示波器、落球法粘度等）：实验指导 + 专属报告模板（含数据表格）+ AI 辅导，无视频分析。
3. **实验课全流程闭环**：课前预习（学生点"去预习"跳到该实验的实验指导页，**通读至最后一页自动完成预习标记**）→ 课中实验（指导型实验手动点"完成课中"；**视频分析实验以跑完分析、拿到实验数据为完成标准，出结果自动标记**，流程条不提供手动按钮）→ 课后报告（按模板撰写提交，提交后可**撤销提交**回到草稿）→ 教师批改打分/退回 → 学生查看反馈。`progress` 表记录 预习/课中；`submissions` 记录 报告/批改。
4. **师生问答（公共讨论区）**：所有登录用户可见全部提问与回复；学生提问（文字 / 图片 / 语音附件），教师和全体学生均可回复，支持多轮；提问者/回复者可**撤回删除**自己的提问/回复（撤回提问连同回复与附件一并删除）。`scope=mine` 可只看自己的提问。
5. **工作台数据看板（默认首页视图）**：教师看 班级维度成绩/报告进度/待批改/成绩分布/各实验报告量/待回复提问；学生看 个人流程进度与统计；未登录看公共概览。在线人数来自看门狗 `_clients`（`/api/dashboard/online`）。
6. **AI 实验助手**：右侧常驻面板，自研轻量 agent（工具调用循环），随上下文全流程辅导。

三栏类 IDE 布局：左导航（工作台/实验列表/师生互动/报告中心）+ 中央区 + 右 AI 面板。
（"学习中心"分组与"理论知识"视图已整体删除，勿再引用 `view_theory`/`loadTheory`/`renderTheory`。）

## 运行

- 启动：`python app.py`（或 `启动服务器.bat`），端口 5000。启动脚本先按 `server.pid` 停掉上次实例，再**兜底清掉仍占着 5000 端口的 python**（pid 文件丢失/过期时新实例会 "Address already in use" 直接退出，表现成"双击没反应"），等端口真正释放后才启动，并打印"正在启动/已就绪"；60 秒没起来会明确报错并 `pause`（不再静默打开一个打不开的页面）。
- 依赖：见 requirements.txt（flask/opencv/ultralytics/torch/openai/pyserial 等，模型权重在 `models/*_best.pt`，缺失时 YOLO 自动降级）。
- LLM 配置两处：**网页「模型配置」**（教师登录 → AI 面板顶部齿轮，存 `llm_config.json`，即时生效）优先于 `.env`（`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`，任意 OpenAI 兼容服务，作后备）。API：`GET|POST /api/llm_config`（teacher_required）、`POST /api/llm_config/test`；合并逻辑在 `app.py get_llm_config()`（网页字段逐项优先，留空回退 .env）。
- **看门狗**：所有页面关闭后约 10s 自动关停服务器（`server.pid` + `/api/heartbeat`）。调试时若进程"自己消失"是正常行为，不是 bug。
- 演示账号：`teacher/123456`（教师）、`student/123456`（学生）、`s240101/123456`（物电 2401，看板演示班级学生，共 3 班 12 人自动预置；删 platform.db 即重置）。
- 删除交互模拟：该视图已移除（nav/view/CSS/JS 与 ai_agent 映射均清理），勿再引用 `sim`。
- 树莓派地址：`pi_config.json`（已 gitignore）记上次连上的 `host/port`，前端回填用；换网络后 DHCP 换 IP 就手输一次。

## 架构与关键文件

| 文件 | 职责 |
|---|---|
| `app.py` | 全部视频分析/串口/树莓派与 OpenMV 路由 + 页面入口；YOLO 懒加载；全局处理锁 |
| `experiments.py` | **实验注册表（单一数据源）**。新增实验 = 加一条记录 + `teaching/guide_<id>.json` + `report_templates/<id>.json`，不改 app.py |
| `platform_hub.py` | Blueprint：登录(session cookie)/用户(含班级)/报告提交批改退回/预习课中进度/师生问答+附件上传/数据看板统计。SQLite(`platform.db`)。**不能改名为 platform.py**——会遮蔽标准库 platform，cv2/torch 会炸 |
| `demo_reports.py` | 演示报告完整内容（6 分节 + 表格数据，公式 `$...$` 包裹）。`_seed_demo_if_empty()` 与 `_upgrade_demo_reports()`（按旧标记识别并升级旧演示报告，幂等）共用 |
| `ai_agent.py` | AI 工具调用循环：get_experiment_guide / get_report_template / analyze_numeric_data；`build_system_prompt` 注入实验/步骤/角色上下文 |
| `processors/` | 每种摆一个处理器（`_run_processing` 的 if/elif 分派）+ pi_manager / openmv_manager + symbolic_regression |
| `processors/pi_manager.py` | **树莓派 5 摄像头 + 二维云台（现行采集设备）**：TCP 收帧（**RPJ2** 协议：33 字节头 `!4sIQQHHHHB` = magic/JPEG长度/帧号/传感器时间戳ns/宽/高/**pan_pulse_us/tilt_pulse_us/limit_code**；旧版 24 字节头 `!4sIQQHH` 的 RPJ1 仍接受，此时无云台字段）+ 预览 + **恒定帧率网格录制**（帧按 `sensor_timestamp_ns` 折算时间格，缺格补上一帧、同格丢弃，落盘 ffmpeg 转 H.264 MP4，无 imageio-ffmpeg 时退 cv2 mp4v）+ 逐帧时间戳 CSV（`pi_capture_<ts>_frames.csv`，含 `pan_pulse_us/tilt_pulse_us/gimbal_limit_code` 三列）。**云台命令走同一条 TCP 反向单字节**（U/D/L/R/C，响应 NODELAY 即时发出），方向/步长/软限位都在树莓派侧。对端是套件 `raspberry/capture_stream.py`（`--no-record` 纯传输 + GPIO12/13 硬件 PWM 云台，端口 9000） |
| `processors/openmv_manager.py` | **已退役**（USB 串口取帧 USBDBG V1 + USB-TTL 云台）。硬件已不在，前端入口已撤，但后端与 `/api/openmv/*`（含 `/api/openmv/gimbal`）保留可用——**别删**，也别再把 OpenMV 的界面加回来 |
| `processors/niubai_processor.py` | **完全对齐底层 `analyze_pendulum_fixed.py`**：3 点标定（原点 + 竖直方向两点）、解缠绕/离群剔除/补缝、Savitzky-Golay 自适应降噪（含防过平滑自检）、包络寻峰、分段阻尼、FFT 周期；输出 `angle_time_data.csv`(`time_s,angle_deg_raw,angle_deg,gap_filled`)/`angle_time_plot.png`/`annotated_video.mp4`/`run_meta.json`。**标注视频里画标定出来的坐标系**（`draw_overlay`：青色 Y 轴=②→③ 竖直方向即 θ=0 基准、橙色 X 轴=θ=+90° 方向、黄原点、绿角度弧 + 左下读数面板），配色与前端标定画布 `drawNiubaiFrame` 一一对应（改配色两边同步）；写盘走 `_AnnotatedVideoWriter`（`imageio-ffmpeg` 管道 → Windows MSMF/avc1（只认 ASCII 路径，非 ASCII 先写临时文件再搬回）→ cv2 自带 H.264 → mp4v 兜底），结果里回 `video_codec/video_playable/video_error`——**只有 h264 浏览器才能内联播放，mp4v 时前端只给提示不放播放器** |
| `processors/cizuni_processor.py` | **完全对齐底层 `cizuni3.py`**：3 点标定、`compute_angle` + `verify_compute_angle` 自检（**勿改回 `sin(-a)/cos(-a)`，会给每个角度叠加恒定 2a 偏差**）、慢放倍数支持 `"auto"`（读 MP4 的 `com.android.capture.fps`）、`gaussian_filter1d`；输出 `pendulum_angle.csv`(`time(s),angle(deg),angle_raw(deg)`)/`angle_time_plot.png`/`annotated_video.mp4` |
| `processors/danbai_processor.py` | 单摆：3 点标定（悬点 + 竖直方向两点）+ 逐帧检测/选点 + 早停（`enable_early_stop_by_cycles`，凑够周期数就停）；输出 `<basename>_angle_time.csv`/`_peaks.csv`/`_angle_time.png`/**`annotated_video.mp4`**（由 `gen_video` 控制，与源同帧率、从第 0 帧起逐帧写，**跳过段原样写入**，保证视频帧号/时间轴与 CSV 的 `time_s` 对齐） |
| `teaching/*.json` | 教学内容（guide_<id>.json；theory.json 尚存但前端已无入口），API 下发，前端渲染 |
| `report_templates/*.json` | 报告模板：每实验 6 个分节（purpose/principle/apparatus/data/analysis/conclusion）+ 分节内可选 `tables`（结构化数据表格，见下） |
| `templates/index.html` | **单文件前端**（约 3400 行，含全部 CSS/JS），无构建步骤 |
| `templates/login.html` | 登录页（独立小页） |

## 时间基准（扭摆/磁阻尼摆，务必别改错）

- `real_fps = video_fps × slow_motion_factor`，`t = (帧号 − 起始帧) / real_fps`
  —— 即**真实时间 = 视频时间 ÷ 慢放倍数**（慢动作 N 倍指的是"拍摄帧率是播放帧率的 N 倍"）。
- **写成 `× slow` 或 `fps/slow` 都会让整条时间轴差该倍数**（旧网页版就踩过：16 倍慢放的 3.06 s 周期被报成 49.2 s）。日志里的 `真实采样率` 可直接自查。
- 磁阻尼摆支持 `slow_motion_factor="auto"`：读 MP4 的 `com.android.capture.fps` 标签（手机慢动作导出常抹掉原始帧率，只有这个标签可靠）；读到就用，读不到按 1.0 并提示。
- **树莓派采集这条路的慢放倍数填 1**：它是实时取流，时间基准来自摄像头传感器时间戳，录制时已按时间格对齐（见 `processors/pi_manager.py`），视频容器帧率就是真实时间轴。
  **别把"按到达顺序写盘"当成时间轴** —— 树莓派 JPEG 编码跟不上 60 FPS 时会掉到 40 上下、局域网也会丢帧，原样写盘会让整条时间轴被压缩（阻尼系数跟着错）。补帧数记在 `filled`、丢格数记在 `trimmed`（>0 说明中途断流过，该段实验要重拍）。

## 平台 API 速查（platform_hub.py / app.py）

- 认证：`POST /api/auth/login|logout`、`GET /api/auth/me`（含 class_name）
- 报告：`GET|POST /api/reports`（教师支持 `?exp=&status=&class=` 筛选）、`GET /api/reports/<id>`、`POST .../submit`、`POST .../<id>/withdraw`（学生撤销提交：submitted→draft）、`DELETE /api/reports/<id>`（学生删除草稿；已提交需先撤销，已批改不可删）、`POST .../<id>/grade|return`、`GET /api/report_template/<exp>`
- 全流程：`POST /api/progress/pre|lab`（学生标记预习/课中）、`GET /api/flow`（学生=10 实验状态；教师=`?class=` 全班聚合）
- 问答：`GET /api/qas`（公共讨论区：默认 scope=all 全员可见；scope=mine 只看自己的；教师可加 `&pending=1` 筛待回复）、`POST /api/qas`（学生提问，attachments=[附件id]）、`GET /api/qas/<id>`（全员可看）、`DELETE /api/qas/<id>`（撤回提问，仅本人，连回复附件一起删）、`POST /api/qas/<id>/reply`（教师和全体学生均可回复）、`DELETE /api/qas/reply/<rid>`（撤回回复，仅本人）
- 附件：`POST /api/qas/attachment`（multipart file，图片/音频/文档白名单）、`GET /api/qas/attachment/<id>/file`（绑定到问答线程后登录用户均可见）
- 看板：`GET /api/dashboard/summary`（未登录/教师/学生三种视图；教师 `class_stats` 含服务端算好的 `grade_rate`——**批改率 = 已批改报告 / 已提交报告，同一学生×同一实验只按最新一份计入**，前端只展示不重算）、`GET /api/dashboard/online`（在线人数=心跳客户端数，app.py）
- 用户/班级：`GET|POST /api/admin/users`（导入行格式：学号,姓名,班级）、`GET /api/classes`（教师）
- 树莓派采集（现行）：`GET /api/pi/config`（返回上次连上的 `{host, port}`，存 `pi_config.json`，前端进"实验获取"面板时回填，手改过就不覆盖）、`POST /api/pi/connect`（Body `{host, port}`，等到首帧才算成功，成功后把地址记进 `pi_config.json`）、`POST /api/pi/disconnect`、`GET /api/pi/status`（1s 轮询：fps/帧数/录制账目/**protocol/gimbal_pan_us/gimbal_tilt_us/gimbal_limit_code/gimbal_limit**）、`GET /api/pi/frame`（最新 JPEG，仅在网页勾了"本机预览"时被请求；带 `?w=880` 时服务端降采样再发——预览更顺，录像/分析用原始帧不受影响；前端为自定速取图，角标显示的是预览刷新率而非相机帧率）、`POST /api/pi/record/start`（Body `{fps}`）、`POST /api/pi/record/stop`（Body `{motion_type}`，返回 `{**录制账目, gimbal_moved/gimbal_span_us, session_id, first_frame_base64, total_frames, fps}`，前端据此直接进标定）、`POST /api/pi/gimbal`（Body `{command: U|D|L|R|C}`，经同一条 TCP 反向发单字节；RPJ1 旧协议下明确报错）
- OpenMV（退役但保留）：`/api/openmv/ports|connect|disconnect|frame|stream|record/start|record/stop|gimbal|status`
- 标注视频预览（扭摆结果页第 5 步）：`GET /api/output_video/<session_id>`（按会话取"本次分析写出的标注视频"，登录可见，`send_file(conditional=True)` 支持 Range 拖进度；路径由服务端从 `session_store` 取，**前端不能传任意路径**）。手动标记窗口的翻帧取图：`GET /api/video_frame/<session_id>?idx=N`

演示数据：`_seed_demo_if_empty()` 在 users 无带班级学生时注入 3 班 12 学生 + 24 份**完整内容**的演示报告（含数据表格与公式）+ 演示进度/问答（幂等，删除 platform.db 重置；旧库中的占位演示报告由 `_upgrade_demo_reports()` 按 `（演示数据）` 标记自动升级）。

## 前端设计规范（务必遵守，勿回退）

- **主题**：「实验手册」暖纸墨色 —— 暖白纸底 `#f5f3ef` + 近黑墨 `#1c1917` + 焦橙强调 `#c2410c` 单一强调色；发丝线分隔、圆角刻度 10/8/6/4px、阴影极轻。所有颜色走 `:root` CSS 变量，改主题只改变量。
- **禁止**：靛紫/渐变文字/霓虹辉光/emoji 当图标/长页下滑式布局/纯黑投影。
- **布局**：三栏 shell（`.shell` 网格，`grid-template-rows:minmax(0,1fr)` 锁行高——不锁则中栏长内容把侧栏/AI 面板撑出视口，此 bug 修过一次）。
- **分析流程 = 5 步向导**：`#wzp1..5` 页 + `goToPage()/canProceed()/updateStepBar()`；每步一屏、底部固定 上一步/下一步，完成自动翻页。步骤：1 选择类型 → 2 视频与参数 → 3 标定 → 4 运行 → 5 结果。
- **实验获取面板**（向导第 2 步）：STM32 电磁铁 / 夹爪云台释放控制 + **树莓派拍摄**（元素 `#pi_*`、JS 前缀 `pi`、`togglePiConnect/startPiRecord/stopPiRecord`）+ **二维云台控制**（`.pi-gimbal-btn` + `#pi_gimbal_status`，方向键 POST `/api/pi/gimbal`，真实位置由 1s 状态轮询回填，撞软限位时整行变红显示中文提示）。**取景与拍摄按钮在"取景窗"弹窗里**：面板上只有 `#btn_pi_view`（连接后可用）打开 `#pi_modal_overlay`，弹窗与标定窗共用同一套可自由拉伸逻辑（`resizableModalEl()` / `fitActiveModal()` / `.calib-resize-grip`，画布由 `fitPiCanvas()` 等比适配），开始/停止拍摄按钮在弹窗底部；关窗即停止取帧，收帧/录制/云台/状态轮询都不依赖它。OpenMV 的串口面板已撤下，别再加回来（后端路由仍在）。
- **手动标记（标定）窗口 = 一个窗口两步**（扭摆/磁阻尼摆）：第①步在弹窗顶部"计时第一帧"条里翻帧选释放帧（滑条 / -10 -1 +1 +10 / `calibFrameSet`），点"确认起始帧"进入第②步点选标定点。选中的帧号经 `start_frame` 传给处理器，**该帧即 t=0（不是被跳过）**；后端由 `GET /api/video_frame/<session_id>?idx=N` 提供（帧宽 ≤1200，与首帧同坐标系），由 `_SessionFrames` 负责：**每会话一个后台解码线程 + 有界缓存 + "只追最新请求"的合并**（4K 随机 seek 单帧可达秒级，顺序读约 40ms/帧，网页翻帧又常来回拖动），响应头 `X-Total-Frames` 回传校正后的总帧数，前端据此修 `S.totalFrames`（**容器帧数会少报甚至报 0，报 0 时滑块会失效——务必保留 `_probe_frame_total` 校正与前端自愈**）。前端取帧有 120ms 防抖。弹窗可自由拉伸：右下角有明确的缩放柄 `#calib_resize_grip`（原生 `resize:both` 柄太小），画布位图固定为图像原始像素、显示尺寸由 `fitCalibCanvas()` 适配，点击坐标一律走 `canvasImageCoords()`（按 `getBoundingClientRect` 反算）——**别再改回直接除 `S._modalScale`**，窗口拉伸后会偏。
- **指导/报告 = 分节翻页**：`renderGuide` 输出 `.doc-sec` 节 + `initDocPager` 页码器（‹ n/N ›），禁止改回整体长页。（理论视图已删除，`renderTheoryMarkdown` 名字虽带 theory 但仍被指导页/批改视图共用，勿删。）
- **结果页（第 5 步）标注视频预览**：`#video_preview_area` + `#video_preview_player`，只在**扭摆**出（`renderResultVideoPreview` 里判 `S.motionType!=='niubai'` 直接 return），源是 `api/output_video/<session_id>`；`video_playable` 为假（mp4v）时只给文字提示不放播放器。换实验类型 / 重新分析时都要 `hideResultVideoPreview()` 收掉旧预览并清 `src`。
- **标定画布上的坐标系（扭摆）**：`redrawCalibCanvas` 里 `calibPoints` 满 3 点后调 `drawNiubaiFrame()`——青色 Y 轴 = ②→③ 竖直方向（θ=0 基准）、橙色 X 轴 = θ=+90° 方向（`vx=-uy, vy=ux`，与后端 `signed_angle_deg` 同向）、黄原点十字 + 白环，反方向用暗一档虚线；轴线长度用 `sFitLen()` 收进画布，避免原点贴边时箭头和轴名跑到画外。**配色与后端 `draw_overlay` 一一对应，改一处要同步另一处。**
- **图标**：只用 SVG symbol 体系（`#i-*`），线宽 1.7；不加 emoji/Unicode 图标。
- **公式（KaTeX）**：`static/katex` 本地部署。教学 JSON 的 `formula/solution/symbol` 等字段走 `data-katex` 属性直渲（可裸写 LaTeX）；`content` 等正文字段必须用 `$...$` / `$$...$$` 包裹，由 `renderKaTeXinHTML` 渲染。指导渲染选择器已含 `p/li/td/th`。

## 报告模板与数据表格

- 模板 JSON 分节可带 `tables: [{id,title,columns:[...],initial:[[...]],hint}]`：`initial` 既定行数也可预填首列行标签（如"金属丝直径 d /mm"）。
- 学生编辑器 `repTableHtml(t,content,true)` 渲染输入格（可"加一行"）；批改视图只读渲染，未填格显示"—"。
- **表格数据键名约定：`tbl_<table.id>`**（保存于 `content` JSON）。保存（saveReport）与回显（repTableHtml）两处必须一致——曾因一侧漏加 `tbl_` 前缀导致回显全空，改时勿再破坏。

## 已踩过的坑（改前端前先读）

1. **index.html 是手工单文件**：改 DOM 结构极易 div 不配平（已发生过：页脚被吞进隐藏页）。改完用脚本对该区域做一次开闭标签配平检查。
2. **TDZ**：顶层立即执行的函数（如 `updateStepBar(1)`）所引用的变量必须在声明顺序上先行（`CURRENT_STEP` 等因此炸过）。顶层变量一律 `var` 或前置 `let`，函数内懒引用的可放 `window.*`。
3. **懒加载标志**：`switchView` 与 `loadGuide` 各自查重标志，别两处同管一个标志（双重置位曾让页面永远停在骨架屏）。
4. 模板自动重载开着，但**浏览器端要 Ctrl+F5 强刷**；验证改动用浏览器截图实测，别只看代码。
5. 同步处理有全局锁（一次只处理一个视频）；设备录制（树莓派 / OpenMV）期间看门狗不关停。
6. **marked v12 会吃掉 `$...$` 内的 `\,` `\;` `\{` `\}`**（如 `0.087\,\mathrm{rad}` 变 `0.087,rad`）。`renderTheoryMarkdown`/`renderChatMarkdown` 已用 `protectMathSpans/restoreMathSpans` 占位符把公式摘出再过 marked，**勿删这套保护**，也别绕过它直接 `marked.parse`。
7. **teaching/*.json 里反斜杠必须双写**：JSON 中单写 `\frac`、`\right` 会被解码成换页符/回车符 + 残字（三线摆指导踩过，公式变成 `rac{...}` 乱码）。写完用 `json.load` + 扫控制字符自检。公式统一 `$`/`$$` 包裹，KaTeX 不认的字符（≪、∝、℃、全角逗号）在公式内要写成 `\ll`、`\propto`、`^{\circ}` 等；行文里的 Unicode 记号（θ²、T_d 等）是刻意的纯文本风格，别批量改写。
8. **platform_hub.py 写库必须放在 `with _conn() as c:` 块内**：sqlite3 的 `with` 负责提交，块外执行的 UPDATE 既不提交、还会一直持有写锁，把后续所有写请求打成 `database is locked`（迁移 `_upgrade_demo_reports` 踩过）。
9. **已提交/已批改的报告前端只读渲染**（`openReportEditor` 按 `status` 分支：draft 走 textarea 编辑器 + `.rep-preview` 实时预览框（`repPreview()` 输入防抖渲染 Markdown+KaTeX，避免只见 TeX 源码），其余走 `repTableHtml(...,false)` + `renderChatMarkdown` 只读视图 + 可选"撤销提交"按钮）；批改视图、学生只读视图与草稿预览的 `.theory-note` 都要在 `whenKatex` 里补渲染一次，否则 KaTeX 未就绪时公式闪现 `$...$` 源码。
10. **富文本安全链勿拆**：`marked.parse` 不净化 HTML（`<img onerror>` 原样通过，实测过），所有用户可控内容（报告/问答/AI 回复）经 `renderChatMarkdown`/`renderTheoryMarkdown` 渲染时必须保留 `sanitizeHtml()`（DOMPurify，本地 `static/purify.min.js`）收口；`esc()` 已转义 `& < > " '`，**所有属性值/textarea 拼接必须走 esc**（alt/placeholder 等上下文，旧版只转义 `&<>` 是 XSS 源）；复制按钮走 `copyBtnHtml()` 的 `data-path` + 事件委托，勿改回内联 onclick 拼路径（反斜杠必炸）。
11. **消耗资源的接口必须登录**：上传视频、运行分析、串口/夹爪/树莓派与 OpenMV 控制、AI 对话已全部加 platform_hub 的 `login_required`（AI 另有每用户 15 次/分钟限频，context.user 由服务端 session 覆盖写入）；心跳 `/api/heartbeat`、`/api/client_exit`、看板概览保持公开（登录页要用心跳保活）。attachments 表有 `owner` 列（幂等迁移补列），temp 附件仅上传者可见、绑定他人附件会被 UPDATE 条件拒掉——新增附件相关逻辑勿绕过 owner 校验。
12. **app.py 里别把类方法插错缩进层级**：`_probe_frame_total()` 曾被写在 `_SessionFrames` 的 `__init__` 与其余方法之间（同级缩进），导致 `_put/_loop/_fulfill/get/close` 全变成该函数内 `return` 之后的死代码，`/api/video_frame/*` 一律 404（`'_SessionFrames' object has no attribute '_loop'`）。新增函数请放到类定义之外。
13. **标注视频要浏览器能播才有预览**：浏览器不认 cv2 的 mp4v，结果页内联播放必须是 H.264（`video_playable` 才给 `<video>`，否则只提示）；写盘降级链在 `niubai_processor._AnnotatedVideoWriter`：ffmpeg 管道 → Windows MSMF/avc1（**只认 ASCII 路径**，中文结果目录要先写临时文件再搬回）→ cv2 H.264 → mp4v 兜底。另外 OpenCV 的 FFmpeg 后端缺 libopenh264 时 `isOpened()` 仍返回 True 却一帧都写不进去，所以选后端前要用临时文件"试写几帧再读回"探针（`_probe_cv_h264`）。

## 约定

- 提交信息用中文、一句话主题 + 分条要点。
- 不要动 `firmware/`（Keil 工程）与 `models/*.pt`。
- 旧的 `index_html_clean.html` / `index_html_full.txt` 是历史备份，未入库，别参考也别删（等用户处理）。
