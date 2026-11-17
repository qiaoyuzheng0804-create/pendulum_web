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
| `processors/common.py` | 公共图像处理/标定辅助与 `H264VideoWriter`（`imageio-ffmpeg` libx264 管道 → Windows MSMF/avc1（只认 ASCII 路径，非 ASCII 先写临时文件再搬回）→ cv2 自带 H.264 → mp4v 兜底，供各摆处理器复用） |
| `processors/observation_pipeline.py` | **单摆 / 磁阻尼摆共享视频观测提取管线**：逐帧顺序解码保真时间轴、`start_frame`（`t=0`）截取（选定起始帧之前跳过 `model.predict`）、`measured_angle`、诊断性展示平滑、`frame_audit` 审计表与 H.264 `annotated_video.mp4` 单遍写出（**不以插值/平滑覆盖原始角度，漏检不在反演 CSV 内插补**） |
| `processors/torsion_signal.py` | **扭摆时序与阻尼分析核心**：连续有限段划分、FFT 周期估计、周期极大值提取、上下包络及分段阻尼统计 `decay_report` |
| `processors/niubai_processor.py` | **完全对齐最新 `analyze_pendulum_fixed.py`**：3 点标定（原点 + 竖直方向两点）、选定起始帧 `start_frame`（`t=0`，移除跳过开头秒数）、漏检时重置 `prev_angle = None` 且保留 `angle_time_data.csv` 原始非空/空值（`gap_filled=0`、`angle_deg == angle_deg_raw`）、Savitzky-Golay 诊断降噪（仅写 `angle_time_diagnostics.csv`）、`torsion_signal` 分段阻尼/FFT 周期；输出 `angle_time_data.csv` / `angle_time_diagnostics.csv` / `cycle_metrics.csv` / `calibration_preview.png` / `angle_time_plot.png` / `angle_time_detail.png` / `annotated_video.mp4` / `run_meta.json`。**标注视频画标定坐标系**（青色 Y 轴=θ=0、橙色 X 轴=θ=+90°、黄原点、绿角度弧，与前端 `drawNiubaiFrame` 同步），写盘走 `common.H264VideoWriter`（别名 `_AnnotatedVideoWriter`），回传 `video_codec/video_playable/video_error`（**只有 h264 浏览器才能内联播放**） |
| `processors/cizuni_processor.py` | **完全对齐最新 `cizuni` + `observation_pipeline.extract`**：兼容 `vertical_ref_1/2/pivot` 与 `vertical_start/end/center` 3 点标定、`compute_angle` + `verify_compute_angle` 自检（**勿改回 `sin(-a)/cos(-a)`**）、慢放倍数支持 `"auto"`（读 MP4 的 `com.android.capture.fps`）、`start_frame`（`t=0`）；输出 `pendulum_angle.csv`（仅含有效观测帧供符号回归直读）、`pendulum_angle_frame_audit.csv`、`提取清单.json`、`angle_time_plot.png`、`annotated_video.mp4` |
| `processors/danbai_processor.py` | **完全对齐 `danbai_ultimate_version.py` + `observation_pipeline.extract`**：3 点标定（悬点 + 竖直方向两点）+ 选定起始帧 `start_frame`（`t=0`，不再使用周期早停 `target_cycles`）；输出 `<basename>_angle_time.csv`、`<basename>_frame_audit.csv`、`提取清单.json`、`<basename>_angle_time.png`、`annotated_video.mp4` |
| `processors/ciliniudun_processor.py` | **完全对齐最新 `magnetic_newton_cradle.py`**：选定 `start_frame`（`t=0`，移除 `skip_start_s`）+ 可选 `skip_end_sec`、`model.predict` + 黄色摆球 HSV 亚像素质心（`_subpixel_yellow_center`）+ 多摆联合最小二乘圆弧拟合精修支点（`refine_origins_by_circle_fit`）、仅对内部小缝隙插值（不做边界外推）；输出 `<prefix>_all_pendulums.csv` / `<prefix>_all_pendulums_raw.csv` / `angle_1..N.csv` / `tracked_video.mp4`（`H264VideoWriter`）/ `<prefix>_all_pendulums_plot.png` |
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

- **主题与设计令牌**：「实验手册」暖纸墨色 —— 暖白纸底 `#f5f3ef` + 近黑墨 `#1c1917` + 焦橙强调 `#c2410c` 单一强调色；发丝线分隔、圆角刻度 `--r-lg..xs`（10/8/6/4px）、字号刻度 `--fs-xs..2xl`、间距刻度 `--sp-1..8`、等宽数字字体 `--mono`、状态胶囊色 `--st-*`。所有颜色走 `:root` CSS 变量，改主题只改变量。
- **禁止**：靛紫/渐变文字/霓虹辉光/emoji 当图标/长页下滑式布局/纯黑投影/冗余中英双语副标签（`.en`/`.eng` 已清理，勿再加回）。
- **布局与左导航**：三栏 shell（`.shell` 网格，`grid-template-rows:minmax(0,1fr)` 锁行高；右栏默认宽 `--w-ai:340px`，折叠状态记 `localStorage.ai_off`）。左导航顶栏用单摆品牌徽标 `#i-brand-pendulum` + `BRAND_NAME` + `BRAND_SUB`（与 `login.html` 保持一致）；主导航项为「首页」+ 三个分组：`智能分析实验`（副标题 `视觉识别 · 方程反演`，4 个实验配专属图标 `#i-exp-danbai/cizuni/niubai/ciliniudun`）、`拓展教学实验`（6 个指导实验）、`教学协作`（合并「师生问答」与角色化报告入口：学生显示「我的报告」、教师显示「批改台」、未登录显示「实验报告」）。
- **首页（默认视图）= 核心方法横幅 + 教学看板**：顶部 `homeHero()` 展示贯通全平台的 4 阶段教学闭环（`.pipeline`：`01 课前预习 → 02 课中实验 → 03 课后报告 → 04 教师批阅`，均可点击跳转，无固定高亮）+ 4 张主线实验卡（`.exp4`，含反演目标、KaTeX 核心方程、「开始分析 / 预习」双按钮）+ 在线人数胶囊；已登录用户可收起横幅（状态记 `localStorage.home_hero_collapsed`）。下方看板统计卡 `.stat-grid` 通过 `--n` 变量整行等分不落单；教师端「班级掌握情况」全宽横排（`.cls-grid`）。
- **通用列表与状态组件**：区块标题用 `.sec-head`；报告列表、批改列表、最近动态、标定点清单统一用 `.list-panel` > `.list-row`（含 `.lr-icon`、`.lr-main`、`.lr-actions`）与状态胶囊 `.pill-st.draft/.submitted/.graded`（`_statusBadge`）。
- **分析流程 = 5 步向导**：`#wzp1..5` 页 + `goToPage()/canProceed()/updateStepBar()/resetWizard()`；每步一屏、底部固定 上一步/下一步，完成自动翻页。
  - **步骤 1（选择类型）**：4 张横排白底实验卡（`.type-cards`），选中后下方 `#type_info`（`renderTypeInfo`）三栏展示「测什么（反演目标） / 怎么拍（采集要求） / 得到什么（输出清单）」。
  - **步骤 2（视频与参数）**：紧凑横排获取方式卡（`.acq-cards`），已移除「跳过开头」「输出周期数」，仅磁力牛顿摆保留「跳过结尾」；实验获取面板 `#acq_panel_experiment` 采用双列网格 `.acq-exp-grid`（左：STM32 电磁铁/夹爪释放；右：树莓派摄像头 + 二维云台控制），底层脚本命令与 GPIO/脉宽等工程调试说明统一收进 `<details class="dev-notes">`。取景与拍摄按钮在「取景窗」弹窗 `#pi_modal_overlay` 里（与标定窗共用 `resizableModalEl()/fitActiveModal()`）。
  - **步骤 3（标定）**：`#calib_preview` 采用左大图（`560×315` 自适应长宽比 `#thumb_canvas`，`updateThumbCanvas`）+ 右侧标定点位清单 `#calib_points_guide`（`renderCalibPointsGuide`）双栏布局。手动标记弹窗 `#calib_modal_overlay` 全部四类实验统一分为两步：第①步在顶部 `#calib_frame_bar` 选定释放前最后一帧作为计时第一帧 `start_frame`（`t=0`），第②步点选标定点；点击坐标走 `canvasImageCoords()`。扭摆满 3 点后 `drawNiubaiFrame()` 画标定坐标系（青色 Y 轴=θ=0、橙色 X 轴=θ=+90°、黄原点，与后端 `draw_overlay` 同步）。
  - **步骤 4（运行）**：顶部 `#run_config_summary`（`renderRunConfigSummary`）汇总实验类型、视频来源、帧数/帧率、标定状态与关键参数；底部 `#output_files_hint` 列出预期生成文件。
  - **步骤 5（结果页，`renderResult`）**：以**反演运动微分方程主卡 `#res_hero`（`.eq-hero`）+ 物理量指标卡 `#res_metrics`（`.metric-grid`）**为视觉主角；下方 `#res_time_note` 自证时间基准（真实采样率/慢放倍数）；图表区 `#res_charts` 用分段标签 `.seg-tabs` 切换「角度-时间曲线」「方程拟合与相图（`sr_result.plot_preview`）」「标注视频预览（仅扭摆，H.264 内联播放）」；扩展物理项库回归与分段衰减收进 `details#res_model`；文件路径清单收进默认折叠的 `details#res_files`；学生端提供 `#btn_res_to_report`（`writeResultToReport` 一键把反演方程与关键物理量追加到报告草稿并跳转预览），右侧提供「重新分析」（`resetWizard({keepType:true})`）。`window.__renderResultFixture(m, type)` 供截图回归直接渲染 fixture。
- **指导/报告/批改台**：
  - **实验指导**：全宽流程条 `#guide_flowbar` + `.guide-shell` 左栏固定章节目录 `.doc-toc` 与实验档案卡 `.guide-side-card` + 右栏概览横幅 `.guide-hero`（含快捷锚点与阅读进度条 `.doc-progress`）+ 结构化分节网格（`.guide-obj-grid` / `.guide-item-grid` / `.guide-step-grid` / `.guide-param-grid` / `.guide-faq-grid`，数据记录分节内嵌报告数据表结构 `.guide-tpl-tables`）+ 底部三栏「实验预习速览工作台」`.guide-companion` + 翻页器 `.doc-pager`（禁止改回整体长页；学生读至最后一节触发 `_guideLastPageCb` 自动完成预习）。
  - **报告与批改台 = 列表/详情切换**：进入编辑器（`openReportEditor`）或批改详情（`openGrading`）时隐藏上方列表卡（`#report_list_area` / `#grading_list_area`），顶栏提供「‹ 返回报告列表 / ‹ 返回批改列表」。草稿编辑态下每个分节用 `.rep-sec-edit` 包裹 `textarea` 与 `.rep-preview`（≥1300px 左右并排实时预览，**务必保持 `textarea` 与 `.rep-preview` 为直接相邻兄弟节点**，供 `repPreview(ta)` 的 `ta.nextElementSibling` 查找）。
- **AI 面板上下文隔离与随境引导**：`currentCtxExp()` 按当前激活视图隔离实验 ID（仅 `analysis`/`guide`/`reports` 返回对应实验，`dashboard`/`qas` 返回空，杜绝跨视图串味）；`updateAICtx()` 末尾调用 `renderAIPresets()` 按当前视图/步骤切换 `.preset-questions` 与初始欢迎语（已有对话时不覆盖）；未登录时禁用输入框；分析结果页下 `aiContext()` 附带 `result_summary` 供 `ai_agent.py build_system_prompt()` 注入。
- **图标与公式**：只用 SVG symbol 体系（`#i-*`），线宽 1.7；`static/katex` 本地部署，正文字段 `$...$` / `$$...$$` 由 `renderKaTeXinHTML` + `whenKatex` 渲染。

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
13. **标注视频要浏览器能播才有预览**：浏览器不认 cv2 的 mp4v，结果页内联播放必须是 H.264（`video_playable` 才给 `<video>`，否则只提示）；写盘降级链集中在 `processors/common.py` `H264VideoWriter`（`niubai_processor.py` 中以 `_AnnotatedVideoWriter` 别名导入）：ffmpeg 管道 → Windows MSMF/avc1（**只认 ASCII 路径**，中文结果目录要先写临时文件再搬回）→ cv2 H.264 → mp4v 兜底。另外 OpenCV 的 FFmpeg 后端缺 libopenh264 时 `isOpened()` 仍返回 True 却一帧都写不进去，所以选后端前要用临时文件"试写几帧再读回"探针（`_probe_cv_h264`）。

## 约定

- 提交信息用中文、一句话主题 + 分条要点。
- 不要动 `firmware/`（Keil 工程）与 `models/*.pt`。
- 旧的 `index_html_clean.html` / `index_html_full.txt` 是历史备份，未入库，别参考也别删（等用户处理）。
- **前端改版（已完成）**：规格与验收清单见 `docs/frontend_redesign.md`（任务包 T0–T9 已全部落地并验收）；截图回归用 `python tools/ui_shots.py`（产物在 `tools/_shots/`，已 gitignore）。
