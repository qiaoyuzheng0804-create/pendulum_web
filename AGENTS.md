# AGENTS.md — 大学物理实验平台

面向 AI 编码助手的项目指南（ZCode 工作区指令文件，本文件为唯一权威版本）。

## 项目是什么

大学生物理实验平台（由"单摆运动视频分析器"升级而来）。核心能力：

1. **视频分析实验**（4 种摆：单摆/磁阻尼摆/扭摆/磁力牛顿摆）：上传或 OpenMV 采集视频 → YOLOv8 逐帧检测 → 提取角度-时间数据 → 拟合阻尼参数，输出 CSV/图/标注视频。**四种摆都有完整实验指导**（`teaching/guide_<id>.json`），入口：工作台"去预习"、分析向导第 1 步"实验指导"按钮、指导页流程条"打开视频分析"。
2. **指导型实验**（三线摆、杨氏模量、牛顿环、迈克尔逊、示波器、落球法粘度等）：实验指导 + 专属报告模板（含数据表格）+ AI 辅导，无视频分析。
3. **实验课全流程闭环**：课前预习（学生点"去预习"跳到该实验的实验指导页，**通读至最后一页自动完成预习标记**）→ 课中实验（指导型实验手动点"完成课中"；**视频分析实验以跑完分析、拿到实验数据为完成标准，出结果自动标记**，流程条不提供手动按钮）→ 课后报告（按模板撰写提交，提交后可**撤销提交**回到草稿）→ 教师批改打分/退回 → 学生查看反馈。`progress` 表记录 预习/课中；`submissions` 记录 报告/批改。
4. **师生问答（公共讨论区）**：所有登录用户可见全部提问与回复；学生提问（文字 / 图片 / 语音附件），教师和全体学生均可回复，支持多轮；提问者/回复者可**撤回删除**自己的提问/回复（撤回提问连同回复与附件一并删除）。`scope=mine` 可只看自己的提问。
5. **工作台数据看板（默认首页视图）**：教师看 班级维度成绩/报告进度/待批改/成绩分布/各实验报告量/待回复提问；学生看 个人流程进度与统计；未登录看公共概览。在线人数来自看门狗 `_clients`（`/api/dashboard/online`）。
6. **AI 实验助手**：右侧常驻面板，自研轻量 agent（工具调用循环），随上下文全流程辅导。

三栏类 IDE 布局：左导航（工作台/实验列表/师生互动/报告中心）+ 中央区 + 右 AI 面板。
（"学习中心"分组与"理论知识"视图已整体删除，勿再引用 `view_theory`/`loadTheory`/`renderTheory`。）

## 运行

- 启动：`python app.py`（或 `启动服务器.bat`），端口 5000。
- 依赖：见 requirements.txt（flask/opencv/ultralytics/torch/openai/pyserial 等，模型权重在 `models/*_best.pt`，缺失时 YOLO 自动降级）。
- LLM 配置两处：**网页「模型配置」**（教师登录 → AI 面板顶部齿轮，存 `llm_config.json`，即时生效）优先于 `.env`（`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`，任意 OpenAI 兼容服务，作后备）。API：`GET|POST /api/llm_config`（teacher_required）、`POST /api/llm_config/test`；合并逻辑在 `app.py get_llm_config()`（网页字段逐项优先，留空回退 .env）。
- **看门狗**：所有页面关闭后约 10s 自动关停服务器（`server.pid` + `/api/heartbeat`）。调试时若进程"自己消失"是正常行为，不是 bug。
- 演示账号：`teacher/123456`（教师）、`student/123456`（学生）、`s240101/123456`（物电 2401，看板演示班级学生，共 3 班 12 人自动预置；删 platform.db 即重置）。
- 删除交互模拟：该视图已移除（nav/view/CSS/JS 与 ai_agent 映射均清理），勿再引用 `sim`。

## 架构与关键文件

| 文件 | 职责 |
|---|---|
| `app.py` | 全部视频分析/串口/OpenMV 路由 + 页面入口；YOLO 懒加载；全局处理锁 |
| `experiments.py` | **实验注册表（单一数据源）**。新增实验 = 加一条记录 + `teaching/guide_<id>.json` + `report_templates/<id>.json`，不改 app.py |
| `platform_hub.py` | Blueprint：登录(session cookie)/用户(含班级)/报告提交批改退回/预习课中进度/师生问答+附件上传/数据看板统计。SQLite(`platform.db`)。**不能改名为 platform.py**——会遮蔽标准库 platform，cv2/torch 会炸 |
| `demo_reports.py` | 演示报告完整内容（6 分节 + 表格数据，公式 `$...$` 包裹）。`_seed_demo_if_empty()` 与 `_upgrade_demo_reports()`（按旧标记识别并升级旧演示报告，幂等）共用 |
| `ai_agent.py` | AI 工具调用循环：get_experiment_guide / get_report_template / analyze_numeric_data；`build_system_prompt` 注入实验/步骤/角色上下文 |
| `processors/` | 每种摆一个处理器（`_run_processing` 的 if/elif 分派）+ openmv_manager + symbolic_regression |
| `teaching/*.json` | 教学内容（guide_<id>.json；theory.json 尚存但前端已无入口），API 下发，前端渲染 |
| `report_templates/*.json` | 报告模板：每实验 6 个分节（purpose/principle/apparatus/data/analysis/conclusion）+ 分节内可选 `tables`（结构化数据表格，见下） |
| `templates/index.html` | **单文件前端**（约 3400 行，含全部 CSS/JS），无构建步骤 |
| `templates/login.html` | 登录页（独立小页） |

## 平台 API 速查（platform_hub.py / app.py）

- 认证：`POST /api/auth/login|logout`、`GET /api/auth/me`（含 class_name）
- 报告：`GET|POST /api/reports`（教师支持 `?exp=&status=&class=` 筛选）、`GET /api/reports/<id>`、`POST .../submit`、`POST .../<id>/withdraw`（学生撤销提交：submitted→draft）、`DELETE /api/reports/<id>`（学生删除草稿；已提交需先撤销，已批改不可删）、`POST .../<id>/grade|return`、`GET /api/report_template/<exp>`
- 全流程：`POST /api/progress/pre|lab`（学生标记预习/课中）、`GET /api/flow`（学生=10 实验状态；教师=`?class=` 全班聚合）
- 问答：`GET /api/qas`（公共讨论区：默认 scope=all 全员可见；scope=mine 只看自己的；教师可加 `&pending=1` 筛待回复）、`POST /api/qas`（学生提问，attachments=[附件id]）、`GET /api/qas/<id>`（全员可看）、`DELETE /api/qas/<id>`（撤回提问，仅本人，连回复附件一起删）、`POST /api/qas/<id>/reply`（教师和全体学生均可回复）、`DELETE /api/qas/reply/<rid>`（撤回回复，仅本人）
- 附件：`POST /api/qas/attachment`（multipart file，图片/音频/文档白名单）、`GET /api/qas/attachment/<id>/file`（绑定到问答线程后登录用户均可见）
- 看板：`GET /api/dashboard/summary`（未登录/教师/学生三种视图；教师 `class_stats` 含服务端算好的 `grade_rate`——**批改率 = 已批改报告 / 已提交报告，同一学生×同一实验只按最新一份计入**，前端只展示不重算）、`GET /api/dashboard/online`（在线人数=心跳客户端数，app.py）
- 用户/班级：`GET|POST /api/admin/users`（导入行格式：学号,姓名,班级）、`GET /api/classes`（教师）

演示数据：`_seed_demo_if_empty()` 在 users 无带班级学生时注入 3 班 12 学生 + 24 份**完整内容**的演示报告（含数据表格与公式）+ 演示进度/问答（幂等，删除 platform.db 重置；旧库中的占位演示报告由 `_upgrade_demo_reports()` 按 `（演示数据）` 标记自动升级）。

## 前端设计规范（务必遵守，勿回退）

- **主题**：「实验手册」暖纸墨色 —— 暖白纸底 `#f5f3ef` + 近黑墨 `#1c1917` + 焦橙强调 `#c2410c` 单一强调色；发丝线分隔、圆角刻度 10/8/6/4px、阴影极轻。所有颜色走 `:root` CSS 变量，改主题只改变量。
- **禁止**：靛紫/渐变文字/霓虹辉光/emoji 当图标/长页下滑式布局/纯黑投影。
- **布局**：三栏 shell（`.shell` 网格，`grid-template-rows:minmax(0,1fr)` 锁行高——不锁则中栏长内容把侧栏/AI 面板撑出视口，此 bug 修过一次）。
- **分析流程 = 5 步向导**：`#wzp1..5` 页 + `goToPage()/canProceed()/updateStepBar()`；每步一屏、底部固定 上一步/下一步，完成自动翻页。步骤：1 选择类型 → 2 视频与参数 → 3 标定 → 4 运行 → 5 结果。
- **指导/报告 = 分节翻页**：`renderGuide` 输出 `.doc-sec` 节 + `initDocPager` 页码器（‹ n/N ›），禁止改回整体长页。（理论视图已删除，`renderTheoryMarkdown` 名字虽带 theory 但仍被指导页/批改视图共用，勿删。）
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
5. 同步处理有全局锁（一次只处理一个视频）；OpenMV 录制期间看门狗不关停。
6. **marked v12 会吃掉 `$...$` 内的 `\,` `\;` `\{` `\}`**（如 `0.087\,\mathrm{rad}` 变 `0.087,rad`）。`renderTheoryMarkdown`/`renderChatMarkdown` 已用 `protectMathSpans/restoreMathSpans` 占位符把公式摘出再过 marked，**勿删这套保护**，也别绕过它直接 `marked.parse`。
7. **teaching/*.json 里反斜杠必须双写**：JSON 中单写 `\frac`、`\right` 会被解码成换页符/回车符 + 残字（三线摆指导踩过，公式变成 `rac{...}` 乱码）。写完用 `json.load` + 扫控制字符自检。公式统一 `$`/`$$` 包裹，KaTeX 不认的字符（≪、∝、℃、全角逗号）在公式内要写成 `\ll`、`\propto`、`^{\circ}` 等；行文里的 Unicode 记号（θ²、T_d 等）是刻意的纯文本风格，别批量改写。
8. **platform_hub.py 写库必须放在 `with _conn() as c:` 块内**：sqlite3 的 `with` 负责提交，块外执行的 UPDATE 既不提交、还会一直持有写锁，把后续所有写请求打成 `database is locked`（迁移 `_upgrade_demo_reports` 踩过）。
9. **已提交/已批改的报告前端只读渲染**（`openReportEditor` 按 `status` 分支：draft 走 textarea 编辑器 + `.rep-preview` 实时预览框（`repPreview()` 输入防抖渲染 Markdown+KaTeX，避免只见 TeX 源码），其余走 `repTableHtml(...,false)` + `renderChatMarkdown` 只读视图 + 可选"撤销提交"按钮）；批改视图、学生只读视图与草稿预览的 `.theory-note` 都要在 `whenKatex` 里补渲染一次，否则 KaTeX 未就绪时公式闪现 `$...$` 源码。

## 约定

- 提交信息用中文、一句话主题 + 分条要点。
- 不要动 `firmware/`（Keil 工程）与 `models/*.pt`。
- 旧的 `index_html_clean.html` / `index_html_full.txt` 是历史备份，未入库，别参考也别删（等用户处理）。
