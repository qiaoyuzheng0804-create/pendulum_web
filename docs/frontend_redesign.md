# 前端改版设计文档（v1）

> 面向接手实现的 AI / 开发者。本文是**可直接执行的规格**：每个任务包都写明了 目标 → 涉及代码 → 具体改动 → 验收标准。
> 开工前必须先读 `AGENTS.md`（尤其「前端设计规范」「已踩过的坑」），本文不重复其中内容，**与 AGENTS.md 冲突时以 AGENTS.md 为准**，并在本文「待决事项」里记一笔。
> 代码位置以**函数名 / 元素 id** 为准，行号仅供参考（会漂移）。

---

## 0. 背景与目标

本作品参加全国大学生物理实验竞赛。

- **主线（核心创新）**：单摆 / 磁阻尼摆 / 扭摆 / 磁力牛顿摆 —— 视觉识别（YOLOv8）+ 标定 + 角度序列 + 降噪 + **方程反演**（解析解最小二乘 + ODE 复现验证 + 扩展物理项库回归）。
- **拓展（教学价值）**：6 个指导型实验、报告模板与批改、师生问答、AI 助手、教学看板。

**现状问题**（截图诊断见附录 A）：界面讲的是"教务系统"，主线技术没露脸；结果页把反演出的方程藏在折叠区；向导页工程调试信息外露；卡片层层嵌套、中英双语标签冗余、大面积留白；AI 上下文徽章串味。

**改版目标**

1. 打开首页 10 秒内让评委明白"这个作品能从视频里反演出运动方程"。
2. 结果页以**方程 + 物理量**为主角。
3. 主线 / 拓展在导航与首页上主次分明。
4. 视觉更干净：少灰块、少嵌套、去冗余英文、无大片空白。
5. 不破坏任何现有功能与 AGENTS.md 中的约束。

**不做的事**

- 不换主题（保持「实验手册」暖纸墨色，`:root` 变量体系不变，单一强调色 `--a1`）。
- 不引入构建工具 / 前端框架 / 图表库；仍是单文件 `templates/index.html`。
- 不改 5 步向导、分节翻页这两个结构；不恢复 OpenMV 界面、理论视图、交互模拟。
- 不改任何后端 API 的既有字段语义（只**新增**字段，见 §8）。

---

## 1. 实施顺序与任务包总览

按收益与依赖排序。**每个任务包独立可交付、可单独提交**（提交信息中文，一句话主题 + 分条要点）。

| 包 | 名称 | 依赖 | 风险 |
|---|---|---|---|
| T0 | 基础：设计令牌 + 通用组件 + 截图工具 | — | 低 |
| T1 | AI 上下文徽章串味修复（bug） | — | 低 |
| T2 | 结果页重做（P0） | T0；后端 B1 | 中 |
| T3 | 导航信息架构（P1） | T0 | 中 |
| T4 | 首页 Hero + 看板重排（P1） | T0、T3 | 中 |
| T5 | 向导减负（P2） | T0 | 中（ID 不能动） |
| T6 | 指导页 / 报告 / 批改台视觉清理（P3） | T0 | 低~中 |
| T7 | 全局去英文副标签 + 卡片去嵌套（P3） | T0 | 低 |
| T8 | AI 面板上下文化（P4） | T1 | 低 |
| T9 | 收尾：AGENTS.md 同步 + 全量截图回归 | 全部 | — |

进度请在 §11 的清单里勾选。

---

## 2. T0 基础：设计令牌、通用组件、截图工具

### 2.1 新增 CSS 变量（追加到 `:root`，不改已有变量值）

```css
/* 字号刻度 */
--fs-xs:.72rem; --fs-sm:.8rem; --fs-base:.875rem; --fs-md:1rem; --fs-lg:1.15rem; --fs-xl:1.5rem; --fs-2xl:2rem;
/* 间距刻度 */
--sp-1:4px; --sp-2:8px; --sp-3:12px; --sp-4:16px; --sp-5:20px; --sp-6:24px; --sp-8:32px;
/* 圆角（与 AGENTS 规定的 10/8/6/4 对应） */
--r-lg:10px; --r-md:8px; --r-sm:6px; --r-xs:4px;
/* 等宽数字字体 */
--mono:'Consolas','JetBrains Mono',ui-monospace,monospace;
/* 状态胶囊底色（暖色系内） */
--st-draft-bg:#efece6; --st-draft-fg:var(--t2);
--st-sub-bg:rgba(194,65,12,.10); --st-sub-fg:var(--a1);
--st-ok-bg:rgba(21,128,61,.10); --st-ok-fg:var(--green);
```

新写的样式一律用这些令牌；旧样式在各任务包触及时顺手迁移，不必一次全改。

### 2.2 通用组件（新增 CSS 类，后续任务包复用）

| 类名 | 用途 | 规格 |
|---|---|---|
| `.sec-head` | 区块标题（替代 `card-header` + 英文 tag 的组合） | flex；`h2` 用 `--fs-md` 700；右侧放操作按钮；下边距 `--sp-4`；**不放英文** |
| `.list-panel` / `.list-row` | 列表（报告、批改、最近动态） | 外层白底 + 1px `--border` + `--r-lg`；行之间用发丝线分隔，行本身**无底色**；行内边距 `10px 14px`；hover 底色 `--g3`；可点击行 `cursor:pointer` |
| `.lr-icon` | 列表行左侧方块 | 28×28，`--r-sm`，底 `--g1`，字色 `--a1`，字 `--fs-sm` 700 |
| `.lr-main` `.lr-title` `.lr-meta` | 行主体 | title `--fs-base` 600；meta `--fs-xs` `--t3` |
| `.pill-st` + `.draft/.submitted/.graded` | 状态胶囊 | `--fs-xs` 600，`padding:2px 10px`，`border-radius:999px`，底色/字色用 §2.1 的 `--st-*` |
| `.metric-grid` / `.metric` | 物理量指标卡 | 见 §4.3 |
| `.seg-tabs` | 分段标签（结果页图表切换、首页切换） | 横排按钮，选中项底 `--text` 字白，其余描边；`--r-md` |
| `.dev-notes` | 设备调试信息折叠区 | `<details>`；summary `--fs-sm` `--t3`，带 `#i-chev-d` 图标，展开后旋转 |
| `.empty-state` | 空状态 | 居中，图标 28px `--t3` + 一行说明 + 可选按钮 |

### 2.3 截图回归工具

已提供 `tools/ui_shots.py`（无头 Chrome + CDP，**不需要 Playwright**）。用法：

```powershell
python app.py            # 先启动服务（另开窗口）
python tools/ui_shots.py # 产出 tools/_shots/*.png（已 gitignore）
python tools/ui_shots.py --only result   # 仅截结果页（需 §4.6 的 fixture）
```

脚本会以未登录 / teacher / student 三种身份截取：首页、向导 1-3 步、指导页（视频实验与指导实验各一）、报告、问答，以及注入 fixture 后的结果页。**每个任务包完成后都要跑一次，打开图片目视核对**（AGENTS 坑 4：别只看代码）。

### 2.4 验收

- [ ] `:root` 新变量存在，旧变量值未变。
- [ ] 通用组件类定义集中放在 `<style>` 中一个新的注释块 `/* ===== 通用组件（改版 v1） ===== */` 里。
- [ ] `tools/ui_shots.py` 能跑通并产出截图。

---

## 3. T1 修复：AI 上下文徽章串味

**现象**：学生打开「实验报告」时徽章显示"扭摆 · 步骤3 标定 · 报告"；在杨氏模量指导页仍带着扭摆状态。
**原因**：`updateAICtx()` 与 `aiContext()` 都用 `S.motionType || window._currentGuideExp`，不看当前视图。

**改动**：新增一个函数 `currentCtxExp()`，供两处共用：

```js
// 按当前视图决定"正在讨论哪个实验"，避免跨视图串味
function currentCtxExp(){
  const v=currentViewName();
  if(v==='analysis')return S.motionType||'';
  if(v==='guide')return window._currentGuideExp||'';
  if(v==='reports')return (window._reportEditor&&_reportEditor.exp_id)||'';
  return '';
}
```

- `updateAICtx()`：`exp` 改用 `currentCtxExp()`；只有 `view==='analysis'` 时才拼"步骤 N"；`reports` 且编辑器打开时显示"<实验名> · 报告"，否则只显示"实验报告"。
- `aiContext()`：`exp_id` 同样改用 `currentCtxExp()`；`step` / `step_name` / `has_video` / `calib_points` 仅在 `view==='analysis'` 时填，否则置空。
- `_reportEditor` 是 `let` 声明，函数内懒引用安全（AGENTS 坑 2）；离开报告视图时**不必**清空它，因为判断依据是视图。

**验收**：分析页选扭摆走到第 3 步 → 切到报告（不打开编辑器）徽章只显示"实验报告"→ 打开杨氏模量报告显示"杨氏模量测量 · 报告"→ 切到指导页选三线摆显示"三线摆… · 实验指导"。

---

## 4. T2 结果页重做（P0，最高优先级）

### 4.1 先重构：把结果渲染抽成独立函数

`startProcessing()` 中 `msg.type==='result'` 分支里那一大段拼 HTML 的代码，抽出为：

```js
function renderResult(r, outputDir){ /* 渲染整个第 5 步 */ }
```

`startProcessing` 只负责：进度收尾 → `goToPage(5)` → `markLab` → `window._lastResult={exp:S.motionType,r,outputDir:msg.output_dir,at:Date.now()}` → `renderResult(r, msg.output_dir)`。
这样第 5 步能用 fixture 单独渲染与截图（§4.6）。

### 4.2 页面结构（替换 `#wzp5 > #card-result` 的内部）

```
#card-result
├─ .sec-head      「分析结果 · 扭摆」  [阻尼类型胶囊]            [写入报告] [重新分析]
├─ #res_hero      方程主卡（.eq-hero）
├─ #res_metrics   物理量指标卡（.metric-grid）
├─ #res_charts    图表区：.seg-tabs [角度-时间] [拟合与相图] [标注视频]
│                 └ 面板：#plot_preview_area / #sr_plot_area / #video_preview_area
├─ details#res_model   模型细节（扩展物理项库方程、分段衰减等）
└─ details#res_files   输出文件（原 result-grid 的路径条目）
```

**保留的 id**（别改名，其他代码在用）：`plot_preview_area`、`plot_preview_img`、`result_grid`（挪进 `#res_files` 内部继续用）、`video_preview_area`、`video_preview_player`、`video_preview_hint`。`renderResultVideoPreview()` / `hideResultVideoPreview()` 的逻辑不变，只是容器换到标签页里。

原底部的「处理新视频」按钮（`location.reload()`）删除，由 sec-head 上的「重新分析」替代（§4.5）。

### 4.3 方程主卡 `.eq-hero`

样式：白底，左侧 3px `--a1` 竖线，`--r-lg`，内边距 `20px 24px`。从上到下：

1. 小标题 `--fs-xs` `--t3`：「从视频反演得到的运动方程」
2. **数值方程**（KaTeX display 模式，字号约 1.35em，用 `katex.render(tex, el, {displayMode:true, throwOnError:false})`）
3. 通式（KaTeX 行内，`--t3`，`--fs-sm`），让人看出数值方程从哪来
4. 拟合优度一行 `--fs-sm`：`时域拟合 R² = 0.9987 · ODE 复现 R² = 0.9981 · RMSE = 0.21°`
5. 方法胶囊（`--fs-xs` 描边胶囊）：`sr.final_source` 原文

**各实验的 TeX**（`fmt()` 见 §4.7；θ 一律用度显示的平衡角，系数保留 4 位有效数字）：

| 实验 | 条件 | 数值方程 | 通式 |
|---|---|---|---|
| cizuni / niubai | 有 `omega0`、`beta` | `\ddot\theta + {2β}\,\dot\theta + {ω0²}\,(\theta - {θeq}^\circ) = 0` | `\ddot\theta + 2\beta\dot\theta + \omega_0^2(\theta-\theta_{eq}) = 0` |
| 同上，补一行解的形式 | 欠阻尼（`omega_d>0`） | `\theta(t) = {θeq}^\circ + e^{-{β}t}\,(A\cos {ωd}t + B\sin {ωd}t)` | — |
| 同上 | 过阻尼 | `\theta(t) = \theta_{eq} + C_1e^{-\lambda_1 t} + C_2e^{-\lambda_2 t}`（只显示形式） | — |
| 同上 | 临界阻尼 | `\theta(t) = \theta_{eq} + (A+Bt)e^{-\lambda t}`（只显示形式） | — |
| danbai | 有 `beta` | 振幅包络 `\theta_A(t) = {A0}^\circ\, e^{-{β}t}`（A0 需后端 B2 提供；没有时写 `\theta_A(t) = A_0 e^{-{β}t}`） | `\ddot\theta + 2\beta\dot\theta + \omega_0^2\theta = 0` |
| ciliniudun | 无 `sr_result` | **不显示方程**；主卡改为"多摆角度时序已提取"：显示 摆数 `num_pendulums`、采样点 `num_points`、步长 `dt`、时长 = `num_points×dt`，配一句说明"碰撞与能量传递分析请在报告表格中完成" | — |
| 任意 | `sr_result.error` 存在 | 主卡改为警示态（左竖线 `--amber`）：「方程拟合未成功：{error}」+ 建议（"检查标定点 / 裁剪起止时间后重新分析"）；图表区照常显示 | — |

> [!IMPORTANT]
> 单摆当前 `run_sr_danbai` 的 `omega0` 是按 **L=0.5 m 假设算的**（`omega0 = sqrt(9.8/0.5)`），不是测出来的，`zeta` 也随之是假设值。在后端 B2 落地前，前端**不得**把单摆的 ω₀、ζ 当测量值展示：指标卡里这两项加注"按 L=0.5 m 估算"，或干脆不显示。答辩时这是硬伤。

### 4.4 物理量指标卡 `.metric-grid`

```css
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0 18px;}
.metric{background:var(--card);border:1px solid var(--border);border-radius:var(--r-md);padding:12px 14px;}
.metric .m-sym{font-size:var(--fs-sm);color:var(--t3);}          /* KaTeX 符号 + 中文名 */
.metric .m-val{font-family:var(--mono);font-size:var(--fs-xl);font-weight:700;font-variant-numeric:tabular-nums;color:var(--text);}
.metric .m-unit{font-size:var(--fs-sm);color:var(--t3);margin-left:4px;}
.metric .m-note{font-size:var(--fs-xs);color:var(--t3);margin-top:2px;}
.metric.key{border-color:var(--a1);} /* 最重要的 1~2 项 */
```

**每个实验展示哪些**（按顺序；字段缺失就跳过该卡）：

| 实验 | 指标（sym · 名 · 来源 · 单位） |
|---|---|
| niubai | **T 周期** `r.period_s`（没有则 `2π/sr.omega_d`）s · **β 衰减系数** `sr.beta` s⁻¹ · ω₀ 固有角频率 `sr.omega0` rad/s · ζ 阻尼比 `sr.zeta` · θ_eq 平衡角 `sr.theta_eq_deg` ° · R² `sr.r2_fit` |
| cizuni | **ζ 阻尼比**（`key`）· β · ω₀ · ω_d `sr.omega_d` · θ_eq · R² |
| danbai | **T 周期**（后端 B2 的 `sr.period_s`；没有就不显示）· **β** · R² `sr.r2` · （B2 前 ω₀/ζ 不显示或带注） |
| ciliniudun | 摆数 · 采样点数 · 采样步长 `r.dt` s · 记录时长 s |

附加信息：cizuni 的结果里如果带 `slow_motion_factor`/`slow_motion_source`/`real_fps`，在指标卡下方用一行 `--fs-xs` 灰字展示"真实采样率 240 fps（慢放 ×8，来源：capture.fps 标签）"——这是时间基准的自证，评委会问。

### 4.5 顶部操作

- **写入报告**（`btn-primary`，仅学生登录可见；教师 / 未登录隐藏）：
  1. 调 `goReportFor(S.motionType)` 打开（或新建）该实验草稿；
  2. 草稿编辑器渲染完后，往 `#sec_analysis` 这个 textarea **末尾追加**一段 Markdown（不覆盖已有内容；如果已经包含同一次结果的标记 `<!-- res:<at> -->` 就不重复追加）；
  3. 触发一次 `repPreview(textarea)` 刷新预览；**不自动保存**，toast 提示"已填入分析结果，请检查后保存"。
  4. 等待编辑器渲染：`goReportFor` 内部是异步的，给 `openReportEditor` 增加可选的第三个参数 `onReady` 回调（渲染完成后调用），别靠 setTimeout 猜。

  追加内容模板（公式用 `$...$`，反斜杠在 JS 字符串里要双写）：

  ```md
  <!-- res:1730966400000 -->
  **视频分析反演结果**（2026-11-07 15:20，平台自动填入）

  - 运动方程：$\ddot\theta + 0.02461\,\dot\theta + 6.712\,(\theta-0.12^\circ) = 0$
  - 周期 $T = 2.425\ \mathrm{s}$，衰减系数 $\beta = 0.01230\ \mathrm{s^{-1}}$，阻尼比 $\zeta = 0.004748$
  - 拟合优度：时域 $R^2 = 0.9987$，ODE 复现 $R^2 = 0.9981$
  ```

- **重新分析**（`btn-outline`）：新函数 `resetWizard({keepType:true})`，替代 `location.reload()`：
  - 清 `S` 中与本次视频相关的字段：`sessionId, firstFrameB64, firstFrameImg, calibPoints=[], calibPhase, _pivotCount, frameIdx, calibFrameImg, totalFrames`，以及 `S` 里其他视频 / 标定相关字段（**实现前先通读 `const S={...}` 定义和 `selectType()` 里的重置逻辑，照着补全，别漏**）；
  - `hideResultVideoPreview()`；清空 `#result_grid`、`#plot_preview_img.src`、`#video_info`、`#status`；`clearVideoPreview()`；`resetCalib()` 的界面部分；
  - 树莓派连接**保持**（不断开）；
  - `goToPage(2)`；不清空 AI 对话。

### 4.6 图表区

`.seg-tabs` 三个标签，默认选中第一个可用的标签：

| 标签 | 内容 | 显示条件 |
|---|---|---|
| 角度-时间 | `#plot_preview_area`（`r.plot_preview`） | 有 `plot_preview` |
| 拟合与相图 | 新增 `#sr_plot_area > img#sr_plot_img`，源为 `sr.plot_preview`（后端 B1 新增） | 有 `sr.plot_preview` |
| 标注视频 | `#video_preview_area`（逻辑仍走 `renderResultVideoPreview`，只有扭摆有） | `S.motionType==='niubai' && r.video_path` |

图片点击仍是 `openLightbox`。去掉原来 img 上内联的 `onmouseover` 缩放。

**Fixture（用于截图和自测）**：新建 `tools/fixtures/result_niubai.json`，结构与 SSE 里 `msg` 一致（`{output_dir, result:{...}}`）。最好真跑一次扭摆分析，把 `msg` 原样存下来（plot_preview 是 base64，体积可以接受）；没条件时可以手写一份不带图的最小样例：

```json
{"output_dir":"D:\\实验结果\\demo","result":{
 "csv_path":"D:\\实验结果\\demo\\angle_time_data.csv","plot_path":"D:\\实验结果\\demo\\angle_time_plot.png",
 "video_path":null,"num_points":1843,"period_s":2.4251,"period_peaks_s":2.4248,
 "sr_result":{"beta":0.0123,"omega0":2.5907,"zeta":0.004748,"omega_d":2.59067,"theta_eq_deg":0.12,
  "damping_type":"欠阻尼","ode_r2":0.9981,"ode_rmse_deg":0.21,"r2_fit":0.9987,
  "library_coef":{"a_q":-6.712,"b_v":-0.0246,"c_q3":0.31,"d_vabs":-0.002,"e_const":0.0001,"accel_r2":0.962},
  "final_source":"欠阻尼解析解时域拟合 + ODE验证"}}}
```

页面上暴露一个**仅调试用**的入口（接收已解析的对象）：

```js
window.__renderResultFixture=function(m,type){
  if(type)selectType(type);           // 让 S.motionType 与界面一致
  goToPage(5);renderResult(m.result,m.output_dir);
};
```

fixture 文件通过 Flask 访问不到，`tools/ui_shots.py --only result` 会读取 `tools/fixtures/result_*.json`，用 `Runtime.evaluate` 调 `__renderResultFixture(<json>, '<type>')`（文件名 `result_<type>.json` 即类型）。

### 4.7 数字格式

```js
// 有效数字格式化：极小/极大值才用科学计数，避免满屏 1.230000e-2
function fmt(v,sig=4){
  if(v==null||!isFinite(v))return '—';
  const a=Math.abs(v);
  if(a!==0&&(a<1e-3||a>=1e5))return Number(v).toExponential(sig-1).replace('e','×10^{')+'}'; // TeX 内用
  return String(Number(Number(v).toPrecision(sig)));
}
```

TeX 场景用上面的形式；纯文本场景另写 `fmtText()`，输出 `1.23×10⁻⁴`，或者直接用 `toPrecision`。

### 4.8 模型细节 `details#res_model`

summary：「模型细节：扩展物理项库回归」。内容：

- KaTeX：`\ddot\theta = a\,q + b\,\dot\theta + c\,q^3 + d\,\dot\theta|\dot\theta| + e,\quad q=\theta-\theta_{eq}`
- 系数表（`.rep-tbl` 样式复用）：a / b / c / d / e 和加速度 R²。
- 一句自动解读（纯前端规则）：设 `lin = |a|`，若 `|c|·(典型q)²/|a| < 5%` 且 `|d|` 相对 `|b|` 很小，就输出"非线性项与平方阻尼项贡献很小，线性阻尼模型足以描述该运动"；否则输出"存在可观的非线性 / 平方阻尼贡献，建议在报告中讨论"。典型 q 取振幅估计 `r.amp_denoised_deg`（扭摆），没有时取 10°，换算成弧度。
- 扭摆：`r.decay_blocks` 非空时附一张"分段衰减"小表（每段的时间区间与 β）。

### 4.9 输出文件 `details#res_files`

summary：「输出文件（N 个）」，默认折叠。内部就是原来的 `#result_grid`，条目改成单行紧凑样式：`键名（固定宽度 96px，--t3）| 路径（等宽字体，单行省略号，title 放全路径）| 复制按钮`。复制按钮必须走 `copyBtnHtml()`（AGENTS 坑 10）。

### 4.10 验收

- [ ] 用 fixture 渲染的扭摆结果页：一屏内（1600×1000）能看到 方程主卡 + 指标卡 + 图表标签。
- [ ] 方程是 KaTeX 渲染，不出现 `$` 源码，也不出现 `θ̈` 这类 Unicode 拼的方程。
- [ ] ciliniudun / sr 失败 两种状态各自显示正确（各手写一份 fixture 验证）。
- [ ] 学生点「写入报告」后进入对应草稿，`analysis` 分节末尾出现结果段，预览框里公式渲染正常，未自动保存。
- [ ] 「重新分析」回到第 2 步、页面不刷新，AI 对话还在；再上传视频能完整跑通。
- [ ] 换实验类型、重新分析时旧的视频预览都被收掉（AGENTS 前端规范）。

---

## 5. T3 导航信息架构（P1）

### 5.1 新结构（`aside#sidenav`）

```
┌ 品牌块 .nav-brand：SVG 标识（复用 favicon 的摆锤图形）+ BRAND_NAME（--fs-base 800）+ BRAND_SUB（--fs-xs --t3）
├ 首页                         (#nav_dashboard，文字由"工作台"改为"首页"，id 不变)
├ 智能分析实验   data-g="video"  组副标："视觉识别 · 方程反演"（组标题下 --fs-xs --t3）
│   单摆 / 磁阻尼摆 / 扭摆 / 磁力牛顿摆   各用专属图标 #i-exp-<id>
├ 拓展教学实验   data-g="guide"
│   6 个指导型实验  图标 #i-book-open
├ 教学协作       data-g="collab"
│   师生问答 (#nav_qas，徽标 #qa_badge 保留)
│   实验报告 (#nav_reports；教师登录后文字改为"批改台")
└ footer 用户区（不变）
```

- 删掉原来的 `qa`、`report` 两个单子项分组，合并成 `collab`。localStorage 里旧的 `navg_qa` / `navg_report` 键留着无害，不用清理。
- `renderNav()`：去掉 `.nav-badge`（"视频 / 指导"小字）。视频实验项的图标改成 `#i-exp-${e.id}`。
- 分组标题 `.nav-group-label`：去掉 `text-transform:uppercase`（中文没意义），改为 `--fs-xs` 600 `--t3` `letter-spacing:.02em`。
- `refreshAuth()` 里按角色设置 `#nav_reports .txt` 的文字（学生"我的报告" / 教师"批改台" / 未登录"实验报告"），`VIEW_TITLES.reports` 同步。

### 5.2 品牌常量

在 `<script>` 顶部附近声明（用 `var`，AGENTS 坑 2）：

```js
var BRAND_NAME='大学物理实验平台';   // 待决事项 D1：等用户给正式作品名
var BRAND_SUB='视觉识别 · 方程反演 · 实验教学';
```

`<title>`、`.nav-brand`、首页 Hero、`templates/login.html` 的标题统一用它（login.html 是独立页面，直接改文字，并加注释提醒两处同步）。

### 5.3 四个实验专属图标（新增 SVG symbol，线宽 1.7，24×24 viewBox，`fill="none" stroke="currentColor"`）

- `i-exp-danbai`：顶部横线 + 斜线摆绳 + 末端实心小圆
- `i-exp-cizuni`：摆绳 + 末端方块 + 下方两道 U 形磁铁弧线
- `i-exp-niubai`：竖直细丝 + 水平圆盘（椭圆）+ 圆盘旁一段弧形箭头
- `i-exp-ciliniudun`：顶部横梁 + 3 根并排吊线 + 3 个相切小圆

如果画不好，宁可退回 `i-flask`，也不要用 emoji / Unicode 符号代替（AGENTS 禁止）。

### 5.4 验收

- [ ] 截图里三组分明，主线组排在最上，组副标可见。
- [ ] 教师 / 学生 / 未登录下"实验报告"项文字正确；问答徽标仍然正常。
- [ ] 折叠分组的记忆功能仍可用。
- [ ] ≤1050px 的窄屏模式（只显示图标）下不崩。

---

## 6. T4 首页 Hero + 看板重排（P1）

### 6.1 结构

`renderDashboard()` 在 `dash-head` 之前插入 `homeHero(role)`：

```
.home-hero
├─ 标题行：BRAND_NAME（--fs-xl 800）  右侧：在线人数胶囊（从 dash-head 挪过来）+ [收起/展开]（仅登录用户）
├─ 一句话：「拍下摆动视频，平台自动识别、标定、提取角度，并反演出运动方程与物理参数。」
├─ .pipeline   技术链路（7 个节点，横排，节点之间用发丝线 + 小箭头连接）
└─ .exp4       4 张主线实验卡（grid 4 列；<1200px 时 2 列）
```

下面接原来的角色看板：

- 标题改为：教师「教学看板」/ 学生「我的学习进度」/ 未登录「平台概况」。
- `dash-head` 里的 `h2` 降为 `--fs-md`，和 Hero 拉开层级。

**收起状态**：登录用户可以收起 Hero，记在 localStorage `home_hero_collapsed`。收起后只保留标题行 + `.exp4`（卡片压成一行紧凑样式），`.pipeline` 隐藏。未登录用户**始终展开**（这是评委视角）。

### 6.2 技术链路 `.pipeline` 节点文案

| # | 节点名 | 小字说明 |
|---|---|---|
| 1 | 视频采集 | 树莓派实时取流 / 手机慢动作 |
| 2 | 目标检测 | YOLOv8 逐帧定位摆体 |
| 3 | 坐标标定 | 3 点建立角度坐标系 |
| 4 | 角度序列 | 解缠绕 · 离群剔除 · 补缝 |
| 5 | 降噪 | Savitzky–Golay / 高斯滤波 |
| 6 | 方程反演 | 解析解拟合 · ODE 复现 · 物理项库回归 |
| 7 | 报告评价 | 自动填入报告 · 教师批改 |

节点样式：白底描边小卡，`--r-md`；上面是 18px 线性图标（用已有的 `#i-*`，没有合适的就加新 symbol），下面是名称 `--fs-sm` 700 + 说明 `--fs-xs` `--t3`。第 6 个节点用 `--a1` 描边和字色，作为视觉焦点。一行放不下时横向滚动（`overflow-x:auto`），不要换行成两排。

> 文案里用"方程反演"作总称，括号里写清具体方法。是否使用"符号回归 / 机器学习"的提法见待决事项 D3：现有实现是"最小二乘 + 物理项库线性回归（SINDy 思路）"，用词要经得起答辩追问。

### 6.3 主线实验卡 `.exp4 > .exp-card`

每张卡：左上是动画图标（复用 type-card 的 `.p-icon/.d-icon/.t-icon/.n-icon`，hover 时摆动），名称 `--fs-base` 700，"反演什么"一行 `--fs-xs` `--t2`，一条代表性公式（KaTeX 行内，`--fs-sm`），底部两个按钮：[开始分析]（`goExpLearn(id)`）[预习]（`goPre(id)`，未登录隐藏）。

| id | 反演什么 | 代表公式 |
|---|---|---|
| danbai | 周期 T 与振幅衰减 β | `T=2\pi\sqrt{L/g}` |
| cizuni | 阻尼比 ζ，判别欠 / 临界 / 过阻尼 | `\ddot\theta+2\beta\dot\theta+\omega_0^2\theta=0` |
| niubai | ω₀、β，进而求转动惯量 I | `T=2\pi\sqrt{I/\kappa}` |
| ciliniudun | 多摆角度时序与能量传递 | `e=\sqrt{h/h_0}` |

这些文案放进 JS 常量 `EXP_HOME_INFO`，向导第 1 步（§7.2）也复用。

### 6.4 看板网格修正

- `statCards(items)`：给 `.stat-grid` 加内联 `style="--n:${items.length}"`；CSS 改为 `grid-template-columns:repeat(var(--n),minmax(0,1fr))`，`@media (max-width:1250px)` 时退回 `repeat(3,1fr)`。这样 6 张卡不会落单。
- 教师看板：「班级掌握情况」panel 改为**跨两列**放在 `dash-cols` 上方（单独一行），`.cls-grid` 改成 `repeat(auto-fit,minmax(150px,1fr))`，4~6 个班一行放得下。下面两列放：左「各实验报告量」，右「成绩分布」+「最近动态」。
- 去掉各 panel 的英文 `.en` 副标（T7 统一做也行）。「班级掌握情况」后面原来的英文说明改成中文小字："环形 = 批改率（已批改 / 已提交）"。
- 「最近动态」用 `.list-row` 和 `.pill-st` 重写。

### 6.5 未登录首页

Hero（完整展开）+ 原 `publicDash` 的统计卡与成绩分布。右侧"登录后解锁"卡改小，放到统计卡那一行的末尾，或作为一行横幅："登录后可使用视频分析、实验报告与 AI 助手 [前往登录]"。

### 6.6 验收

- [ ] 未登录截图：首屏能看到 品牌 / 一句话 / 7 节点链路 / 4 张实验卡。
- [ ] 教师截图：统计卡不落单；班级卡一行排完；可以收起 Hero，刷新后保持收起。
- [ ] 学生截图：Hero 下面是「我的学习进度」，流程表不受影响。
- [ ] 实验卡按钮能正确进入向导 / 指导页。

---

## 7. T5 向导减负（P2）

> [!WARNING]
> 向导里的 JS 大量按 id 取元素。**本包只允许移动 / 包裹元素、改样式和文案，不允许改任何现有 id、`data-*` 属性或 onclick 绑定。** 改完用脚本检查 `#wzp1..5` 区域的 div 开闭是否配平（AGENTS 坑 1）。

### 7.1 步骤条

- 去掉 `.wz-chip:not(.active) .wz-label{display:none}`，5 个步骤名称**始终显示**；已完成步骤用 `--t2`，未到达用 `--t3`，当前步骤 `--text` 700。
- `@media (max-width:1250px)` 时才隐藏非当前步骤的名称。

### 7.2 第 1 步：选择类型

- `.type-cards` 改成 `repeat(4,1fr)`（<1200px 时 2 列），卡片高度压到约 120px；去掉 `.eng` 英文行；删掉重复的"运动类型 EXPERIMENT TYPE"字段标签。
- 卡片下方新增 `#type_info` 说明面板（`selectType()` 末尾调用 `renderTypeInfo(type)` 填充；未选择时显示 `.empty-state`："选择上方一个实验，这里会显示拍摄要求与分析输出"）。面板分三栏：
  - **测什么**：`EXP_HOME_INFO[type]` 里的反演目标 + 代表公式
  - **怎么拍**：`EXP_SHOOT_TIPS[type]`（新常量，内容见下）
  - **得到什么**：输出清单（CSV / 角度-时间图 / 方程与物理量 / 标注视频（单摆、扭摆、磁阻尼摆））
  - 右上角放原来 card-header 里的「实验指导」按钮（`openCurrentGuide()`）

`EXP_SHOOT_TIPS` 初稿（实现时可对照 `teaching/guide_<id>.json` 的拍摄章节再校对）：

- danbai：相机正对摆动平面、固定不动；画面里要包含悬点；小角度释放（< 5°）；至少录 10 个周期。
- cizuni：手机慢动作或树莓派 60 FPS；画面里要包含转轴；铝板、磁铁位置在整个记录过程中不变。
- niubai：俯拍圆盘，标记点清晰；释放后不要碰支架；录到振幅明显衰减。
- ciliniudun：正面拍摄，所有摆球都在画面内；背景简洁；手动释放最外侧的摆球。

### 7.3 第 2 步：视频获取

- `.acq-cards` 改成紧凑的横向样式：高度约 72px，图标在左、文字在右，去掉 `.eng` 英文行；选中态 = `--a1` 描边 + 右上角小勾，**不铺橙色底**。
- **实验获取面板**（`#acq_panel_experiment`）在 ≥1200px 时改为两列：
  - 左列：释放装置（`#release_card_magnet` / `#release_card_gripper` / `#release_card_manual`，三选一显示的逻辑不变）
  - 右列：树莓派摄像头卡 + 二维云台卡
- 放进 `.dev-notes`（`<details>`，默认折叠，summary「设备调试信息」）的内容：
  - 树莓派卡里的 `start_stream.sh` / `stop_stream.sh` 两行命令和复制按钮
  - "树莓派屏上要取景：把 --preview none 改成 …"那段
  - 云台卡里的"上/下 俯仰 GPIO12 · … · 软限位 1000~2000 µs"那行
  - 夹爪卡里的"方向键每次约 2°"
- **留在外面**的提示（与操作强相关）："拍摄由网页开始/停止，慢放倍数填 1"；"录制期间避免移动云台（姿态变化会使标定失效）"。
- `#params_area`：内层 `.param-card` 改成白底描边（T7 统一处理）；英文标签去掉（`慢放倍数 Slow Motion` → `慢放倍数`）。

### 7.4 第 3 步：标定

- `#calib_preview` 改成左右两栏：
  - 左：`#thumb_canvas` 放大。显示宽度 `min(100%, 640px)`，位图尺寸按首帧宽高比设置（例如宽 640、高按比例）。**`updateThumbCanvas()` 是按 `c.width/c.height` 计算缩放的，改了 canvas 尺寸后这里不用改逻辑，但要在拿到首帧后设置 `thumbCanvas.width/height`。**点的半径从 4 调到 5。
  - 右：「标定说明」——按类型列出要点几个点、每个点的含义，颜色圆点必须取自 `calibPointStyle(i).color`（与标定弹窗一致），文字可以从 `teaching/guide_<id>.json` 的 `calibration_points` 读（`/api/teaching_content/guide_<id>`），读不到就用内置短文案。下面放「开始标定」「重置」按钮（原按钮挪过来，id 不变）和 `#calib_preview_text`。
- 未拿到首帧时，左栏显示 `.empty-state`："先在上一步上传或拍摄视频"。

### 7.5 第 4 步：运行

- 在保存路径上方加「本次分析配置」摘要（只读的 kv 列表，两列网格）：实验类型 / 视频来源（文件名或"树莓派录制 n 帧"）/ 总帧数 / 标定状态（"已完成 3/3 点"）/ 关键参数（慢放、跳过开头结尾、起始帧）。数据都从 `S` 和参数输入框读。
- 输出说明里的"输出 *_angle_time.csv & *_angle_time.png"改为按类型给出真实文件名（见 AGENTS 架构表）。

### 7.6 验收

- [ ] 4 种实验 × 上传 / 实验获取 两种来源，从第 1 步走到第 5 步都能跑通（至少用一段真实视频跑通扭摆和单摆）。
- [ ] 第 1 步、第 3 步截图里不再有大块空白。
- [ ] 树莓派连接 / 取景窗 / 云台 / 释放按钮功能不变；调试信息默认折叠。
- [ ] div 配平检查通过。

---

## 8. 后端配合改动（小）

### B1（T2 必需）把方程拟合图发给前端

`app.py` 的 `_run_processing()` 里，`result["sr_result"] = sr_result` 之前加上：

```python
sp = sr_result.get("plot_path") if isinstance(sr_result, dict) else None
if sp and os.path.exists(sp):
    sr_result["plot_preview"] = file_to_base64(sp)
```

### B2（建议，**需用户确认**后再做）单摆的 ω₀ 改为实测

`symbolic_regression.run_sr_danbai` 目前用 `L=0.5 m` 假设 `omega0`。建议改成：

- `period_s = 2*median(diff(pt))`（`|θ|` 的峰每半个周期出现一次）
- `omega_d = 2π/period_s`；`omega0 = sqrt(omega_d² + beta²)`；`zeta = beta/omega0`
- 返回结果新增 `A0`、`period_s`、`omega_d`、`omega0_source:"measured"`；保留 `pendulum_length` 参数但不再参与计算。
- 如果用户在参数里填了摆长 L（以后可加输入框），再额外给出 `g_est = 4π²L/T²`。

这改动影响物理结论，**必须先问用户**，再用一段真实单摆视频核对。

---

## 9. T6 / T7 视觉清理（P3）

### 9.1 T7 全局：去英文副标签、卡片去嵌套

- **删除**（直接删 HTML/JS 模板里的文字，不要用 CSS 隐藏）：
  - 所有 `.card-header .tag` 里的英文（`Experiment Type`、`Video & Parameters`、`Coordinate Calibration`、`Specify Output & Run`、`Processing Complete`、`My Reports`、`Grading Console`）。**例外**：报告编辑器 `.tag` 用来放状态胶囊，保留。
  - `.field-label .en`、`.panel-sec .en` 的英文内容（`.panel-sec .en` 里本来是中文说明的保留，例如"课前预习 → 课中实验 → …"）。
  - `.p-label`、`.section-label` 中的英文半句（如"串口设置 Serial Port"→"串口设置"）。
  - type-card / acq-card 的 `.eng` 行。
  - 登录页副标题"University Physics Experiment Platform"改为 `BRAND_SUB`。
- `.param-card`：`background:var(--card)`，保留 1px `--border`，去掉 hover 变色。
- `.card:hover`：删掉 hover 阴影和描边变化（卡片不可点击，hover 反馈是误导）。
- `.type-card` / `.acq-card`：底色 `--card`；hover 只把描边加深到 `--ba`，去掉 `translateY(-3px)`；选中态 `border-color:var(--a1)` + 外圈 `0 0 0 3px var(--fr)`，**不改底色**。
- 顶栏标题 `.topbar .view-title` 改用 `--fs-md`。
- 全局 `select{width:100%}` 这类规则如果存在，新增 `.sel-auto{width:auto;min-width:140px}` 供筛选器使用。

### 9.2 T6 指导页（`renderGuide` / `initDocPager`）

- 正文容器 `#guide_content` 设 `max-width:820px`。
- **顶部**：标题 + 细进度条（高 3px，`--a1`，宽 = n/N）+ 小字"第 n / N 节 · 节名"（继续用 `.doc-count` 承载）。
- **底部**：上一节 / 下一节按钮（`.doc-prev` / `.doc-next`，类名不变，`initDocPager` 不用改查找逻辑）。按钮右侧显示下一节的节名预告："下一节：实验原理 ›"。
- **左侧目录**（≥1400px 才显示）：`.doc-toc` 列出所有节标题，点击跳到对应节，当前节高亮 `--a1`。在 `initDocPager` 的 `show(n)` 里同步高亮。整体布局：`#guide_content` 内部改成 `grid-template-columns:180px minmax(0,820px)`。
- 「最后一页自动完成预习」回调 `_guideLastPageCb` 逻辑保持不变；通过目录直接跳到最后一节也算读完（这是现状行为，保持）。
- 每页内容过短的问题：不合并节（会影响"读到最后一页"的语义），靠目录 + 底部预告 + 限宽解决视觉上的空。

### 9.3 T6 报告（学生）

- `renderStudentReports`：列表改用 `.list-panel/.list-row`，**不再借用 `.calib-point` 样式**。行内容：`.lr-icon`（实验名首字）+ 标题 / 更新时间 + 右侧 [撤销 / 删除图标按钮] + `.pill-st`。状态胶囊字号**必须小于**标题。
- 新建报告那一行：`select.sel-auto` + 按钮，放在 `.sec-head` 右侧。
- **编辑器改为"进入"而不是"追加在下面"**：`openReportEditor` 渲染时隐藏 `#report_list_area`（新包一层容器），在编辑器顶部放「‹ 返回报告列表」按钮（点击调 `renderReports()`）。去掉 `area.scrollIntoView`，改为 `#center.scrollTop=0`。
- 草稿编辑态 ≥1300px 时，每个分节的 textarea 和 `.rep-preview` 左右并排（`grid-template-columns:1fr 1fr`），把页面长度减半；窄屏仍然上下排。`repPreview()` 用的是 `ta.nextElementSibling`，**并排时两者仍须是相邻兄弟元素**（外面包一层 grid 容器，别插入其他节点）。

### 9.4 T6 批改台（教师）

- 三个筛选 `select` 加 `.sel-auto` 横排一行，"查看 / 导入学生"放在这一行最右。
- 列表同样改用 `.list-row`；`.lr-icon` 显示学生姓氏，标题"姓名 · 实验"，meta"班级 · 提交时间"。
- `openGrading` 同样改成"进入"模式（隐藏列表，加返回按钮）。

### 9.5 T6 学生流程表（`fillMyFlow`）

- `.undo-btn` 默认 `opacity:0`，`.mf-row:hover .undo-btn, .undo-btn:focus-visible{opacity:1}`。
- panel 标题旁的说明改为："点击行进入实验；悬停可撤销已完成环节"。

### 9.6 验收

- [ ] 全站截图中找不到大写英文副标签（报告状态胶囊等例外除外）。
- [ ] 报告列表、批改台列表是"白底 + 发丝线分隔行"，状态是小胶囊。
- [ ] 报告编辑器进入 / 返回正常；保存、提交、撤销、删除、表格加行、预览、公式渲染全部正常（AGENTS 坑 9、表格键名 `tbl_` 约定不能破坏）。
- [ ] 指导页 ≥1400px 有目录；读到最后一节仍会自动标记预习。

---

## 10. T8 AI 面板上下文化（P4）

- 新常量 `AI_PRESETS`，按上下文 key 给出欢迎语 + 4~5 个预设问题：

| key | 欢迎语 | 预设问题 |
|---|---|---|
| `dashboard.public` | 我是实验 AI 助手，登录后可以就任一实验向我提问。 | （不显示预设；输入框禁用，placeholder"登录后可使用 AI 助手"） |
| `dashboard.teacher` | 可以帮你看班级数据、设计实验题或批改要点。 | 哪些班级批改进度落后？ / 给单摆实验出 3 道课前测验 / 杨氏模量报告常见错误有哪些？ |
| `dashboard.student` | 选一个实验开始吧，我会结合实验内容回答你。 | 我下一步该做什么？ / 怎么预习扭摆实验？ |
| `guide` | 正在阅读《{exp}》指导，有不懂的地方直接问。 | 这个实验的原理是什么？ / 本节的公式怎么推导？ / 有哪些注意事项？ / 误差主要来自哪里？ |
| `analysis.1-2` | 我们来做《{exp}》的视频分析。 | 拍摄时要注意什么？ / 慢放倍数怎么填？ / 树莓派怎么连接？ |
| `analysis.3` | 标定决定角度基准，有疑问随时问。 | 三个标定点分别点在哪里？ / 起始帧怎么选？ |
| `analysis.4-5` | 分析完成后可以和我一起解读结果。 | 解释一下这个方程的物理意义 / 阻尼比说明了什么？ / R² 不高可能是什么原因？ / 帮我写数据分析段落 |
| `reports` | 我可以帮你组织报告结构、检查计算与不确定度。 | 实验报告应该怎么写？ / 帮我检查数据处理 / 不确定度怎么算？ |
| `qas` | 提问前也可以先问我，常见问题我能直接回答。 | — |

- `updateAICtx()` 末尾调 `renderAIPresets(key)`：重绘 `.preset-questions`；**仅当对话里还没有用户消息时**，把首条 bot 欢迎语替换成新的欢迎语。
- 分析第 5 步的上下文里带上 `window._lastResult` 的摘要（β、ω₀、ζ、R²），加到 `aiContext()` 返回值的新字段 `result_summary`（后端 `ai_agent.build_system_prompt` 如果不认识这个字段会忽略；要让 AI 真正用上，需要在 `build_system_prompt` 里补一行把它写进系统提示——**这是可选的后端改动**）。
- 面板宽度默认值 380 → 340（`--w-ai` 默认值与 `initGutters` 里的后备值同步改）。
- 折叠状态记到 localStorage `ai_off`，刷新后保持。

**验收**：在各视图之间切换，预设问题跟着变；已经开始对话后切视图，欢迎语不会被覆盖；未登录时输入框禁用。

---

## 11. 进度清单（实现者逐项勾选并注明提交号）

- [x] T0 设计令牌 / 通用组件 / 截图工具（`tools/ui_shots.py` + `tools/fixtures/`）
- [x] T1 AI 徽章串味修复（`currentCtxExp()` 按当前视图隔离）
- [x] B1 sr 拟合图 base64（`app.py` 注入 `sr_result["plot_preview"]`）
- [x] T2 结果页重做（含 `renderResult` 抽取、fixture、方程主卡、指标卡、图表分段标签、写入报告、`resetWizard` 重新分析）
- [x] T3 导航重组（含 `BRAND_NAME`/`BRAND_SUB` 品牌常量、4 个实验专属 SVG 图标、主线「智能分析实验」与「拓展教学实验」分层、「教学协作」分组、角色化报告标题）
- [x] T4 首页 Hero + 看板网格（7 节点技术链路 + 4 张主线实验卡 + 在线状态 + 教师/学生可收起横幅 + 统计卡 `--n` 整行对齐 + 班级掌握情况全宽排列）
- [x] T5 向导减负（步骤 1 选中后三栏实验要点面板、步骤 2 紧凑横向获取卡片+双列设备区+`<details class="dev-notes">` 折叠硬件调试说明、步骤 3 560×315 缩略图与标定点位清单双栏布局、步骤 4 本次分析配置摘要与输出清单）
- [x] T7 去英文 / 去嵌套（全站移除冗余英文副标签、参数卡白底去嵌套、统一 `.sec-head` / `.list-panel` / `.list-row` / `.pill-st`）
- [x] T6 指导页 / 报告 / 批改台 / 流程表（指导页左侧固定章节目录 `.doc-toc` + 顶部阅读进度条 + 底部翻页器；报告与批改台改为列表/详情切换含「‹ 返回列表」；草稿编辑器宽屏左右并排预览；学生流程行撤销按钮悬停显现）
- [x] T8 AI 面板上下文化（`AI_PRESETS` 随视图/步骤动态切换快捷问题与欢迎语、未登录禁用输入、`aiContext()` 注入 `result_summary` 且 `ai_agent.py` 写入系统提示词、`ai_off` 折叠记忆、默认宽度 340px）
- [ ] B2 单摆 ω₀ 实测（**需用户确认**，当前前端对单摆仅展示实测包络衰减率 $\beta$ 与 $R^2$，估算 $\omega_0$ 标明参考）
- [x] T9 AGENTS.md 同步：已更新「前端设计规范」中导航、首页、结果页、报告编辑器进入模式、通用组件、`renderResult`/`resetWizard`/`currentCtxExp` 等新约定，并通过 `tools/ui_shots.py` 全量 29 张截图回归验证

## 12. 待决事项（需用户拍板）

| # | 问题 | 当前默认 |
|---|---|---|
| D1 | 作品正式名称（品牌名） | 暂用"大学物理实验平台"，放在 `BRAND_NAME` 常量里，改一处即可 |
| D2 | 比赛演示方式：现场操作 / 录屏 / 评委自行点击 | 按"评委自行点击"设计：未登录首页完整展示 Hero |
| D3 | 是否对外使用"符号回归 / 机器学习"的提法 | 前端统一用"方程反演（解析解拟合 · ODE 复现 · 物理项库回归）"，避免答辩时被追问 |
| D4 | B2 单摆 ω₀ 改为实测 | 未做；前端先不把单摆的 ω₀ / ζ 当测量值展示 |
| D5 | 扭摆以外的实验是否也做结果页标注视频预览 | 维持只有扭摆（AGENTS 现行规定） |

---

## 附录 A：改版前截图诊断摘要

- 首页（教师）：6 张统计卡第 6 张落单；班级卡第 4 张落单；整页都是成绩管理，没有任何技术展示。
- 导航：4 个视频实验与 6 个指导实验平铺，区别只是一个很小的"视频 / 指导"徽标；"师生互动""报告中心"两个分组各只有 1 项。
- 向导第 1 步：内容只占半屏；"选择实验类型 EXPERIMENT TYPE"下面紧跟一个重复的"运动类型 EXPERIMENT TYPE"。
- 向导第 2 步（实验获取）：COM 口、GPIO、µs、shell 命令全部直接铺开。
- 向导第 3 步：一个 200×110 的缩略图加一个按钮，下面整屏空白。
- 结果页（读代码得出）：5 个文件路径卡在前；方程在默认折叠的 `<details>` 里，用 Unicode 拼写；单摆的 ω₀ 是假设值。
- 指导页：每页约 5 行内容，下面大片空白；翻页器夹在标题和正文之间。
- 报告 / 批改台：借用标定点的 `.calib-point` 样式，灰底块堆叠；状态字比标题还大；三个筛选下拉各占一整行。
- AI 面板：徽章串味（在报告页显示"扭摆 · 步骤3 标定"）；欢迎语和预设问题固定不变。
