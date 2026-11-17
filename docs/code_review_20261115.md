# 仓库代码审查（2026-11-15）

本轮以当前工作区为审查对象，包含已有未提交修改。确认 14 项问题：5 项 P1（优先修复），9 项 P2（常规修复）。业务源码未修改。

覆盖后端、前端、四种视频处理器、公共观测与拟合模块、树莓派/OpenMV 管理器、启动停止脚本、演示数据、教学/报告配置，以及自有固件入口与上位机脚本。25 个 Python 文件合计 10,782 行通过语法解析。前端两个模板合计 5,622 行，执行了关键交互验证。STM32 厂商标准库、CMSIS 与第三方压缩 JavaScript 未逐行重审；没有执行 Keil 编译、真实 YOLO 推理或硬件测试。

| 编号 | 等级 | 问题 | 验证方式 |
| --- | --- | --- | --- |
| R01 | P1 | 启动迁移覆盖真实报告、教师评语与部分成绩 | 临时数据库复现 |
| R02 | P1 | 问答附件可承载同源脚本 | 上传/下载接口与真实浏览器复现 |
| R03 | P1 | Python 3.9–3.11 的固定依赖存在冲突 | PyPI 发布元数据核验 |
| R04 | P1 | 页面仍开着时停止脚本跳过断电清理 | 停止脚本、看门狗与固件控制路径核对 |
| R05 | P1 | 磁力牛顿摆忽略慢放倍数 | 合成视频与实际路由函数复现 |
| R06 | P2 | 分析途中换实验导致结果和课中进度串写 | 真实前端、延迟响应复现 |
| R07 | P2 | 磁力牛顿摆零检出仍返回成功 | 合成视频、空检测结果复现 |
| R08 | P2 | 单摆衰减拟合受平衡角偏移严重影响 | 无噪声合成信号复现 |
| R09 | P2 | 逐帧回退图像与确认的起始帧不一致 | 真实前端、回退帧响应复现 |
| R10 | P2 | 会话清理按上传年龄删除运行中的视频 | 正在处理的过期会话复现 |
| R11 | P2 | 清理接口没有登录与会话归属校验 | 匿名请求复现 |
| R12 | P2 | 磁阻尼摆的 auto 慢放模式被 API 提前拒绝 | 实际参数解析函数复现 |
| R13 | P2 | 教师“批改本实验报告”没有筛选实验 | 真实浏览器复现 |
| R14 | P2 | AI 流中的错误被吞掉并写入空回复 | 真实前端、错误 SSE 复现 |

## R01：启动迁移必须限制到明确标记的演示记录

位置：[platform_hub.py:175](C:/Users/MECHREV/Desktop/pendulum_web/platform_hub.py:175)、[platform_hub.py:190](C:/Users/MECHREV/Desktop/pendulum_web/platform_hub.py:190)。

`_upgrade_demo_reports()` 查询全部报告，把所有已提交/已批改但不完整的报告视为需要替换；完整报告如果教师评语为空，也会被填入演示评语。函数还会按预设用户名和实验重置成绩。`init_platform()` 每次启动均调用它，没有先确认记录确实属于演示数据。

复现：为非演示用户 `real_student` 插入包含独有文字的已提交报告，运行迁移后原文消失，内容变成演示报告；将 `s240101` 的单摆成绩改为 61，迁移后恢复成 92。旧版或不完整的真实报告会在启动时被永久改写，实际批改也可能被撤销。

建议：先按明确、持久的演示数据标识筛选；缺分节、空评语、用户名碰巧相同均不能作为覆盖真实内容的依据。对已被改写的数据应从迁移前备份恢复，不能靠现存内容猜回。

## R02：附件响应不能信任上传者提供的 MIME

位置：[platform_hub.py:992](C:/Users/MECHREV/Desktop/pendulum_web/platform_hub.py:992)、[platform_hub.py:1026](C:/Users/MECHREV/Desktop/pendulum_web/platform_hub.py:1026)。

上传只按扩展名放行，却把客户端 `f.mimetype` 存入数据库；下载时以此 MIME 原样内联返回。学生可以上传扩展名为 `.txt` 的 HTML，声明 `text/html`，通过公共问答里的文件链接诱导教师打开。页面脚本将拥有平台同源访问能力，报告渲染的 DOMPurify 不覆盖此路径。

复现：上传正文为设置无害标志的 script 的 `notes.txt`，响应头为 `Content-Type: text/html; charset=utf-8`、`Content-Disposition: inline`；浏览器访问附件 URL 后 `window.auditXSS === true`。后端测试也确认附件绑定到公共问题后仍以相同类型返回。

建议：服务端确定响应类型，文档附件强制下载；图片/音频按内容验证后使用受控 MIME，并设置 `X-Content-Type-Options: nosniff`。不能仅改正文净化函数。

## R03：固定的 OpenCV 与 NumPy 版本在 Python 3.9 以上不兼容

位置：[requirements.txt:5](C:/Users/MECHREV/Desktop/pendulum_web/requirements.txt:5)、[requirements.txt:16](C:/Users/MECHREV/Desktop/pendulum_web/requirements.txt:16)。

固定的 `opencv-python==4.13.0.92` 对 Python >=3.9 要求 `numpy>=2`，而仓库固定 `numpy==1.24.4`。因此在注释所覆盖的 Python 3.9–3.11 环境，全新依赖解析无法成功。Python 3.8 对应 OpenCV 的另一条条件依赖，不属于此冲突范围。[OpenCV 4.13.0.92 官方发布元数据](https://pypi.org/pypi/opencv-python/4.13.0.92/json)。

建议：选定受支持的 Python 版本，统一验证 OpenCV、NumPy、SciPy、Pandas、Matplotlib 与 Torch 的兼容组合；不要单独强升 NumPy 后保留其他旧二进制依赖。

## R04：停止脚本没有请求真正的安全关停

位置：[停止服务器.bat:13](C:/Users/MECHREV/Desktop/pendulum_web/停止服务器.bat:13)、[停止服务器.bat:28](C:/Users/MECHREV/Desktop/pendulum_web/停止服务器.bat:28)、[app.py:1687](C:/Users/MECHREV/Desktop/pendulum_web/app.py:1687)。

脚本发送的是移除 `manual-stop` 客户端的告别请求，而浏览器心跳使用随机客户端 ID。网页仍打开时，实际客户端不会被移除，看门狗继续保活；脚本超时后执行 `taskkill /f`，绕过 `_shutdown_safety()`。启动脚本直接强杀旧实例也有相同缺口。

STM32 电磁铁固件只在收到字符 `0` 时断电，没有串口断开自动断电逻辑；因此关闭 Python 进程不保证解除此前的通电状态。这是控制路径确认，未操作实物验证电气状态。

建议：增加经过适当权限限制的显式关停流程，先确认安全清理完成，再终止进程；启动替换旧实例也复用该流程。强杀只能是清理失败后的兜底，并明确提示断电未确认。

## R05：磁力牛顿摆必须使用实际慢放倍数换算时间

位置：[app.py:850](C:/Users/MECHREV/Desktop/pendulum_web/app.py:850)、[processors/ciliniudun_processor.py:162](C:/Users/MECHREV/Desktop/pendulum_web/processors/ciliniudun_processor.py:162)。

前端所有实验均展示、发送慢放倍数，但磁力牛顿摆分派未传该参数，处理器也没有对应入口。时间步长始终为播放帧间隔，`skip_end_sec` 也按播放帧率换算，慢动作素材的 CSV 时间轴和速度等后续分析会出错。

复现：30 FPS 的合成视频，经实际 `_run_processing()` 传入慢放 16 倍，返回 `dt=0.0333333333 s`，应为 `0.0020833333 s`，整条时间轴扩大 16 倍。

建议：贯通前端、路由、处理器的慢放参数，对真实时间、采样和末尾截取明确单位，并返回采用的时间基准。

## R06：分析响应需要绑定启动时的实验和会话

位置：[templates/index.html:2913](C:/Users/MECHREV/Desktop/pendulum_web/templates/index.html:2913)、[templates/index.html:2982](C:/Users/MECHREV/Desktop/pendulum_web/templates/index.html:2982)。

开始处理只禁用开始按钮，用户仍可从导航切换实验，`selectType()` 会重置全局 `S`。旧请求返回时，用当前 `S.motionType` 调用 `markLab()`、渲染结果，并生成 `_lastResult.exp`，没有检查响应仍属于当前会话。

复现：启动扭摆处理，在返回前选择磁阻尼摆，再释放旧响应。结果被标为 `cizuni`，磁阻尼摆课中进度变成已完成，扭摆仍未完成；一键写入报告也会沿用错误实验 ID。

建议：开始时捕获不可变的实验 ID、会话 ID 和请求 token；返回时校验 token，进度与报告归属使用该次任务保存的信息。

## R07：零检出不能视为完成实验

位置：[processors/ciliniudun_processor.py:717](C:/Users/MECHREV/Desktop/pendulum_web/processors/ciliniudun_processor.py:717)、[processors/ciliniudun_processor.py:778](C:/Users/MECHREV/Desktop/pendulum_web/processors/ciliniudun_processor.py:778)。

成功条件只检查读过的帧数，不检查真实角度观测数。检测全空或所有分配被拒绝时，仍导出全 NaN 角度 CSV、空图，并返回正常结果；前端随后自动标记课中完成。

复现：45 帧视频配空检测器，返回 `num_points=45`、正常结果字典，角度列有限值总数为 0。

建议：分别统计每个摆的真实有效观测，零检出直接返回明确失败；部分摆缺失应下发明确质量状态，不能用解码帧数冒充有效数据点数。

## R08：单摆衰减拟合不能直接对未归中心的角度取绝对值

位置：[processors/symbolic_regression.py:23](C:/Users/MECHREV/Desktop/pendulum_web/processors/symbolic_regression.py:23)。

观测管线保留相对标定参考方向的原始角度；拟合却直接取绝对值并把峰值拟合成趋零指数，没有估计平衡角。标定零偏或静平衡偏移会被误识别为缓慢衰减。

复现：用 30 秒、100 Hz 的无噪声信号 `angle(t)=offset+5*exp(-0.1*t)*cos(2*pi*t)`。当 offset=0 时，beta=0.1、R2=1；仅将 offset 改成 10°，beta 变成 0.0122045、R2 仍约 0.9104，误差约 88%。

建议：保持原始 CSV 不变，在拟合阶段估计平衡角，或用上下包络半峰谷差构建振幅。拟合优度不能代替零偏稳健性检查。

## R09：逐帧回退时必须采用服务端实际返回的帧号

位置：[app.py:455](C:/Users/MECHREV/Desktop/pendulum_web/app.py:455)、[templates/index.html:2521](C:/Users/MECHREV/Desktop/pendulum_web/templates/index.html:2521)。

服务端取帧超时可返回最近缓存帧，并通过 `X-Frame-Index` 告知实际编号。前端只读取 `X-Total-Frames`，继续显示、确认请求的帧号；在尚未加载完时也允许确认。浏览器不能播放的编码或高分辨率慢 seek 最容易触发。

复现：请求第 20 帧，模拟服务端按设计回退到第 3 帧；画面已经换成第 3 帧，`frameIdx` 和确认后的 `startFrame` 仍为 20。

建议：图像与实际帧号作为一个原子结果更新；精确目标帧尚未到达时禁用确认，或者明确提示用户接受实际返回帧。

## R10：TTL 清理必须排除正在处理的会话

位置：[app.py:535](C:/Users/MECHREV/Desktop/pendulum_web/app.py:535)。

会话只有创建时间，取帧、调整标定和开始分析均不更新；清理仅按上传后 30 分钟判断过期，不参考处理锁或该会话的进度。后台每五分钟及新上传时都会触发，能删除长视频正在使用的会话及源文件，使分析中断或结果无法再取流。

复现：创建年龄 1801 秒的会话，并在 `_progress` 中记录它正在处理；调用清理后会话与其测试视频均被删除。

建议：保存最后活动时间和任务运行状态；正在运行的输入必须固定保留，结束后再进入过期清理。

## R11：会话清理需要登录与所有者检查

位置：[app.py:988](C:/Users/MECHREV/Desktop/pendulum_web/app.py:988)、[app.py:696](C:/Users/MECHREV/Desktop/pendulum_web/app.py:696)。

`/api/cleanup` 是公开的删除接口，持有任一有效会话 ID 即可删去上传和会话。会话创建时没有保存所有者，处理、原视频和逐帧接口也只有“已登录”检查，缺少会话归属边界。

复现：匿名客户端向一个已知的、临时构造的有效会话发送 cleanup 请求，得到 HTTP 200，测试视频被删除。这里没有证明随机 UUID 可被猜中；问题是 ID 一旦通过共享或其他渠道获得，就充当了删除权限。

建议：上传与设备会话记录创建者，对读取、处理、进度和清理统一检查归属；如允许教师共享访问，应显式定义该权限。

## R12：auto 模式在公共数值解析阶段被拒绝

位置：[app.py:724](C:/Users/MECHREV/Desktop/pendulum_web/app.py:724)、[app.py:835](C:/Users/MECHREV/Desktop/pendulum_web/app.py:835)。

磁阻尼处理器支持 `slow_motion_factor="auto"`，分派代码也意图原样传入，但公共解析先执行 `float("auto")` 并直接返回格式错误，读取 MP4 拍摄帧率的逻辑无法通过 API 到达。当前网页固定发送数字，所以此问题影响公开 API 的 auto 能力，并非网页数字模式。

复现：直接调用实际 `_run_processing()`，传入 auto，在会话与视频检查前即返回“处理参数格式错误”。

建议：按实验类型解析该字段，磁阻尼摆允许 auto，其他分支再验证有限正数。

## R13：教师快捷批改入口仍引用已不存在的筛选控件

位置：[templates/index.html:3797](C:/Users/MECHREV/Desktop/pendulum_web/templates/index.html:3797)、[templates/index.html:4629](C:/Users/MECHREV/Desktop/pendulum_web/templates/index.html:4629)。

`openTeacherReportsFor()` 寻找 `#t_f_exp` 并尝试调用 `loadTeacherReports()`，当前批改台实际使用 `_gf.exp`，下拉框没有该 ID，也没有该函数。

复现：教师调用 `openTeacherReportsFor('danbai')` 后，`_gf.exp` 和下拉值仍为空，显示全部 21 份非草稿演示报告。

建议：先设置 `_gf.exp` 再进入报告视图，复用现有 `renderGradingConsole()` 的筛选数据源，去除固定延时和旧函数引用。

## R14：AI 错误不能在 JSON 解析 catch 中被吞掉

位置：[templates/index.html:4235](C:/Users/MECHREV/Desktop/pendulum_web/templates/index.html:4235)。

收到 SSE 的 error 字段后，代码在同一个 try 里抛错，紧接着由空 catch 吞掉。错误不会进入外层失败处理，反而显示“无回复”，再把空 assistant 消息加入历史；已经输出部分内容时甚至会把失败显示成正常截断回答。

复现：返回合法错误 SSE 和 DONE，页面显示“（无回复）”，历史中保留此次用户请求及空 assistant 回复。

建议：单独捕获 JSON 解析异常；解析成功的服务错误应传播到外层，展示错误状态并按既定策略恢复历史。

## 验证记录与局限

- 后端隔离复现：[check_review.py](C:/Users/MECHREV/Desktop/pendulum_web/tools/_shots/audit_20261115/check_review.py)，[结果](C:/Users/MECHREV/Desktop/pendulum_web/tools/_shots/audit_20261115/results.json)。
- 无头 Chrome 交互复现：[check_browser.py](C:/Users/MECHREV/Desktop/pendulum_web/tools/_shots/audit_20261115/check_browser.py)，[结果](C:/Users/MECHREV/Desktop/pendulum_web/tools/_shots/audit_20261115/browser_results.json)。
- 两个前端模板经过 Jinja 渲染后，内联 JavaScript 均通过 `node --check`。
- 对教学 JSON、报告模板、结果 fixture 和演示报告提取的 688 个公式，使用仓库本地 KaTeX 严格解析，全部通过；这些文本的意外 C0 控制字符扫描及显示公式分隔符检查均通过。[公式检查结果](C:/Users/MECHREV/Desktop/pendulum_web/tools/_shots/audit_20261115/formula_results.json)。
- 复现脚本和结果位于已忽略的 `tools/_shots/`，不会随普通提交自动入库。正式交付文本另行做控制字符检查。
- 后端测试使用临时 SQLite、合成视频和固定检测器；浏览器测试复用真实模板与 platform_hub Blueprint，另建本地临时服务，对分析/AI 返回使用受控响应。app.py 的目标函数以 AST 抽取后执行原始函数体，避免导入时迁移正式数据库及清理真实上传文件。
- 测试解释器为现有 Python 3.12.7，现有 OpenCV 4.11.0.86、NumPy 1.26.4。它不是 requirements.txt 所锁定的全新环境；R03 使用官方发布元数据单独核验，没有改装用户环境。
- 未启动正式 app.py，未改动现有 platform.db、模型文件或固件；未调用真实 LLM、串口或树莓派。运行验证说明软件路径能够复现，不能替代真实视频检测质量、传感器时间基准及实物断电测试。
