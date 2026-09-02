# -*- coding: utf-8 -*-
"""
轻量实验指导 Agent
===================
借鉴 opencode / DeepSeek-Reasonix 的 agent-loop 模式（工具调用循环），
但用平台现有的 OpenAI 兼容接口自研实现，不引入额外运行时。

工具集（领域工具，均只读）：
  get_experiment_guide     查询实验指导全文
  get_report_template      查询实验报告模板结构
  analyze_numeric_data     对学生贴的数据做均值/标准差/逐差等基础统计

流程：最多 3 轮工具调用（非流式），末轮流式输出最终回答；
每轮工具调用以 {'tool': ...} 事件流式通知前端展示状态。
"""
import json
from pathlib import Path

from experiments import get_experiment

TEACHING_DIR = Path(__file__).parent / "teaching"
TEMPLATE_DIR = Path(__file__).parent / "report_templates"

_MAX_TOOL_ROUNDS = 3


# ---------- 工具实现 ----------
def _tool_get_experiment_guide(experiment_id=None):
    exp = get_experiment(experiment_id or "")
    if exp is None:
        known = "、".join(e["id"] for e in
                          [x for x in __import__("experiments").EXPERIMENTS])
        return json.dumps({"error": f"未知实验 {experiment_id}", "可选": known}, ensure_ascii=False)
    path = TEACHING_DIR / f"{exp['guide_topic']}.json"
    if not path.exists():
        return json.dumps({"error": "该实验暂无指导内容"}, ensure_ascii=False)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _tool_get_report_template(experiment_id=None):
    path = TEMPLATE_DIR / f"{experiment_id}.json"
    if not path.exists():
        return json.dumps({"error": "该实验暂无报告模板"}, ensure_ascii=False)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _tool_analyze_numeric_data(data=None, operation="describe"):
    """基础统计：describe（均值/标准差/不确定度）、diff（逐差）。"""
    if not isinstance(data, list) or len(data) < 2:
        return json.dumps({"error": "data 需为至少 2 个数字的数组"}, ensure_ascii=False)
    try:
        vals = [float(x) for x in data]
    except (TypeError, ValueError):
        return json.dumps({"error": "data 中含非数字项"}, ensure_ascii=False)
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        std = var ** 0.5
        ua = std / (n ** 0.5)  # A 类不确定度（平均值标准差）
    else:
        std = ua = 0.0
    out = {"n": n, "mean": round(mean, 6), "std": round(std, 6), "uA_mean": round(ua, 6)}
    if operation == "diff":
        half = n // 2
        if half >= 1:
            diffs = [vals[i + half] - vals[i] for i in range(half)]
            out["successive_diff"] = [round(d, 6) for d in diffs]
            out["diff_mean"] = round(sum(diffs) / len(diffs), 6)
    return json.dumps(out, ensure_ascii=False)


TOOLS_IMPL = {
    "get_experiment_guide": _tool_get_experiment_guide,
    "get_report_template": _tool_get_report_template,
    "analyze_numeric_data": _tool_analyze_numeric_data,
}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_experiment_guide",
        "description": "查询某实验的完整指导内容（目的/原理/器材/步骤/注意事项）",
        "parameters": {"type": "object", "properties": {
            "experiment_id": {"type": "string", "description": "实验 id，如 danbai、sanxianbai、yangshi"},
        }, "required": ["experiment_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_report_template",
        "description": "查询某实验的报告模板结构（各节标题与写作提示），用于指导学生写报告",
        "parameters": {"type": "object", "properties": {
            "experiment_id": {"type": "string"},
        }, "required": ["experiment_id"]},
    }},
    {"type": "function", "function": {
        "name": "analyze_numeric_data",
        "description": "对学生提供的一组数值做基础统计：均值、标准差、A类不确定度，可选逐差法",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "array", "items": {"type": "number"}},
            "operation": {"type": "string", "enum": ["describe", "diff"]},
        }, "required": ["data"]},
    }},
]


# ---------- 上下文注入 ----------
def build_system_prompt(base_prompt, context):
    if not context:
        return base_prompt
    lines = ["", "## 当前用户环境（由平台自动注入，回答时主动结合）"]
    if context.get("user"):
        u = context["user"]
        lines.append(f"- 用户：{u.get('name','')}（{'教师' if u.get('role')=='teacher' else '学生'}）")
    if context.get("exp_id"):
        lines.append(f"- 当前实验：{context.get('exp_name') or ''}（id={context['exp_id']}，模式={context.get('exp_mode','')}）")
    view_name = {"analysis": "数据分析", "theory": "理论知识", "guide": "实验指导",
                 "sim": "交互模拟", "reports": "实验报告"}.get(context.get("view"), context.get("view"))
    if view_name:
        lines.append(f"- 用户正在查看：{view_name}")
    lines.append("- 学生提问实验相关问题时，可调用 get_experiment_guide / analyze_numeric_data 工具获取准确资料后再回答。")
    return base_prompt + "\n".join(lines)


# ---------- agent 主循环 ----------
def run_agent_chat(client, model, messages, context, base_system_prompt):
    """generator：逐个产出 SSE payload dict（{'tool'} / {'content'} / {'error'}）。"""
    msgs = [{"role": "system", "content": build_system_prompt(base_system_prompt, context)}]
    msgs += list(messages)

    for _ in range(_MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=model, messages=msgs, tools=TOOLS_SPEC,
            max_tokens=1024, temperature=0.3,
        )
        m = resp.choices[0].message
        if not m.tool_calls:
            if m.content:
                yield {"content": m.content}
            return
        msgs.append({
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [tc.model_dump() for tc in m.tool_calls],
        })
        for tc in m.tool_calls:
            name = tc.function.name
            yield {"tool": f"查询 {name} …"}
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = TOOLS_IMPL[name](**args)
            except Exception as e:
                result = json.dumps({"error": f"工具执行失败: {e}"}, ensure_ascii=False)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result[:6000]})

    # 工具轮次用尽后的最终回答（流式）
    stream = client.chat.completions.create(
        model=model, messages=msgs, stream=True, max_tokens=2048, temperature=0.7,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield {"content": chunk.choices[0].delta.content}
