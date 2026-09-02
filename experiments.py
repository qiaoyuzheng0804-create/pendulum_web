# -*- coding: utf-8 -*-
"""
实验注册表
===========
平台所有实验的单一数据源：左侧导航、教学指导白名单、报告模板路由都从这里取。
mode="video"  ｜ 完整视频分析实验（YOLO 流水线，processors/ 中有对应处理器）
mode="guide"  ｜ 指导型实验（理论/指导/报告模板/AI 辅导，暂无视频分析）
新增实验：在 EXPERIMENTS 加一条，再补 teaching/guide_<id>.json 与
report_templates/<id>.json 即可，无需改 app.py 的任何硬编码。
"""

BASE_DIR_LABEL = "物理实验"

EXPERIMENTS = [
    # ---------- 视频分析实验（摆） ----------
    {
        "id": "danbai",
        "name": "单摆",
        "name_en": "Simple Pendulum",
        "category": "力学 · 振动",
        "mode": "video",
        "brief": "测量摆动周期与阻尼系数，YOLO 视频自动分析",
        "guide_topic": "guide_danbai",
        "report_template": "danbai",
    },
    {
        "id": "cizuni",
        "name": "磁阻尼摆",
        "name_en": "Magnetic Damping Pendulum",
        "category": "力学 · 振动",
        "mode": "video",
        "brief": "涡流阻尼下的衰减振动，提取阻尼比",
        "guide_topic": "guide_cizuni",
        "report_template": "cizuni",
    },
    {
        "id": "niubai",
        "name": "扭摆",
        "name_en": "Torsional Pendulum",
        "category": "力学 · 振动",
        "mode": "video",
        "brief": "扭振周期与转动惯量测量",
        "guide_topic": "guide_niubai",
        "report_template": "niubai",
    },
    {
        "id": "ciliniudun",
        "name": "磁力牛顿摆",
        "name_en": "Magnetic Newton's Cradle",
        "category": "力学 · 碰撞",
        "mode": "video",
        "brief": "多摆球动量传递与能量损耗分析",
        "guide_topic": "guide_ciliniudun",
        "report_template": "ciliniudun",
    },
    # ---------- 指导型实验（大学基础物理常见实验） ----------
    {
        "id": "sanxianbai",
        "name": "三线摆测转动惯量",
        "name_en": "Trifilar Pendulum",
        "category": "力学 · 刚体",
        "mode": "guide",
        "brief": "用三线摆测量刚体绕中心轴的转动惯量",
        "guide_topic": "guide_sanxianbai",
        "report_template": "sanxianbai",
    },
    {
        "id": "yangshi",
        "name": "杨氏模量测量",
        "name_en": "Young's Modulus",
        "category": "力学 · 弹性",
        "mode": "guide",
        "brief": "拉伸法（光杠杆/读数显微镜）测金属丝杨氏模量",
        "guide_topic": "guide_yangshi",
        "report_template": "yangshi",
    },
    {
        "id": "niudunhuan",
        "name": "牛顿环测曲率半径",
        "name_en": "Newton's Rings",
        "category": "光学 · 干涉",
        "mode": "guide",
        "brief": "等厚干涉条纹测量透镜曲率半径与光波波长",
        "guide_topic": "guide_niudunhuan",
        "report_template": "niudunhuan",
    },
    {
        "id": "mikesun",
        "name": "迈克尔逊干涉仪",
        "name_en": "Michelson Interferometer",
        "category": "光学 · 干涉",
        "mode": "guide",
        "brief": "等倾干涉测激光波长与空气折射率",
        "guide_topic": "guide_mikesun",
        "report_template": "mikesun",
    },
    {
        "id": "shiboqi",
        "name": "示波器的使用",
        "name_en": "Oscilloscope",
        "category": "电学 · 仪器",
        "mode": "guide",
        "brief": "示波器基本操作与李萨如图形测频率",
        "guide_topic": "guide_shiboqi",
        "report_template": "shiboqi",
    },
    {
        "id": "luoqiu",
        "name": "落球法测液体粘度",
        "name_en": "Falling Ball Viscometer",
        "category": "力学 · 流体",
        "mode": "guide",
        "brief": "斯托克斯公式法测量蓖麻油等液体的粘滞系数",
        "guide_topic": "guide_luoqiu",
        "report_template": "luoqiu",
    },
]

_INDEX = {e["id"]: e for e in EXPERIMENTS}


def get_experiment(exp_id):
    return _INDEX.get(exp_id)


def guide_topics():
    """teaching/ 白名单：共享理论 + 所有实验指导。"""
    return {"theory"} | {e["guide_topic"] for e in EXPERIMENTS}


def public_list():
    """下发给前端的精简列表（左导航用）。"""
    return [
        {k: e[k] for k in ("id", "name", "name_en", "category", "mode", "brief")}
        for e in EXPERIMENTS
    ]
