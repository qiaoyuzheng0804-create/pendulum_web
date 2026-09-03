# -*- coding: utf-8 -*-
"""
师生交互平台（Blueprint）
========================
登录认证（Flask session cookie）+ 实验报告提交批改 + 实验课全流程闭环
（课前预习 → 课中实验 → 课后报告 → 教师批改反馈）+ 师生公共问答讨论区
（全员可见、师生均可回复；文字 / 图片 / 语音附件）+ 教学数据看板统计。
SQLite，零外部依赖。

表：
  users        —— 用户名 / 口令哈希 / 角色(student|teacher) / 姓名 / 班级(class_name)
  submissions  —— 报告：学生、实验、分节 Markdown 内容(JSON)、状态、分数、评语
  progress     —— 学习流程：每学生×每实验的 课前预习 / 课中实验 完成标记
  questions    —— 学生提问（可关联某实验；标题 + 正文）
  replies      —— 问答线程回复（公共讨论区：教师和全体学生均可回复，支持多轮）
  attachments  —— 消息附件（image / audio / file），归属 question 或 reply
首次运行自动建库并预置演示账号：teacher / 123456（教师）、student / 123456（学生）；
若不存在任何"带班级的学生"，还会预置 3 个演示班级及学生（便于看板演示）。
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Blueprint, render_template, request, jsonify, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

from experiments import get_experiment, EXPERIMENTS
from demo_reports import build_demo_content, is_old_demo_content

BASE_DIR_MARKER = "platform"
_db_path = None  # 由 init_platform(app) 注入
_qa_dir = None   # 问答附件目录（uploads/qas），由 init_platform 注入

# 附件类型判定与白名单
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_AUDIO_EXTS = {".webm", ".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac"}
_DOC_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
             ".txt", ".md", ".csv"}
_ALLOWED_EXTS = _IMAGE_EXTS | _AUDIO_EXTS | _DOC_EXTS
_SIZE_LIMITS = {"image": 10 * 1024 * 1024, "audio": 20 * 1024 * 1024,
                "file": 10 * 1024 * 1024}


# ---------- DB ----------
def _conn():
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------- 演示数据（幂等；仅当没有任何带班级的学生时注入） ----------
_DEMO_CLASSES = [
    ("物电 2401", ["s240101", "王明"], ["s240102", "李娜"],
     ["s240103", "张伟"], ["s240104", "刘洋"]),
    ("物电 2402", ["s240201", "陈晨"], ["s240202", "赵磊"],
     ["s240203", "孙悦"], ["s240204", "周杰"]),
    ("应物 2401", ["s240301", "吴迪"], ["s240302", "郑爽"],
     ["s240303", "冯刚"], ["s240304", "许晴"]),
]

# 每班学生的演示报告：(username, exp_id, status, score|None)
_DEMO_SUBS = [
    # 物电 2401
    ("s240101", "danbai", "graded", 92), ("s240101", "sanxianbai", "submitted", None),
    ("s240102", "cizuni", "graded", 85), ("s240102", "yangshi", "submitted", None),
    ("s240103", "niubai", "graded", 78), ("s240103", "sanxianbai", "draft", None),
    ("s240104", "ciliniudun", "graded", 66), ("s240104", "yangshi", "submitted", None),
    # 物电 2402
    ("s240201", "danbai", "graded", 88), ("s240201", "niudunhuan", "submitted", None),
    ("s240202", "cizuni", "graded", 74), ("s240202", "sanxianbai", "submitted", None),
    ("s240203", "sanxianbai", "graded", 81), ("s240203", "danbai", "draft", None),
    ("s240204", "yangshi", "graded", 69), ("s240204", "mikesun", "submitted", None),
    # 应物 2401
    ("s240301", "shiboqi", "graded", 90), ("s240301", "luoqiu", "submitted", None),
    ("s240302", "luoqiu", "graded", 84), ("s240302", "shiboqi", "submitted", None),
    ("s240303", "mikesun", "graded", 63), ("s240303", "niudunhuan", "submitted", None),
    ("s240304", "niudunhuan", "graded", 76), ("s240304", "cizuni", "draft", None),
]

_DEMO_FLOW_EXPS = ["danbai", "cizuni", "niubai", "sanxianbai", "yangshi",
                   "shiboqi", "luoqiu"]  # 预置预习/课中完成的实验子集


def _seed_demo_if_empty():
    """无任何带班级的学生时，注入演示班级 + 学生 + 报告/流程/问答数据。"""
    with _conn() as c:
        n_classed = c.execute(
            "SELECT COUNT(*) AS n FROM users WHERE class_name IS NOT NULL"
            " AND class_name!=''").fetchone()["n"]
        if n_classed > 0:
            return
        now = datetime.now().isoformat(timespec="seconds")
        for cls_name, *members in _DEMO_CLASSES:
            for username, dname in members:
                c.execute(
                    "INSERT INTO users(username,pw_hash,role,display_name,class_name)"
                    " VALUES(?,?,?,?,?)",
                    (username, generate_password_hash("123456"), "student", dname, cls_name))
        # 演示报告（完整内容：6 分节 + 数据表格，见 demo_reports.py）
        for username, exp_id, status, score in _DEMO_SUBS:
            content = json.dumps(build_demo_content(exp_id) or {}, ensure_ascii=False)
            graded_at = None
            if status == "graded":
                graded_at = now
                feedback = "（演示评语）实验步骤完整，误差分析可再深入。"
            elif status == "submitted":
                feedback = ""
            else:
                feedback = ""
            c.execute(
                "INSERT INTO submissions(username,exp_id,content,status,score,feedback,"
                " created_at,updated_at,graded_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (username, exp_id, content, status, score, feedback, now, now, graded_at))
        # 演示流程进度：演示学生大多完成预习与课中实验
        demo_students = [u["username"] for u in
                         c.execute("SELECT username FROM users WHERE role='student'"
                                   " AND class_name IS NOT NULL AND class_name!=''")]
        for i, username in enumerate(demo_students):
            for exp_id in _DEMO_FLOW_EXPS:
                pre = 1 if (i + hash(exp_id)) % 5 != 0 else 0
                lab = 1 if (i * 2 + len(exp_id)) % 6 != 0 else 0
                c.execute(
                    "INSERT INTO progress(username,exp_id,pre_done,pre_at,lab_done,lab_at)"
                    " VALUES(?,?,?,?,?,?)",
                    (username, exp_id, pre, now if pre else None,
                     lab, now if lab else None))
        # 演示问答：2 条已回复 + 1 条待回复
        qa_specs = [
            ("s240101", "danbai", "周期测量时如何减少人为计时误差？",
             "平台视频分析得到 T=1.72s，想确认手动停表方法的改进建议。",
             [("teacher", "建议测量 50 个周期总时间后除以周期数；用视频慢放对齐摆球通过最低点的瞬间，"
                         "并用光电门或本平台的 YOLO 逐帧分析代替人工读数。")]),
            ("s240301", "shiboqi", "李萨如图形不稳定怎么办？",
             "两个通道频率比设置后图形一直在旋转。",
             [("teacher", "旋转说明两信号存在微小频率差：先校准 X、Y 通道同一信号的频率与幅度，"
                         "再用标准频率源比对，图形稳定后再读数。")]),
            ("s240201", "niudunhuan", "牛顿环中心是亮斑而非暗斑", "", None),
        ]
        for username, exp_id, title, body, replies in qa_specs:
            cur = c.execute(
                "INSERT INTO questions(username,exp_id,title,body,created_at)"
                " VALUES(?,?,?,?,?)", (username, exp_id, title, body or "", now))
            qid = cur.lastrowid
            if replies:
                for author, text in replies:
                    c.execute("INSERT INTO replies(qid,username,body,created_at)"
                              " VALUES(?,?,?,?)", (qid, author, text, now))


def _upgrade_demo_reports(c):
    """把旧版占位演示报告升级为完整内容（按旧标记识别，幂等：升级后不再含标记）。"""
    rows = c.execute("SELECT id, exp_id, content FROM submissions").fetchall()
    for r in rows:
        if is_old_demo_content(r["content"]):
            new = build_demo_content(r["exp_id"])
            if new:
                c.execute("UPDATE submissions SET content=? WHERE id=?",
                          (json.dumps(new, ensure_ascii=False), r["id"]))


def init_platform(app, db_path):
    """建表 + 幂等迁移（班级列 / 流程 / 问答 / 附件）+ 预置账号与演示数据。"""
    global _db_path, _qa_dir
    _db_path = str(db_path)
    _qa_dir = Path(__file__).parent / "uploads" / "qas"
    _qa_dir.mkdir(parents=True, exist_ok=True)

    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('student','teacher')),
                display_name TEXT NOT NULL DEFAULT '',
                class_name TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS submissions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                exp_id TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'draft'
                    CHECK(status IN ('draft','submitted','graded')),
                score REAL,
                feedback TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                graded_at TEXT
            );
            CREATE TABLE IF NOT EXISTS progress(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                exp_id TEXT NOT NULL,
                pre_done INTEGER NOT NULL DEFAULT 0,
                pre_at TEXT,
                lab_done INTEGER NOT NULL DEFAULT 0,
                lab_at TEXT,
                UNIQUE(username, exp_id)
            );
            CREATE TABLE IF NOT EXISTS questions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                exp_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL,
                body TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS replies(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                qid INTEGER NOT NULL,
                username TEXT NOT NULL,
                body TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attachments(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_kind TEXT NOT NULL DEFAULT 'temp'
                    CHECK(owner_kind IN ('temp','question','reply')),
                owner_id INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL,
                filename TEXT NOT NULL,
                orig_name TEXT NOT NULL,
                mime TEXT NOT NULL DEFAULT '',
                size INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sub_user ON submissions(username);
            CREATE INDEX IF NOT EXISTS idx_sub_exp ON submissions(exp_id);
            CREATE INDEX IF NOT EXISTS idx_q_user ON questions(username);
            CREATE INDEX IF NOT EXISTS idx_r_qid ON replies(qid);
            CREATE INDEX IF NOT EXISTS idx_att_owner ON attachments(owner_kind, owner_id);
            """
        )
        # 幂等迁移：老库 users 无 class_name 列 → ALTER 补列
        cols = [r[1] for r in c.execute("PRAGMA table_info(users)")]
        if "class_name" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN class_name TEXT NOT NULL DEFAULT ''")
        # 预置演示账号（仅当用户表为空时）
        if c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] == 0:
            c.execute(
                "INSERT INTO users(username,pw_hash,role,display_name) VALUES(?,?,?,?)",
                ("teacher", generate_password_hash("123456"), "teacher", "演示教师"),
            )
            c.execute(
                "INSERT INTO users(username,pw_hash,role,display_name) VALUES(?,?,?,?)",
                ("student", generate_password_hash("123456"), "student", "演示学生"),
            )
        # 旧版占位演示报告 → 完整内容（幂等迁移，随本事务提交）
        _upgrade_demo_reports(c)
    _seed_demo_if_empty()
    app.register_blueprint(platform_bp)


# ---------- helpers ----------
def current_user():
    if not session.get("uid"):
        return None
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()
    if row is None:
        session.clear()
    return row


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        u = current_user()
        if u is None:
            return jsonify({"error": "请先登录。"}), 401
        return fn(u, *a, **kw)
    return wrapper


def teacher_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(u, *a, **kw):
        if u["role"] != "teacher":
            return jsonify({"error": "需要教师权限。"}), 403
        return fn(u, *a, **kw)
    return wrapper


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _sub_public(row, with_content=False):
    d = {
        "id": row["id"], "username": row["username"], "exp_id": row["exp_id"],
        "status": row["status"], "score": row["score"], "feedback": row["feedback"],
        "updated_at": row["updated_at"], "graded_at": row["graded_at"],
    }
    if with_content:
        try:
            d["content"] = json.loads(row["content"])
        except Exception:
            d["content"] = {}
    return d


def _user_brief(username):
    with _conn() as c:
        r = c.execute("SELECT username,display_name,role,class_name FROM users"
                      " WHERE username=?", (username,)).fetchone()
    if r is None:
        return {"username": username, "display_name": username, "role": "student",
                "class_name": ""}
    return {"username": r["username"], "display_name": r["display_name"],
            "role": r["role"], "class_name": r["class_name"]}


def _attachments_of(owner_kind, owner_id):
    with _conn() as c:
        rows = c.execute(
            "SELECT id,kind,orig_name,mime,size FROM attachments"
            " WHERE owner_kind=? AND owner_id=? ORDER BY id", (owner_kind, owner_id)).fetchall()
    return [dict(r) for r in rows]


def _bind_attachments(att_ids, owner_kind, owner_id):
    """把提问/回复请求中带上的附件 id 绑定到刚创建的消息上。"""
    if not att_ids:
        return
    marks = ",".join("?" for _ in att_ids)
    with _conn() as c:
        c.execute(
            f"UPDATE attachments SET owner_kind=?, owner_id=? WHERE id IN ({marks})"
            " AND owner_kind='temp'", (owner_kind, owner_id, *att_ids))


# ---------- blueprint ----------
platform_bp = Blueprint("platform", __name__)


# ---------- 页面 ----------
@platform_bp.route("/login")
def login_page():
    return render_template("login.html")


# ---------- 认证 ----------
@platform_bp.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"error": "请输入用户名和密码。"}), 400
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row is None or not check_password_hash(row["pw_hash"], password):
        return jsonify({"error": "用户名或密码错误。"}), 401
    session.clear()
    session["uid"] = row["id"]
    session.permanent = True
    return jsonify({"username": row["username"], "role": row["role"],
                    "display_name": row["display_name"], "class_name": row["class_name"]})


@platform_bp.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"ok": True})


@platform_bp.route("/api/auth/me")
def auth_me():
    u = current_user()
    if u is None:
        return jsonify({"user": None})
    return jsonify({"user": {"username": u["username"], "role": u["role"],
                             "display_name": u["display_name"],
                             "class_name": u["class_name"]}})


# ---------- 报告模板 ----------
@platform_bp.route("/api/report_template/<exp_id>")
@login_required
def report_template(u, exp_id):
    exp = get_experiment(exp_id)
    if exp is None:
        return jsonify({"error": f"Unknown experiment: {exp_id}"}), 400
    tpl_path = Path(__file__).parent / "report_templates" / f"{exp_id}.json"
    if not tpl_path.exists():
        return jsonify({"error": "该实验暂无报告模板。"}), 404
    with open(tpl_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


# ---------- 报告 CRUD ----------
@platform_bp.route("/api/reports")
@login_required
def reports_list(u):
    if u["role"] == "teacher":
        sql = ("SELECT s.*, us.display_name, us.class_name FROM submissions s "
               "JOIN users us ON us.username = s.username WHERE 1=1")
        args = []
        exp = request.args.get("exp")
        status = request.args.get("status")
        cls = request.args.get("class")
        if exp:
            sql += " AND s.exp_id=?"
            args.append(exp)
        if status:
            sql += " AND s.status=?"
            args.append(status)
        if cls:
            sql += " AND us.class_name=?"
            args.append(cls)
        sql += " ORDER BY s.updated_at DESC"
        with _conn() as c:
            rows = c.execute(sql, args).fetchall()
        out = [_sub_public(r) for r in rows]
        for d, r in zip(out, rows):
            d["display_name"] = r["display_name"]
            d["class_name"] = r["class_name"]
        return jsonify(out)
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM submissions WHERE username=? ORDER BY updated_at DESC",
            (u["username"],)).fetchall()
    return jsonify([_sub_public(r) for r in rows])


@platform_bp.route("/api/reports/<int:rid>")
@login_required
def report_get(u, rid):
    with _conn() as c:
        row = c.execute("SELECT * FROM submissions WHERE id=?", (rid,)).fetchone()
    if row is None:
        return jsonify({"error": "报告不存在。"}), 404
    if u["role"] != "teacher" and row["username"] != u["username"]:
        return jsonify({"error": "无权查看他人报告。"}), 403
    return jsonify(_sub_public(row, with_content=True))


@platform_bp.route("/api/reports", methods=["POST"])
@login_required
def report_save(u):
    if u["role"] != "student":
        return jsonify({"error": "仅学生可编辑报告。"}), 403
    data = request.get_json(silent=True) or {}
    exp_id = data.get("exp_id") or ""
    if get_experiment(exp_id) is None:
        return jsonify({"error": f"Unknown experiment: {exp_id}"}), 400
    content = json.dumps(data.get("content") or {}, ensure_ascii=False)
    rid = data.get("id")
    now = _now()
    with _conn() as c:
        if rid:
            row = c.execute("SELECT * FROM submissions WHERE id=?", (rid,)).fetchone()
            if row is None or row["username"] != u["username"]:
                return jsonify({"error": "报告不存在或无权编辑。"}), 404
            if row["status"] != "draft":
                return jsonify({"error": "已提交的报告不能再修改，如需修改请联系教师退回。"}), 400
            c.execute("UPDATE submissions SET content=?, updated_at=? WHERE id=?",
                      (content, now, rid))
        else:
            cur = c.execute(
                "INSERT INTO submissions(username,exp_id,content,status,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?)",
                (u["username"], exp_id, content, "draft", now, now))
            rid = cur.lastrowid
    return jsonify({"id": rid, "status": "draft"})


@platform_bp.route("/api/reports/<int:rid>/submit", methods=["POST"])
@login_required
def report_submit(u, rid):
    with _conn() as c:
        row = c.execute("SELECT * FROM submissions WHERE id=?", (rid,)).fetchone()
        if row is None or row["username"] != u["username"]:
            return jsonify({"error": "报告不存在。"}), 404
        if row["status"] != "draft":
            return jsonify({"error": "该报告已提交。"}), 400
        c.execute("UPDATE submissions SET status='submitted', updated_at=?, graded_at=NULL,"
                  " score=NULL, feedback='' WHERE id=?", (_now(), rid))
    return jsonify({"ok": True, "status": "submitted"})


@platform_bp.route("/api/reports/<int:rid>", methods=["DELETE"])
@login_required
def report_delete(u, rid):
    """学生删除自己的草稿报告；已提交的需先撤销，已批改的不可删。"""
    with _conn() as c:
        row = c.execute("SELECT * FROM submissions WHERE id=?", (rid,)).fetchone()
        if row is None or row["username"] != u["username"]:
            return jsonify({"error": "报告不存在。"}), 404
        if row["status"] != "draft":
            return jsonify({"error": "仅草稿可以删除；已提交的报告请先撤销提交。"}), 400
        c.execute("DELETE FROM submissions WHERE id=?", (rid,))
    return jsonify({"ok": True})


@platform_bp.route("/api/reports/<int:rid>/withdraw", methods=["POST"])
@login_required
def report_withdraw(u, rid):
    """学生撤销提交：已提交、尚未批改的报告回到草稿状态，可继续修改。"""
    with _conn() as c:
        row = c.execute("SELECT * FROM submissions WHERE id=?", (rid,)).fetchone()
        if row is None or row["username"] != u["username"]:
            return jsonify({"error": "报告不存在。"}), 404
        if row["status"] != "submitted":
            return jsonify({"error": "仅已提交、尚未批改的报告可以撤销。"}), 400
        c.execute("UPDATE submissions SET status='draft', updated_at=? WHERE id=?",
                  (_now(), rid))
    return jsonify({"ok": True, "status": "draft"})


@platform_bp.route("/api/reports/<int:rid>/grade", methods=["POST"])
@teacher_required
def report_grade(u, rid):
    data = request.get_json(silent=True) or {}
    score = data.get("score")
    feedback = (data.get("feedback") or "").strip()
    if score is None or not isinstance(score, (int, float)) or not (0 <= float(score) <= 100):
        return jsonify({"error": "分数须为 0~100 的数字。"}), 400
    with _conn() as c:
        row = c.execute("SELECT * FROM submissions WHERE id=?", (rid,)).fetchone()
        if row is None:
            return jsonify({"error": "报告不存在。"}), 404
        if row["status"] == "draft":
            return jsonify({"error": "学生尚未提交，不能批改。"}), 400
        c.execute("UPDATE submissions SET status='graded', score=?, feedback=?, graded_at=?,"
                  " updated_at=? WHERE id=?",
                  (float(score), feedback, _now(), _now(), rid))
    return jsonify({"ok": True, "status": "graded"})


@platform_bp.route("/api/reports/<int:rid>/return", methods=["POST"])
@teacher_required
def report_return(u, rid):
    """教师退回：状态回到 draft，学生可继续修改。"""
    with _conn() as c:
        row = c.execute("SELECT * FROM submissions WHERE id=?", (rid,)).fetchone()
        if row is None:
            return jsonify({"error": "报告不存在。"}), 404
        c.execute("UPDATE submissions SET status='draft', score=NULL, feedback=?,"
                  " updated_at=? WHERE id=?",
                  (("退回修改：" + (request.get_json(silent=True) or {}).get("feedback", "")).strip(),
                   _now(), rid))
    return jsonify({"ok": True, "status": "draft"})


# ---------- 用户管理（教师） ----------
@platform_bp.route("/api/admin/users", methods=["GET"])
@teacher_required
def users_list(u):
    with _conn() as c:
        rows = c.execute("SELECT username, role, display_name, class_name FROM users"
                         " ORDER BY role DESC, class_name, username").fetchall()
    return jsonify([dict(r) for r in rows])


@platform_bp.route("/api/admin/users", methods=["POST"])
@teacher_required
def users_add(u):
    data = request.get_json(silent=True) or {}
    items = data.get("users") or []
    if not items:
        return jsonify({"error": "请提供用户列表。"}), 400
    added, skipped = [], []
    with _conn() as c:
        for it in items:
            username = (it.get("username") or "").strip()
            if not username:
                continue
            if c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                skipped.append(username)
                continue
            c.execute(
                "INSERT INTO users(username,pw_hash,role,display_name,class_name)"
                " VALUES(?,?,?,?,?)",
                (username, generate_password_hash(it.get("password") or "123456"),
                 "student", it.get("display_name") or username,
                 it.get("class_name") or ""))
            added.append(username)
    return jsonify({"added": added, "skipped": skipped})


# ---------- 班级列表（教师） ----------
@platform_bp.route("/api/classes")
@teacher_required
def classes_list(u):
    with _conn() as c:
        rows = c.execute(
            "SELECT class_name AS name, COUNT(*) AS count FROM users"
            " WHERE role='student' AND class_name!=''"
            " GROUP BY class_name ORDER BY class_name").fetchall()
    return jsonify([dict(r) for r in rows])


# ---------- 实验课全流程：预习 / 课中进度 ----------
def _ensure_progress_row(c, username, exp_id):
    if c.execute("SELECT 1 FROM progress WHERE username=? AND exp_id=?",
                 (username, exp_id)).fetchone() is None:
        c.execute("INSERT INTO progress(username,exp_id,pre_done,lab_done)"
                  " VALUES(?,?,0,0)", (username, exp_id))


@platform_bp.route("/api/progress/pre", methods=["POST"])
@login_required
def progress_pre(u):
    if u["role"] != "student":
        return jsonify({"error": "仅学生可标记预习。"}), 403
    data = request.get_json(silent=True) or {}
    exp_id = data.get("exp_id") or ""
    if get_experiment(exp_id) is None:
        return jsonify({"error": f"Unknown experiment: {exp_id}"}), 400
    now = _now()
    with _conn() as c:
        _ensure_progress_row(c, u["username"], exp_id)
        c.execute("UPDATE progress SET pre_done=1, pre_at=? WHERE username=? AND exp_id=?",
                  (now, u["username"], exp_id))
    return jsonify({"ok": True, "pre_done": 1})


@platform_bp.route("/api/progress/lab", methods=["POST"])
@login_required
def progress_lab(u):
    if u["role"] != "student":
        return jsonify({"error": "仅学生可标记课中实验。"}), 403
    data = request.get_json(silent=True) or {}
    exp_id = data.get("exp_id") or ""
    if get_experiment(exp_id) is None:
        return jsonify({"error": f"Unknown experiment: {exp_id}"}), 400
    now = _now()
    with _conn() as c:
        _ensure_progress_row(c, u["username"], exp_id)
        c.execute("UPDATE progress SET lab_done=1, lab_at=? WHERE username=? AND exp_id=?",
                  (now, u["username"], exp_id))
    return jsonify({"ok": True, "lab_done": 1})


@platform_bp.route("/api/flow")
@login_required
def flow_overview(u):
    """学生：每个实验的 预习/课中/报告 四阶段状态；教师：全部学生聚合。"""
    if u["role"] == "teacher":
        cls = request.args.get("class") or ""
        sql = ("SELECT p.username, us.display_name, us.class_name, p.exp_id,"
               " p.pre_done, p.lab_done, s.status AS rep_status, s.score"
               " FROM progress p"
               " JOIN users us ON us.username=p.username"
               " LEFT JOIN submissions s ON s.username=p.username AND s.exp_id=p.exp_id")
        args = []
        if cls:
            sql += " WHERE us.class_name=?"
            args.append(cls)
        with _conn() as c:
            rows = c.execute(sql, args).fetchall()
        return jsonify([dict(r) for r in rows])
    with _conn() as c:
        prog = {r["exp_id"]: r for r in c.execute(
            "SELECT * FROM progress WHERE username=?", (u["username"],)).fetchall()}
        subs = {r["exp_id"]: r for r in c.execute(
            "SELECT * FROM submissions WHERE username=?", (u["username"],)).fetchall()}
    out = []
    for e in EXPERIMENTS:
        p = prog.get(e["id"])
        s = subs.get(e["id"])
        out.append({
            "exp_id": e["id"], "exp_name": e["name"], "mode": e["mode"],
            "pre_done": 1 if p and p["pre_done"] else 0,
            "pre_at": (p["pre_at"] if p else None),
            "lab_done": 1 if p and p["lab_done"] else 0,
            "lab_at": (p["lab_at"] if p else None),
            "report_status": (s["status"] if s else "none"),
            "report_id": (s["id"] if s else None),
            "score": (s["score"] if s else None),
        })
    return jsonify(out)


# ---------- 师生问答（公共讨论区：全员可见，师生均可回复） ----------
@platform_bp.route("/api/qas")
@login_required
def qa_list(u):
    """公共问答区：所有登录用户可见全部问题；scope=mine 只看自己的提问。"""
    scope = request.args.get("scope", "all")
    pending_only = request.args.get("pending") == "1"
    with _conn() as c:
        if scope == "all":
            sql = ("SELECT q.*, us.display_name AS q_name, us.class_name AS q_class,"
                   " (SELECT COUNT(*) FROM replies r WHERE r.qid=q.id) AS reply_count,"
                   " (SELECT MAX(r.created_at) FROM replies r WHERE r.qid=q.id) AS last_reply_at,"
                   " (SELECT r.username FROM replies r WHERE r.qid=q.id"
                   "  ORDER BY r.id DESC LIMIT 1) AS last_reply_by"
                   " FROM questions q JOIN users us ON us.username=q.username")
            args = []
            if pending_only:
                sql += " WHERE (SELECT COUNT(*) FROM replies r WHERE r.qid=q.id)=0"
            sql += " ORDER BY q.id DESC"
            rows = c.execute(sql, args).fetchall()
        else:
            rows = c.execute(
                "SELECT q.*, us.display_name AS q_name, us.class_name AS q_class,"
                " (SELECT COUNT(*) FROM replies r WHERE r.qid=q.id) AS reply_count,"
                " (SELECT MAX(r.created_at) FROM replies r WHERE r.qid=q.id) AS last_reply_at,"
                " (SELECT r.username FROM replies r WHERE r.qid=q.id"
                "  ORDER BY r.id DESC LIMIT 1) AS last_reply_by"
                " FROM questions q JOIN users us ON us.username=q.username"
                " WHERE q.username=? ORDER BY q.id DESC", (u["username"],)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        exp = get_experiment(d["exp_id"]) if d["exp_id"] else None
        d["exp_name"] = exp["name"] if exp else ""
        d["attachments"] = _attachments_of("question", d["id"])
        out.append(d)
    return jsonify(out)


@platform_bp.route("/api/qas", methods=["POST"])
@login_required
def qa_create(u):
    if u["role"] != "student":
        return jsonify({"error": "仅学生可发起提问。"}), 403
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    exp_id = data.get("exp_id") or ""
    if exp_id and get_experiment(exp_id) is None:
        return jsonify({"error": f"Unknown experiment: {exp_id}"}), 400
    if not title and not body and not data.get("attachments"):
        return jsonify({"error": "请填写问题内容或添加图片/语音附件。"}), 400
    now = _now()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO questions(username,exp_id,title,body,created_at)"
            " VALUES(?,?,?,?,?)",
            (u["username"], exp_id, title[:200], body, now))
        qid = cur.lastrowid
    _bind_attachments(data.get("attachments") or [], "question", qid)
    return jsonify({"ok": True, "id": qid})


@platform_bp.route("/api/qas/<int:qid>")
@login_required
def qa_get(u, qid):
    """问题详情 + 回复线程（公共讨论区：所有登录用户可见）。"""
    with _conn() as c:
        q = c.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        if q is None:
            return jsonify({"error": "问题不存在。"}), 404
        repl = c.execute("SELECT * FROM replies WHERE qid=? ORDER BY id", (qid,)).fetchall()
    d = dict(q)
    exp = get_experiment(d["exp_id"]) if d["exp_id"] else None
    d["exp_name"] = exp["name"] if exp else ""
    d["q_user"] = _user_brief(d["username"])
    d["attachments"] = _attachments_of("question", qid)
    d["replies"] = []
    for r in repl:
        rd = dict(r)
        rd["user"] = _user_brief(r["username"])
        rd["attachments"] = _attachments_of("reply", r["id"])
        d["replies"].append(rd)
    return jsonify(d)


@platform_bp.route("/api/qas/<int:qid>/reply", methods=["POST"])
@login_required
def qa_reply(u, qid):
    """回复提问（公共讨论区：教师和所有学生均可回复）。"""
    data = request.get_json(silent=True) or {}
    body = (data.get("body") or "").strip()
    if not body and not data.get("attachments"):
        return jsonify({"error": "回复内容不能为空。"}), 400
    with _conn() as c:
        q = c.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        if q is None:
            return jsonify({"error": "问题不存在。"}), 404
        cur = c.execute("INSERT INTO replies(qid,username,body,created_at)"
                        " VALUES(?,?,?,?)", (qid, u["username"], body, _now()))
        rid = cur.lastrowid
    _bind_attachments(data.get("attachments") or [], "reply", rid)
    return jsonify({"ok": True, "id": rid})


def _delete_attachments(c, owner_kind, owner_id):
    """删除某问答/回复名下的附件记录与文件（尽力而为），返回被删附件数。"""
    rows = c.execute("SELECT id, filename FROM attachments WHERE owner_kind=? AND owner_id=?",
                     (owner_kind, owner_id)).fetchall()
    for r in rows:
        c.execute("DELETE FROM attachments WHERE id=?", (r["id"],))
        try:
            (Path(_qa_dir) / r["filename"]).unlink(missing_ok=True)
        except OSError:
            pass
    return len(rows)


@platform_bp.route("/api/qas/<int:qid>", methods=["DELETE"])
@login_required
def qa_delete(u, qid):
    """撤回提问：仅提问者本人可撤，连同全部回复与附件一并删除。"""
    with _conn() as c:
        q = c.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        if q is None:
            return jsonify({"error": "问题不存在。"}), 404
        if q["username"] != u["username"]:
            return jsonify({"error": "只能撤回自己的提问。"}), 403
        for r in c.execute("SELECT id FROM replies WHERE qid=?", (qid,)).fetchall():
            _delete_attachments(c, "reply", r["id"])
        _delete_attachments(c, "question", qid)
        c.execute("DELETE FROM replies WHERE qid=?", (qid,))
        c.execute("DELETE FROM questions WHERE id=?", (qid,))
    return jsonify({"ok": True})


@platform_bp.route("/api/qas/reply/<int:rid>", methods=["DELETE"])
@login_required
def qa_reply_delete(u, rid):
    """撤回回复：仅回复者本人（教师或学生）可撤。"""
    with _conn() as c:
        row = c.execute("SELECT r.*, q.id AS qid FROM replies r JOIN questions q ON q.id=r.qid"
                        " WHERE r.id=?", (rid,)).fetchone()
        if row is None:
            return jsonify({"error": "回复不存在。"}), 404
        if row["username"] != u["username"]:
            return jsonify({"error": "只能撤回自己的回复。"}), 403
        _delete_attachments(c, "reply", rid)
        c.execute("DELETE FROM replies WHERE id=?", (rid,))
    return jsonify({"ok": True})


# ---------- 附件上传与访问 ----------
@platform_bp.route("/api/qas/attachment", methods=["POST"])
@login_required
def qa_attachment_upload(u):
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "未收到文件。"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in _ALLOWED_EXTS:
        return jsonify({"error": "不支持的文件类型（仅图片 / 音频 / 文档）。"}), 400
    if ext in _IMAGE_EXTS:
        kind = "image"
    elif ext in _AUDIO_EXTS:
        kind = "audio"
    else:
        kind = "file"
    f.stream.seek(0, os.SEEK_END)
    size = f.stream.tell()
    f.stream.seek(0)
    if size > _SIZE_LIMITS[kind]:
        return jsonify({"error": f"{kind} 文件过大（上限 {_SIZE_LIMITS[kind]//1024//1024} MB）。"}), 413
    stored = f"{uuid.uuid4().hex}{ext}"
    (Path(_qa_dir) / stored).write_bytes(f.read())
    now = _now()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO attachments(owner_kind,owner_id,kind,filename,orig_name,mime,size,"
            " created_at) VALUES('temp',0,?,?,?,?,?,?)",
            (kind, stored, f.filename, f.mimetype or "", size, now))
        aid = cur.lastrowid
    return jsonify({"id": aid, "kind": kind, "orig_name": f.filename,
                    "size": size, "url": f"/api/qas/attachment/{aid}/file"})


def _can_access_attachment(username, role, att_row):
    """附件可见性：temp（未绑定）仅存在于上传请求的生命周期内；绑定到问答
    线程后属于公共讨论内容，登录用户均可见。"""
    if att_row["owner_kind"] == "temp":
        return True
    with _conn() as c:
        if att_row["owner_kind"] == "question":
            q = c.execute("SELECT username FROM questions WHERE id=?",
                          (att_row["owner_id"],)).fetchone()
        else:
            q = c.execute(
                "SELECT q.username FROM replies r JOIN questions q ON q.id=r.qid"
                " WHERE r.id=?", (att_row["owner_id"],)).fetchone()
    return q is not None


@platform_bp.route("/api/qas/attachment/<int:aid>/file")
@login_required
def qa_attachment_file(u, aid):
    with _conn() as c:
        row = c.execute("SELECT * FROM attachments WHERE id=?", (aid,)).fetchone()
    if row is None:
        return jsonify({"error": "附件不存在。"}), 404
    if not _can_access_attachment(u["username"], u["role"], row):
        return jsonify({"error": "无权访问该附件。"}), 403
    path = Path(_qa_dir) / row["filename"]
    if not path.exists():
        return jsonify({"error": "附件文件已丢失。"}), 404
    return send_from_directory(
        str(_qa_dir), row["filename"],
        mimetype=row["mime"] or "application/octet-stream",
        as_attachment=False, download_name=row["orig_name"])


# ---------- 数据看板统计 ----------
@platform_bp.route("/api/dashboard/summary")
def dashboard_summary():
    u = current_user()
    with _conn() as c:
        stu_total = c.execute("SELECT COUNT(*) AS n FROM users WHERE role='student'").fetchone()["n"]
        cls_rows = c.execute(
            "SELECT class_name AS name, COUNT(*) AS count FROM users"
            " WHERE role='student' AND class_name!=''"
            " GROUP BY class_name ORDER BY class_name").fetchall()
        rep_total = c.execute("SELECT COUNT(*) AS n FROM submissions").fetchone()["n"]
        sub_total = c.execute("SELECT COUNT(*) AS n FROM submissions"
                              " WHERE status!='draft'").fetchone()["n"]
        graded_total = c.execute("SELECT COUNT(*) AS n FROM submissions"
                                 " WHERE status='graded'").fetchone()["n"]
        pending = sub_total - graded_total
        avg_row = c.execute("SELECT AVG(score) AS a FROM submissions"
                            " WHERE status='graded'").fetchone()["a"]
        avg_score = round(float(avg_row), 1) if avg_row is not None else None
        dist_rows = c.execute(
            "SELECT CASE WHEN score>=90 THEN '90-100' WHEN score>=80 THEN '80-89'"
            " WHEN score>=70 THEN '70-79' WHEN score>=60 THEN '60-69' ELSE '<60' END AS b,"
            " COUNT(*) AS n FROM submissions WHERE status='graded' GROUP BY b").fetchall()
        dist = {r["b"]: r["n"] for r in dist_rows}
        for b in ("90-100", "80-89", "70-79", "60-69", "<60"):
            dist.setdefault(b, 0)
        exp_rows = c.execute(
            "SELECT exp_id, COUNT(*) AS n,"
            " SUM(CASE WHEN status='graded' THEN 1 ELSE 0 END) AS graded"
            " FROM submissions GROUP BY exp_id").fetchall()
        by_exp = []
        exp_names = {e["id"]: e["name"] for e in EXPERIMENTS}
        for r in exp_rows:
            by_exp.append({"exp_id": r["exp_id"],
                           "exp_name": exp_names.get(r["exp_id"], r["exp_id"]),
                           "count": r["n"], "graded": r["graded"] or 0})
        by_exp.sort(key=lambda x: -x["count"])
        pre_done = c.execute("SELECT COUNT(*) AS n FROM progress WHERE pre_done=1").fetchone()["n"]
        lab_done = c.execute("SELECT COUNT(*) AS n FROM progress WHERE lab_done=1").fetchone()["n"]
        prog_total = c.execute("SELECT COUNT(*) AS n FROM progress").fetchone()["n"]
        qa_open = c.execute(
            "SELECT COUNT(*) AS n FROM questions q"
            " WHERE (SELECT COUNT(*) FROM replies r WHERE r.qid=q.id)=0").fetchone()["n"]
        qa_total = c.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]

    base = {
        "students": stu_total,
        "reports": rep_total,
        "submitted": sub_total,
        "graded": graded_total,
        "pending_grading": pending,
        "avg_score": avg_score,
        "score_dist": dist,
        "by_exp": by_exp,
        "pre_done": pre_done,
        "lab_done": lab_done,
        "progress_total": prog_total,
        "qa_open": qa_open,
        "qa_total": qa_total,
        "classes": [dict(r) for r in cls_rows],
        "exp_total": len(EXPERIMENTS),
    }
    if u is None:
        return jsonify({"user": None, **base})

    if u["role"] == "teacher":
        # 班级维度成绩统计（含整体）
        class_stats = []
        # 同一学生×同一实验只按最新一份报告计入班级统计（重复提交不灌水分母），
        # 批改率在服务端算好下发：已批改报告 / 已提交报告
        dedup_sub = ("SELECT s.* FROM submissions s JOIN"
                     " (SELECT username, exp_id, MAX(id) AS mx FROM submissions"
                     "  GROUP BY username, exp_id) m ON s.id=m.mx")
        with _conn() as c:
            for cr in cls_rows:
                name = cr["name"]
                agg = c.execute(
                    "SELECT COUNT(DISTINCT us.username) AS stu,"
                    " COUNT(DISTINCT CASE WHEN s.status!='draft' THEN us.username END) AS sub_stu,"
                    " COUNT(s.id) AS reports,"
                    " SUM(CASE WHEN s.status!='draft' THEN 1 ELSE 0 END) AS sub,"
                    " SUM(CASE WHEN s.status='graded' THEN 1 ELSE 0 END) AS graded,"
                    " AVG(CASE WHEN s.status='graded' THEN s.score END) AS avg"
                    f" FROM users us LEFT JOIN ({dedup_sub}) s ON s.username=us.username"
                    " WHERE us.role='student' AND us.class_name=?", (name,)).fetchone()
                sub = agg["sub"] or 0
                graded = agg["graded"] or 0
                class_stats.append({
                    "name": name, "students": agg["stu"],
                    "sub_students": agg["sub_stu"] or 0,
                    "reports": agg["reports"] or 0,
                    "submitted": sub, "graded": graded,
                    "grade_rate": (round(graded / sub * 100) if sub else 0),
                    "avg_score": round(float(agg["avg"]), 1) if agg["avg"] is not None else None,
                })
        recent = []
        with _conn() as c:
            rows = c.execute(
                "SELECT s.id,s.username,s.exp_id,s.status,s.score,s.updated_at,"
                " us.display_name,us.class_name FROM submissions s"
                " JOIN users us ON us.username=s.username"
                " WHERE s.status!='draft' ORDER BY s.updated_at DESC LIMIT 8").fetchall()
            recent = [{"id": r["id"], "username": r["username"],
                       "display_name": r["display_name"], "class_name": r["class_name"],
                       "exp_id": r["exp_id"],
                       "exp_name": exp_names.get(r["exp_id"], r["exp_id"]),
                       "status": r["status"], "score": r["score"],
                       "updated_at": r["updated_at"]} for r in rows]
        return jsonify({"user": {"username": u["username"], "role": "teacher",
                                 "display_name": u["display_name"]},
                        **base, "class_stats": class_stats, "recent": recent})

    # 学生视图
    with _conn() as c:
        my_pre = c.execute("SELECT COUNT(*) AS n FROM progress WHERE username=? AND pre_done=1",
                           (u["username"],)).fetchone()["n"]
        my_lab = c.execute("SELECT COUNT(*) AS n FROM progress WHERE username=? AND lab_done=1",
                           (u["username"],)).fetchone()["n"]
        my_graded = c.execute("SELECT COUNT(*) AS n FROM submissions WHERE username=?"
                              " AND status='graded'", (u["username"],)).fetchone()["n"]
        my_submitted = c.execute("SELECT COUNT(*) AS n FROM submissions WHERE username=?"
                                 " AND status IN ('submitted','graded')",
                                 (u["username"],)).fetchone()["n"]
        my_avg = c.execute("SELECT AVG(score) AS a FROM submissions WHERE username=?"
                           " AND status='graded'", (u["username"],)).fetchone()["a"]
        my_qa = c.execute(
            "SELECT COUNT(*) AS n FROM questions q WHERE q.username=?"
            " AND (SELECT COUNT(*) FROM replies r WHERE r.qid=q.id)>0",
            (u["username"],)).fetchone()["n"]
        my_qa_pending = c.execute("SELECT COUNT(*) AS n FROM questions q WHERE q.username=?"
                                  " AND (SELECT COUNT(*) FROM replies r WHERE r.qid=q.id)=0",
                                  (u["username"],)).fetchone()["n"]
    my = {
        "pre_done": my_pre, "lab_done": my_lab,
        "submitted": my_submitted, "graded": my_graded,
        "avg_score": round(float(my_avg), 1) if my_avg is not None else None,
        "qa_replied": my_qa, "qa_pending": my_qa_pending,
        "class_name": u["class_name"],
    }
    return jsonify({"user": {"username": u["username"], "role": "student",
                             "display_name": u["display_name"],
                             "class_name": u["class_name"]},
                    **base, "my": my})
