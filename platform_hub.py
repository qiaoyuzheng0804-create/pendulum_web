# -*- coding: utf-8 -*-
"""
师生交互平台（Blueprint）
=========================
登录认证（Flask session cookie）+ 实验报告的提交与批改（SQLite，零外部依赖）。

表：
  users       —— 用户名 / 口令哈希 / 角色(student|teacher) / 姓名
  submissions —— 报告：学生、实验、分节 Markdown 内容(JSON)、状态、分数、评语
首次运行自动建库并预置演示账号：teacher / 123456（教师）、student / 123456（学生）。
"""
import json
import sqlite3
from datetime import datetime
from functools import wraps

from flask import Blueprint, render_template, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash

from experiments import get_experiment, EXPERIMENTS

BASE_DIR_MARKER = "platform"
_db_path = None  # 由 init_platform(app) 注入


# ---------- DB ----------
def _conn():
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_platform(app, db_path):
    """建表 + 预置演示账号（幂等）。"""
    global _db_path
    _db_path = str(db_path)
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('student','teacher')),
                display_name TEXT NOT NULL DEFAULT ''
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
            CREATE INDEX IF NOT EXISTS idx_sub_user ON submissions(username);
            CREATE INDEX IF NOT EXISTS idx_sub_exp ON submissions(exp_id);
            """
        )
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
                    "display_name": row["display_name"]})


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
                             "display_name": u["display_name"]}})


# ---------- 报告模板 ----------
@platform_bp.route("/api/report_template/<exp_id>")
@login_required
def report_template(u, exp_id):
    exp = get_experiment(exp_id)
    if exp is None:
        return jsonify({"error": f"Unknown experiment: {exp_id}"}), 400
    import os
    from pathlib import Path
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
        sql = ("SELECT s.*, us.display_name FROM submissions s "
               "JOIN users us ON us.username = s.username WHERE 1=1")
        args = []
        exp = request.args.get("exp")
        status = request.args.get("status")
        if exp:
            sql += " AND s.exp_id=?"
            args.append(exp)
        if status:
            sql += " AND s.status=?"
            args.append(status)
        sql += " ORDER BY s.updated_at DESC"
        with _conn() as c:
            rows = c.execute(sql, args).fetchall()
        out = [_sub_public(r) for r in rows]
        for d, r in zip(out, rows):
            d["display_name"] = r["display_name"]
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
        rows = c.execute("SELECT username, role, display_name FROM users"
                         " ORDER BY role DESC, username").fetchall()
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
                "INSERT INTO users(username,pw_hash,role,display_name) VALUES(?,?,?,?)",
                (username, generate_password_hash(it.get("password") or "123456"),
                 "student", it.get("display_name") or username))
            added.append(username)
    return jsonify({"added": added, "skipped": skipped})
