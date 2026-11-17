# -*- coding: utf-8 -*-
"""
前端截图回归工具（无头 Chrome + CDP，无需 Playwright）
=====================================================
用法：
    python tools/ui_shots.py                 # 全量：未登录 / teacher / student 各视图
    python tools/ui_shots.py --only result   # 只截结果页（读 tools/fixtures/result_<type>.json）
    python tools/ui_shots.py --size 1366x768 # 指定视口

说明：
- 若 127.0.0.1:5000 未运行（例如被看门狗自动关停），本脚本会自动在后台拉起 app.py，
  等待就绪后再截图，并在结束时清理临时拉起的进程。
- 产出：tools/_shots/*.png（已 gitignore）。截完务必打开图片目视核对。
"""
import argparse, base64, glob, json, os, subprocess, sys, tempfile, time
import requests, websocket

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "_shots")
BASE = "http://127.0.0.1:5000"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def ensure_server():
    """检查 5000 端口是否就绪；未启动则自动拉起 app.py 并等待就绪。"""
    try:
        r = requests.get(BASE + "/login", timeout=1.5)
        if r.status_code == 200:
            return None
    except Exception:
        pass
    print("[ui_shots] 5000 端口未就绪，正在后台启动 app.py ...")
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            r = requests.get(BASE + "/login", timeout=1.5)
            if r.status_code == 200:
                print("[ui_shots] app.py 已就绪")
                return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    sys.exit("[ui_shots] 启动 app.py 超时（30s）")


class Page:
    def __init__(self, w, h, port=9333):
        exe = next((b for b in BROWSERS if os.path.exists(b)), None)
        if not exe:
            sys.exit("未找到 Chrome / Edge")
        self.proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={port}",
                                      f"--user-data-dir={tempfile.mkdtemp()}", f"--window-size={w},{h}",
                                      "--hide-scrollbars", "about:blank"])
        tabs = None
        for _ in range(60):
            try:
                tabs = requests.get(f"http://127.0.0.1:{port}/json").json(); break
            except Exception:
                time.sleep(0.2)
        url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
        self.ws = websocket.create_connection(url, timeout=60)
        self.mid = 0
        self.cmd("Page.enable")
        self.cmd("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)

    def cmd(self, method, **params):
        self.mid += 1
        self.ws.send(json.dumps({"id": self.mid, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.mid:
                return m.get("result", m)

    def ev(self, js):
        r = self.cmd("Runtime.evaluate", expression=js, awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"):
            print("  [JS 异常]", r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r

    def nav(self, url, wait=2.5):
        self.cmd("Page.navigate", url=url); time.sleep(wait)

    def shot(self, name, wait=1.5):
        time.sleep(wait)
        data = self.cmd("Page.captureScreenshot", format="png")["data"]
        with open(os.path.join(OUT, name + ".png"), "wb") as f:
            f.write(base64.b64decode(data))
        print("saved", name)

    def login(self, user):
        self.nav(BASE + "/login", 1.5)
        self.ev(f"fetch('/api/auth/login',{{method:'POST',headers:{{'Content-Type':'application/json'}},"
                f"body:JSON.stringify({{username:'{user}',password:'123456'}})}}).then(r=>r.status)")
        self.nav(BASE + "/", 3)

    def logout(self):
        self.ev("fetch('/api/auth/logout',{method:'POST'})"); time.sleep(0.5)

    def close(self):
        try: self.ws.close()
        finally: self.proc.terminate()


def full(p):
    p.nav(BASE + "/login"); p.shot("00_login")
    p.nav(BASE + "/"); p.shot("01_home_public")
    for role in ("teacher", "student"):
        p.login(role)
        p.shot(f"{role}_10_home", 2)
        p.ev("switchView('analysis')"); p.shot(f"{role}_20_wz1")
        p.ev("selectType('niubai')"); p.shot(f"{role}_21_wz1_selected")
        p.ev("goToPage(2)"); p.shot(f"{role}_22_wz2")
        p.ev("selectAcqMode('experiment')"); p.shot(f"{role}_23_wz2_exp")
        p.ev("goToPage(3)"); p.shot(f"{role}_24_wz3")
        p.ev("goToPage(4)"); p.shot(f"{role}_25_wz4")
        p.ev("switchView('guide');loadGuide('niubai')"); p.shot(f"{role}_30_guide_video", 3)
        p.ev("switchView('guide');loadGuide('yangshi')"); p.shot(f"{role}_31_guide_other", 3)
        p.ev("switchView('reports')"); p.shot(f"{role}_40_reports", 2.5)
        if role == "student":
            p.ev("openReportEditor(null,'niubai')"); p.shot(f"{role}_41_report_edit", 2.5)
        else:
            p.ev("document.querySelector('#grading_list_area .list-row')&&document.querySelector('#grading_list_area .list-row').click()"); p.shot(f"{role}_41_grading_detail", 2.5)
        p.ev("switchView('qas')"); p.shot(f"{role}_50_qas", 2.5)
        p.logout()


def _demo_plot_b64():
    """为不含 base64 大图的轻量 fixture 动态生成一张阻尼振荡示意 PNG。"""
    try:
        import io, numpy as np, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.linspace(0, 20, 400)
        y = 35 * np.exp(-0.08 * t) * np.cos(2.6 * t)
        fig, ax = plt.subplots(figsize=(8, 2.8), dpi=120)
        ax.plot(t, y, color="#c2410c", lw=1.4, label="Angle (deg)")
        ax.plot(t, 35 * np.exp(-0.08 * t), "--", color="#57534e", lw=1, label="Envelope")
        ax.plot(t, -35 * np.exp(-0.08 * t), "--", color="#57534e", lw=1)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def result(p):
    files = sorted(glob.glob(os.path.join(HERE, "fixtures", "result_*.json")))
    if not files:
        print("tools/fixtures/ 下没有 result_<type>.json，跳过结果页"); return
    b64 = _demo_plot_b64()
    p.login("student")
    for fp in files:
        tag = os.path.basename(fp)[len("result_"):-len(".json")]
        with open(fp, encoding="utf-8") as f:
            obj = json.load(f)
        typ = obj.get("motion_type") or tag
        res = obj.setdefault("result", {})
        if b64 and not res.get("plot_preview"):
            res["plot_preview"] = b64
        sr = res.get("sr_result")
        if b64 and isinstance(sr, dict) and not sr.get("error") and not sr.get("plot_preview"):
            sr["plot_preview"] = b64
        payload = json.dumps(obj, ensure_ascii=False)
        p.ev("switchView('analysis')")
        p.ev(f"window.__renderResultFixture?window.__renderResultFixture({payload},'{typ}'):console.warn('no fixture hook')")
        p.shot(f"result_{tag}", 2.5)
    p.logout()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["full", "result"], default=None)
    ap.add_argument("--size", default="1600x1000")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    srv = ensure_server()
    w, h = map(int, a.size.lower().split("x"))
    page = Page(w, h)
    try:
        if a.only in (None, "full"): full(page)
        if a.only in (None, "result"): result(page)
    finally:
        page.close()
        if srv is not None:
            srv.terminate()
