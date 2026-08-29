"""
符号回归拟合处理器 — 纯 scipy 实现，不依赖 PySR/Julia
"""
import os, math, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import odeint
from sklearn.metrics import r2_score

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# ============================================================
# 单摆：振幅包络提取 + 指数衰减拟合
# ============================================================
def run_sr_danbai(csv_path, output_dir, pendulum_length=0.5):
    try:
        df = pd.read_csv(csv_path)
        tc, ac = df.columns[0], df.columns[1]
        t = df[tc].values.astype(float)
        angle = np.abs(df[ac].values.astype(float))

        peaks, pt = [], []
        for i in range(2, len(angle)-2):
            if angle[i] > angle[i-1] and angle[i] > angle[i-2] and angle[i] > angle[i+1] and angle[i] > angle[i+2]:
                if angle[i] > 0.1 * np.max(angle):
                    peaks.append(angle[i]); pt.append(t[i])
        if len(peaks) < 4:
            return {"error": "振幅峰值不足，无法拟合阻尼"}

        peaks, pt = np.array(peaks), np.array(pt)
        g = 9.8
        omega0 = math.sqrt(g / pendulum_length)
        popt, _ = curve_fit(lambda t, A0, b: A0*np.exp(-b*t), pt, peaks, p0=[peaks[0], 0.01], maxfev=10000)
        A0, beta = popt
        zeta = beta / omega0
        r2 = r2_score(peaks, A0*np.exp(-beta*pt))

        plot_path = os.path.join(output_dir, "sr_damping_fit.png")
        fig, ax = plt.subplots(1,2,figsize=(12,5))
        ax[0].plot(t, angle, alpha=0.4); ax[0].scatter(pt, peaks, c='red', s=20)
        tf = np.linspace(pt[0], pt[-1], 200)
        ax[0].plot(tf, A0*np.exp(-beta*tf), 'r-', lw=2, label=f'beta={beta:.6f}')
        ax[0].legend(); ax[0].set_xlabel("Time"); ax[0].set_ylabel("Amplitude"); ax[0].grid(alpha=0.3)
        ax[1].scatter(pt, np.log(peaks), c='green'); ax[1].plot(tf, np.log(A0)-beta*tf, 'r-')
        ax[1].set_xlabel("Time"); ax[1].set_ylabel("ln(Amplitude)"); ax[1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()

        return {
            "beta": float(beta), "omega0": float(omega0), "zeta": float(zeta),
            "damping_type": "欠阻尼" if zeta<0.1 else ("欠阻尼" if zeta<1 else "临界阻尼" if abs(zeta-1)<0.05 else "过阻尼"),
            "r2": float(r2), "plot_path": plot_path, "pendulum_length": pendulum_length,
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 磁阻尼摆 / 扭摆：最小二乘欠阻尼/过阻尼拟合 + ODE验证
# ============================================================
def _library_fit(t, y, theta_eq):
    """扩展物理项库拟合：alpha = a*q + b*v + c*q³ + d*v|v| + e"""
    dt = np.median(np.diff(t))
    v = np.gradient(y, dt)
    alpha = np.gradient(v, dt)
    q = y - theta_eq
    X = np.column_stack([q, v, q**3, v*np.abs(v), np.ones_like(q)])
    coef, _, _, _ = np.linalg.lstsq(X, alpha, rcond=None)
    a, b, c3, d2, e = coef
    pred = X @ coef
    r2 = r2_score(alpha, pred)
    omega0 = math.sqrt(-a) if a < 0 else None
    beta_lib = -b/2 if b < 0 else None
    zeta_lib = beta_lib/omega0 if (omega0 and beta_lib) else None
    return {"a_q": float(a), "b_v": float(b), "c_q3": float(c3), "d_vabs": float(d2), "e_const": float(e),
            "omega0": omega0, "beta": beta_lib, "zeta": zeta_lib, "accel_r2": float(r2)}

def _fit_damped_oscillation(csv_path, output_dir, damping_subtype="欠阻尼"):
    """
    欠阻尼: theta = theta_eq + exp(-beta*t) * [A*cos(omega_d*t) + B*sin(omega_d*t)]
    过阻尼: theta = theta_eq + C1*exp(-lam1*t) + C2*exp(-lam2*t)
    临界阻尼: theta = theta_eq + (A + B*t)*exp(-lam*t)
    """
    try:
        df = pd.read_csv(csv_path)
        tc, ac = df.columns[0], df.columns[1]
        t_raw = df[tc].values.astype(float)
        theta_raw = np.deg2rad(df[ac].values.astype(float))

        # Skip initial quiet period
        n = len(theta_raw)
        start = 0
        m0 = np.mean(theta_raw[:max(3, n//20)])
        for i in range(n):
            if abs(theta_raw[i] - m0) > np.deg2rad(0.3):
                start = i; break
        t = t_raw[start:] - t_raw[start]
        theta = theta_raw[start:]

        theta_eq = np.mean(theta[-max(3, n//10):])

        if damping_subtype in ("欠阻尼",):
            return _fit_underdamped(t, theta, theta_eq, output_dir)
        elif damping_subtype in ("过阻尼",):
            return _fit_overdamped(t, theta, theta_eq, output_dir)
        else:  # 临界阻尼
            return _fit_critical(t, theta, theta_eq, output_dir)
    except Exception as e:
        return {"error": str(e)}

def _fit_underdamped(t, y, theta_eq0, output_dir):
    def model(p):
        theta_eq, A, B, omega0, beta = p
        if omega0<=0 or beta<0 or beta>=omega0: return np.full_like(t, 1e6)
        od = math.sqrt(omega0**2 - beta**2)
        return theta_eq + np.exp(-beta*t)*(A*np.cos(od*t) + B*np.sin(od*t))
    def res(p): return model(p) - y

    A0 = y[0] - theta_eq0
    best, best_cost = None, np.inf
    for o0 in [5, 10, 15, 20, 30, 50]:
        for b0 in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            if b0 >= o0: continue
            try:
                r = least_squares(res, [theta_eq0, A0, 0.0, o0, b0],
                    bounds=([-np.inf,-np.inf,-np.inf,1e-6,0],[np.inf,np.inf,np.inf,500,500]),
                    loss="soft_l1", max_nfev=10000)
                c = np.sum(res(r.x)**2)
                if c < best_cost: best_cost, best = c, r
            except Exception as _fit_exc: print(f'[SR] 拟合初值组合失败: {_fit_exc}', flush=True)
    if best is None: return {"error": "欠阻尼拟合失败"}

    theta_eq, A, B, omega0, beta = best.x
    omega_d = math.sqrt(max(omega0**2-beta**2, 0))
    zeta = beta/omega0
    pred = model(best.x)
    r2 = r2_score(y, pred)

    # ODE validation
    def ode(state, t):
        th, v = state; q = th - theta_eq
        return [v, -omega0**2*q - 2*beta*v]
    sol = odeint(ode, [y[0], (y[1]-y[0])/(t[1]-t[0]) if len(t)>1 else 0], t)
    ode_r2 = r2_score(y, sol[:,0])
    ode_rmse = math.sqrt(np.mean((y-sol[:,0])**2))

    lib = _library_fit(t, y, theta_eq)
    plot_path = os.path.join(output_dir, "sr_analysis.png")
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    ax[0].plot(t, np.rad2deg(y), alpha=0.4, label="Data"); ax[0].plot(t, np.rad2deg(pred), 'r-', lw=1.5, label="Fit")
    ax[0].plot(t, np.rad2deg(sol[:,0]), 'g--', lw=1.5, label="ODE"); ax[0].legend()
    ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Angle (deg)"); ax[0].grid(alpha=0.3)
    ax[1].plot(np.rad2deg(y-theta_eq), np.gradient(y)/np.max(np.abs(np.gradient(y))), lw=1)
    ax[1].set_xlabel("q (deg)"); ax[1].set_ylabel("v (norm)"); ax[1].set_title("Phase Portrait"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()

    result = {
        "beta": float(beta), "omega0": float(omega0), "zeta": float(zeta), "omega_d": float(omega_d),
        "theta_eq_deg": float(np.rad2deg(theta_eq)),
        "damping_type": "欠阻尼" if zeta<1 else "临界阻尼",
        "ode_r2": float(ode_r2), "ode_rmse_deg": float(np.rad2deg(ode_rmse)),
        "r2_fit": float(r2), "plot_path": plot_path,
        "library_coef": lib, "final_source": "欠阻尼解析解时域拟合 + ODE验证",
    }
    return result

def _fit_critical(t, y, theta_eq0, output_dir):
    def model(p):
        theta_eq, A, B, lam = p
        if lam<=0: return np.full_like(t,1e6)
        return theta_eq + (A+B*t)*np.exp(-lam*t)
    def res(p): return model(p)-y

    A0 = y[0]-theta_eq0
    best, best_cost = None, np.inf
    for lam0 in [0.5, 1, 2, 5, 8, 10, 15]:
        for B0 in [-lam0*A0, 0, lam0*A0]:
            try:
                r = least_squares(res, [theta_eq0, A0, B0, lam0],
                    bounds=([-np.inf,-np.inf,-np.inf,1e-8],[np.inf,np.inf,np.inf,500]),
                    loss="soft_l1", max_nfev=10000)
                c = np.sum(res(r.x)**2)
                if c < best_cost: best_cost, best = c, r
            except Exception as _fit_exc: print(f'[SR] 拟合初值组合失败: {_fit_exc}', flush=True)
    if best is None: return {"error":"临界阻尼拟合失败"}

    theta_eq, A, B, lam = best.x
    pred = model(best.x)
    r2 = r2_score(y, pred)
    beta = lam; omega0 = lam; zeta = 1.0

    def ode(state, t):
        th, v = state; q = th - theta_eq
        return [v, -lam**2*q - 2*lam*v]
    sol = odeint(ode, [y[0], (y[1]-y[0])/(t[1]-t[0]) if len(t)>1 else 0], t)
    ode_r2 = r2_score(y, sol[:,0])
    ode_rmse = math.sqrt(np.mean((y-sol[:,0])**2))

    lib = _library_fit(t, y, theta_eq)
    plot_path = os.path.join(output_dir, "sr_analysis.png")
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    ax[0].plot(t, np.rad2deg(y), alpha=0.4, label="Data"); ax[0].plot(t, np.rad2deg(pred), 'r-', lw=1.5, label="Fit")
    ax[0].plot(t, np.rad2deg(sol[:,0]), 'g--', lw=1.5, label="ODE"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(np.rad2deg(y-theta_eq), np.gradient(y)/np.max(np.abs(np.gradient(y))), lw=1)
    ax[1].set_title("Phase"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()

    result = {
        "beta": float(beta), "omega0": float(omega0), "zeta": float(zeta),
        "theta_eq_deg": float(np.rad2deg(theta_eq)),
        "damping_type": "临界阻尼",
        "ode_r2": float(ode_r2), "ode_rmse_deg": float(np.rad2deg(ode_rmse)),
        "r2_fit": float(r2), "plot_path": plot_path,
        "library_coef": lib, "final_source": "临界阻尼解析解时域拟合 + ODE验证",
    }
    return result

def _fit_overdamped(t, y, theta_eq0, output_dir):
    def model(p):
        theta_eq, C1, C2, lam1, lam2 = p
        if lam1<=0 or lam2<=0: return np.full_like(t,1e6)
        return theta_eq + C1*np.exp(-lam1*t) + C2*np.exp(-lam2*t)
    def res(p): return model(p)-y

    A0 = y[0]-theta_eq0
    best, best_cost = None, np.inf
    pairs = [(0.2,5),(0.5,10),(1,20),(2,30),(0.5,5),(1,10),(2,20),(5,50),(8,9)]
    for l1,l2 in pairs:
        try:
            r = least_squares(res, [theta_eq0, 0.8*A0, 0.2*A0, l1, l2],
                bounds=([-np.inf,-np.inf,-np.inf,1e-8,1e-8],[np.inf,np.inf,np.inf,500,500]),
                loss="soft_l1", max_nfev=10000)
            c = np.sum(res(r.x)**2)
            if c < best_cost: best_cost, best = c, r
        except Exception as _fit_exc: print(f'[SR] 拟合初值组合失败: {_fit_exc}', flush=True)
    if best is None: return {"error":"过阻尼拟合失败"}

    theta_eq, C1, C2, lam1, lam2 = best.x
    ls, lf = sorted([lam1, lam2])
    beta = 0.5*(ls+lf)
    omega0 = math.sqrt(ls*lf)
    zeta = beta/omega0
    pred = model(best.x)
    r2 = r2_score(y, pred)

    def ode(state, t):
        th, v = state; q = th - theta_eq
        return [v, -omega0**2*q - 2*beta*v]
    sol = odeint(ode, [y[0], (y[1]-y[0])/(t[1]-t[0]) if len(t)>1 else 0], t)
    ode_r2 = r2_score(y, sol[:,0])
    ode_rmse = math.sqrt(np.mean((y-sol[:,0])**2))

    lib = _library_fit(t, y, theta_eq)
    plot_path = os.path.join(output_dir, "sr_analysis.png")
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    ax[0].plot(t, np.rad2deg(y), alpha=0.4, label="Data"); ax[0].plot(t, np.rad2deg(pred), 'r-', lw=1.5, label="Fit")
    ax[0].plot(t, np.rad2deg(sol[:,0]), 'g--', lw=1.5, label="ODE"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(np.rad2deg(y-theta_eq), np.gradient(y)/np.max(np.abs(np.gradient(y))), lw=1); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()

    result = {
        "beta": float(beta), "omega0": float(omega0), "zeta": float(zeta),
        "theta_eq_deg": float(np.rad2deg(theta_eq)),
        "damping_type": "过阻尼" if zeta>1 else ("临界阻尼" if abs(zeta-1)<0.05 else "欠阻尼"),
        "ode_r2": float(ode_r2), "ode_rmse_deg": float(np.rad2deg(ode_rmse)),
        "r2_fit": float(r2), "plot_path": plot_path,
        "library_coef": lib, "final_source": "过阻尼解析解时域拟合 + ODE验证",
    }
    return result

# ---- 对外的统一接口 ----
def run_sr_ci_underdamped(csv_path, output_dir):
    return _fit_damped_oscillation(csv_path, output_dir, "欠阻尼")

def run_sr_ci_critical(csv_path, output_dir):
    return _fit_damped_oscillation(csv_path, output_dir, "临界阻尼")

def run_sr_ci_overdamped(csv_path, output_dir):
    return _fit_damped_oscillation(csv_path, output_dir, "过阻尼")

def run_sr_niubai(csv_path, output_dir):
    return _fit_damped_oscillation(csv_path, output_dir, "欠阻尼")
