"""暗状態ダイオード I-V データをガウス過程回帰 (GPR) で解析し、式を推定する。

手順
1. 順方向 (V>0): y = ln(I) を GP で回帰。カーネル = 線形 (DotProduct) + RBF、ノイズは分解能から既知として与える。
   ln I が V に対して直線 ⇔ Shockley 式 I = I0 exp(V / (n Vt))。
   GP 事後サンプルごとに直線を当てはめ、n と I0 の分布（不確かさ）を得る。
   さらに GP 平均の微分から局所理想係数 n(V) を求め、高電流側での直列抵抗の効果を確認する。
2. 逆方向 (V<0): I を GP で回帰 (線形 + RBF)。傾きからシャント抵抗 Rsh を推定。
3. 1, 2 で得た値を初期値として一ダイオードモデル
       I = I0 [exp((V - I Rs)/(n Vt)) - 1] + (V - I Rs)/Rsh
   を全データにフィットし、最終的な式とする。
"""
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C, DotProduct, RBF)

plt.rcParams["font.family"] = "IPAGothic"
# 点数が少ないためハイパーパラメータが境界に張り付く警告は表示しない
warnings.filterwarnings("ignore", category=ConvergenceWarning)

K_B, Q = 1.380649e-23, 1.602176634e-19
T = 300.0
VT = K_B * T / Q  # 熱電圧 ≈ 25.85 mV

data = np.loadtxt("data/dark_iv.csv", delimiter=",", skiprows=1)
V_all, I_all = data[:, 0], data[:, 1] * 1e-3  # A
fwd, rev = V_all > 0, V_all < 0
Vf, If = V_all[fwd], I_all[fwd]
Vr, Ir = V_all[rev], I_all[rev]
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. 順方向 GP
# 測定ノイズは既知として与える: 電流計の分解能 0.0001 mA (±半分) と相対誤差 1 %
RES = 0.5 * 0.0001e-3  # A
yf = np.log(If)
var_lnI = (RES / If) ** 2 + 0.01 ** 2
kf = (C(1.0, (1e-8, 1e5)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
      + C(0.1) * RBF(length_scale=0.2, length_scale_bounds=(0.05, 10)))
# normalize_y=True では alpha も正規化後の尺度で解釈されるため y の分散で割る
gp_f = GaussianProcessRegressor(kf, alpha=var_lnI / yf.var(), normalize_y=True,
                                n_restarts_optimizer=20, random_state=0)
gp_f.fit(Vf[:, None], yf)

Vg = np.linspace(0.30, 0.70, 401)
mu_f, sd_f = gp_f.predict(Vg[:, None], return_std=True)

# 局所理想係数 n(V) = 1 / (Vt d lnI/dV)
slope = np.gradient(mu_f, Vg)
n_local = 1.0 / (VT * slope)

# 事後サンプル → 直線当てはめ (低電流域 0.33–0.60 V: 直列抵抗の影響が小さい領域)
samples = gp_f.sample_y(Vg[:, None], n_samples=2000, random_state=1)
lin = (Vg >= 0.33) & (Vg <= 0.60)
coef = np.polyfit(Vg[lin], samples[lin], 1)  # (2, n_samples)
n_s = 1.0 / (VT * coef[0])
I0_s = np.exp(coef[1])

def ci(x):
    lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
    return med, lo, hi

n_med, n_lo, n_hi = ci(n_s)
I0_med, I0_lo, I0_hi = ci(I0_s)

# ---------------------------------------------------------------- 2. 逆方向 GP
Ir_uA = Ir * 1e6
kr = (C(1.0) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-8, 1e3))
      + C(0.1) * RBF(length_scale=2.0, length_scale_bounds=(0.5, 50)))
# 分解能 0.1 µA の量子化誤差 (一様分布の標準偏差 0.1/√12 µA)
gp_r = GaussianProcessRegressor(kr, alpha=(0.1 / np.sqrt(12)) ** 2 / Ir_uA.var(),
                                normalize_y=True, n_restarts_optimizer=20,
                                random_state=0)
# 物理的拘束 I(0) = 0 を擬似観測点として追加
gp_r.alpha = np.append(np.full(len(Vr), gp_r.alpha), 1e-6)
gp_r.fit(np.append(Vr, 0.0)[:, None], np.append(Ir_uA, 0.0))
Vgr = np.linspace(-5.2, 0.0, 300)
mu_r, sd_r = gp_r.predict(Vgr[:, None], return_std=True)
samp_r = gp_r.sample_y(Vgr[:, None], n_samples=2000, random_state=2)
g_s = np.polyfit(Vgr, samp_r, 1)[0] * 1e-6  # コンダクタンス [S]
Rsh_s = 1.0 / g_s[g_s > 0]
Rsh_med, Rsh_lo, Rsh_hi = ci(Rsh_s)

# ---------------------------------------------------------------- 3. 一ダイオードモデル
def diode_I(V, I0, n, Rs, Rsh):
    """陰関数 I = I0[exp((V-IRs)/nVt)-1] + (V-IRs)/Rsh を数値的に解く。"""
    V = np.atleast_1d(V)
    out = np.empty_like(V, dtype=float)
    for k, v in enumerate(V):
        f = lambda i: I0 * np.expm1(np.minimum((v - i * Rs) / (n * VT), 700)) + (v - i * Rs) / Rsh - i
        hi = max(abs(v) / max(Rs, 1e-9), 1.0)
        out[k] = brentq(f, -hi, hi, xtol=1e-18)
    return out

def resid(p):
    I0, n, Rs, Rsh = np.exp(p[0]), p[1], p[2], np.exp(p[3])
    Im = diode_I(V_all, I0, n, Rs, Rsh)
    # 桁が大きく違うので asinh (≒ 対数) スケールで残差をとり、分解能由来の誤差で重み付け
    s = 1e-9
    sig = np.sqrt((RES / np.abs(I_all)) ** 2 + 0.01 ** 2)
    return (np.arcsinh(Im / s) - np.arcsinh(I_all / s)) / sig

p0 = [np.log(I0_med), n_med, 1.0, np.log(Rsh_med)]
fit = least_squares(resid, p0, bounds=([-60, 0.8, 0.0, np.log(1e4)],
                                       [0, 4.0, 200.0, np.log(1e10)]))
I0_fit, n_fit, Rs_fit, Rsh_fit = (np.exp(fit.x[0]), fit.x[1], fit.x[2],
                                  np.exp(fit.x[3]))

# ---------------------------------------------------------------- 出力
print(f"熱電圧 Vt = {VT*1e3:.2f} mV (T = {T:.0f} K)\n")
print("[1] 順方向 GP (ln I)")
print(f"  学習カーネル: {gp_f.kernel_}")
print(f"  n  = {n_med:.3f}  (95% CI {n_lo:.3f} – {n_hi:.3f})")
print(f"  I0 = {I0_med:.3e} A  (95% CI {I0_lo:.2e} – {I0_hi:.2e})")
for v in (0.35, 0.45, 0.55, 0.62, 0.67):
    j = np.argmin(abs(Vg - v))
    print(f"  局所 n({v:.2f} V) = {n_local[j]:.2f}")
print("\n[2] 逆方向 GP (I)")
print(f"  学習カーネル: {gp_r.kernel_}")
print(f"  Rsh = {Rsh_med/1e6:.1f} MΩ  (95% CI {Rsh_lo/1e6:.1f} – {Rsh_hi/1e6:.1f})")
print("\n[3] 一ダイオードモデル (GP 値を初期値に全点フィット)")
print(f"  I0 = {I0_fit:.3e} A, n = {n_fit:.3f}, Rs = {Rs_fit:.2f} Ω, "
      f"Rsh = {Rsh_fit/1e6:.2f} MΩ")
print("\n  V[V]    I_meas[mA]   I_model[mA]   I_GP[mA]")
I_model = diode_I(V_all, I0_fit, n_fit, Rs_fit, Rsh_fit)
for v, im, ip in zip(V_all, I_all, I_model):
    if v > 0:
        ig = np.exp(gp_f.predict([[v]])[0]) * 1e3
    else:
        ig = gp_r.predict([[v]])[0] * 1e-3
    print(f"  {v:7.3f}  {im*1e3:10.4g}  {ip*1e3:12.4g}  {ig:10.4g}")

# ---------------------------------------------------------------- 図
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

a = ax[0, 0]
a.fill_between(Vg, np.exp(mu_f - 2 * sd_f) * 1e3, np.exp(mu_f + 2 * sd_f) * 1e3,
               alpha=0.25, label="GP ±2σ")
a.semilogy(Vg, np.exp(mu_f) * 1e3, label="GP 平均")
a.semilogy(Vg, I0_med * np.exp(Vg / (n_med * VT)) * 1e3, "--",
           label=f"GP由来 Shockley: n={n_med:.2f}")
Vm = np.linspace(0.30, 0.70, 200)
a.semilogy(Vm, diode_I(Vm, I0_fit, n_fit, Rs_fit, Rsh_fit) * 1e3, ":k",
           label="一ダイオードモデル")
a.semilogy(Vf, If * 1e3, "o", color="C3", label="測定")
a.set(xlabel="電圧 [V]", ylabel="電流 [mA]", title="順方向 (片対数)")
a.legend(); a.grid(which="both", alpha=0.3)

a = ax[0, 1]
meas = (Vg >= Vf.min()) & (Vg <= Vf.max())  # 外挿域は除く
a.plot(Vg[meas], n_local[meas])
a.axhline(n_med, ls="--", color="gray", label=f"直線近似 n = {n_med:.2f}")
a.set(xlabel="電圧 [V]", ylabel="局所理想係数 n(V)",
      title="GP 平均の微分から求めた n(V)", ylim=(0, 4))
a.legend(); a.grid(alpha=0.3)

a = ax[1, 0]
a.fill_between(Vgr, mu_r - 2 * sd_r, mu_r + 2 * sd_r, alpha=0.25, label="GP ±2σ")
a.plot(Vgr, mu_r, label="GP 平均")
a.plot(Vgr, Vgr / Rsh_med * 1e6, "--", label=f"Rsh = {Rsh_med/1e6:.1f} MΩ")
a.plot(Vr, Ir_uA, "o", color="C3", label="測定")
a.set(xlabel="電圧 [V]", ylabel="電流 [μA]", title="逆方向")
a.legend(); a.grid(alpha=0.3)

a = ax[1, 1]
Vfull = np.linspace(-5.2, 0.7, 400)
a.plot(Vfull, diode_I(Vfull, I0_fit, n_fit, Rs_fit, Rsh_fit) * 1e3, "k",
       label="一ダイオードモデル")
a.plot(V_all, I_all * 1e3, "o", color="C3", label="測定")
a.set(xlabel="電圧 [V]", ylabel="電流 [mA]", title="全範囲 (線形)")
a.legend(); a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("gp_diode.png", dpi=130)
print("\n図を gp_diode.png に保存しました")
