"""
TechValue AI - core logic (no Streamlit code here, so it can be tested on its own).

Contains:
  * loading the saved model + metadata
  * building and validating the model input row (must match the training columns exactly)
  * price helpers (percentiles, reference prices)
  * optional currency conversion helpers
  * matplotlib charts that use only real dataset statistics stored in metadata.json
  * the HTML/CSS illustration of the animated gaming PC
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_artifacts" / "model.joblib"
META_PATH = ROOT / "model_artifacts" / "metadata.json"

# ---- colour palette shared by the CSS and the charts -------------------------------------------
BG = "#0b0d12"
PANEL = "#12151d"
CYAN = "#22e4ff"
PURPLE = "#8b5cf6"
MAGENTA = "#ff2bd6"
GREEN = "#39ff88"
TEXT = "#e8ecf4"
MUTED = "#8b93a7"
GRID = "#262b38"


# =================================================================================================
# Loading
# =================================================================================================
def load_metadata(path: Path = META_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(path: Path = MODEL_PATH):
    import joblib
    return joblib.load(path)


# =================================================================================================
# Model input
# =================================================================================================
def build_input_frame(config: dict, meta: dict):
    """One-row DataFrame with the exact columns (and order) the pipeline was trained on."""
    import pandas as pd
    cols = meta["features"]["all"]
    missing = [c for c in cols if c not in config]
    if missing:
        raise KeyError(f"Missing inputs: {missing}")
    return pd.DataFrame([{c: config[c] for c in cols}])[cols]


def validate_config(config: dict, meta: dict):
    """Return (errors, warnings). Errors block prediction; warnings are shown but do not block it."""
    errors, warnings = [], []
    feats = meta["features"]
    for c in feats["all"]:
        if c not in config or config[c] is None or config[c] == "":
            errors.append(f"'{c}' has no value.")
    if errors:
        return errors, warnings

    for c in feats["categorical"]:
        if config[c] not in meta["categories"][c]:
            warnings.append(f"'{config[c]}' was never seen for '{c}' in the training data; the model will treat it "
                            f"as an unknown category, so the estimate may be less reliable.")
    for c in feats["numeric"]:
        try:
            v = float(config[c])
        except (TypeError, ValueError):
            errors.append(f"'{c}' must be a number.")
            continue
        info = meta["numeric_info"][c]
        if v < info["min"] or v > info["max"]:
            warnings.append(f"{c} = {v:g} is outside the range seen in training ({info['min']:g} to {info['max']:g}); "
                            f"the estimate is an extrapolation.")
    if errors:
        return errors, warnings

    # Combinations that essentially never occur in the training data
    if config["cpu_threads"] < config["cpu_cores"]:
        errors.append("CPU threads cannot be lower than CPU cores.")
    if config["cpu_boost_ghz"] < config["cpu_base_ghz"]:
        errors.append("CPU boost clock cannot be lower than the base clock.")
    if (config["brand"] == "Apple") != (config["cpu_brand"] == "Apple"):
        warnings.append("In the training data only Apple computers use Apple CPUs (and vice versa); this combination is unusual.")
    if config["device_type"] == "Desktop" and config["battery_wh"] > 0:
        warnings.append("Desktops in the training data have no battery (0 Wh).")
    if config["device_type"] == "Laptop" and config["psu_watts"] > 0:
        warnings.append("Laptops in the training data have no separate PSU (0 W).")
    return errors, warnings


def predict_price(model, row) -> float:
    """Run the saved pipeline (imputers + scaler + one-hot + regressor). Prices cannot be negative."""
    return float(max(model.predict(row)[0], 0.0))


# =================================================================================================
# Statistics from metadata (all derived from the real training data)
# =================================================================================================
def price_percentile(price: float, meta: dict) -> float:
    """Share (0-100) of dataset computers that cost less than `price`."""
    grid = np.array(meta["price_sorted_grid"])
    levels = np.linspace(0, 100, len(grid))
    return float(np.interp(price, grid, levels))


def feature_share_at_or_below(col: str, value: float, meta: dict) -> float:
    """Share (0-100) of dataset computers whose `col` is <= value."""
    e = meta["feature_ecdf"][col]
    x, cdf = np.array(e["x"]), np.array(e["cdf"])
    if value < x[0]:
        return 0.0
    if value >= x[-1]:
        return 100.0
    return float(cdf[np.searchsorted(x, value, side="right") - 1] * 100)


def reference_prices(config: dict, meta: dict):
    """Real average/median prices of comparable groups in the dataset: [(label, value)]."""
    gm = meta["group_means"]
    out = [("All computers (median)", meta["price_stats"]["median"])]
    for key, label, cast in [("device_type", "Same device type", str),
                             ("brand", "Same brand", str),
                             ("cpu_tier", "Same CPU tier", lambda v: str(int(v))),
                             ("gpu_tier", "Same GPU tier", lambda v: str(int(v))),
                             ("ram_gb", "Same RAM size", lambda v: str(int(v)))]:
        group = gm.get(key, {})
        k = cast(config[key])
        if k in group:
            out.append((f"{label} (mean)", group[k]["mean"]))
    return out


def format_price(v: float) -> str:
    return f"{v:,.2f}"


# =================================================================================================
# Currency conversion (display only - never changes the model prediction)
# =================================================================================================
CURRENCIES = ["USD", "EUR", "GBP", "PKR", "AED", "SAR", "INR", "CAD", "AUD", "JPY"]
RATE_SOURCE_LABEL = "open.er-api.com (Exchange Rate API)"


def fetch_live_rate(base: str, target: str, timeout: float = 5.0):
    """Try to fetch a live exchange rate. Returns dict(rate, date, source) or None when unavailable."""
    if base == target:
        return {"rate": 1.0, "date": "n/a", "source": "same currency"}
    try:
        import requests
        r = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("result") == "success" and target in data.get("rates", {}):
            return {"rate": float(data["rates"][target]),
                    "date": str(data.get("time_last_update_utc", "unknown date")),
                    "source": RATE_SOURCE_LABEL}
    except Exception:
        return None
    return None


def convert(amount: float, rate: float) -> float:
    return amount * rate


# =================================================================================================
# Charts (matplotlib, dark theme, transparent background)
# =================================================================================================
def _style(ax):
    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def _fig(w, h):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    plt.rcParams["axes.titlecolor"] = TEXT
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    _style(ax)
    return fig, ax


def gauge_figure(price: float, meta: dict):
    """Half-circle gauge: where the estimate sits between the 1st and 99th percentile of dataset prices."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    lo, hi = meta["price_stats"]["percentiles"]["1"], meta["price_stats"]["percentiles"]["99"]
    frac = float(np.clip((price - lo) / (hi - lo), 0, 1))

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.25, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    n = 60
    cols = plt.cm.cool(np.linspace(0, 1, n))        # cyan -> magenta
    for i in range(n):
        a0, a1 = 180 - 180 * i / n, 180 - 180 * (i + 1) / n
        active = (i + 1) / n <= frac + 1e-9
        ax.add_patch(Wedge((0, 0), 1.0, a1, a0, width=0.22, facecolor=cols[i],
                           alpha=1.0 if active else 0.16, edgecolor="none"))
    ang = np.deg2rad(180 - 180 * frac)
    ax.plot([0.6 * np.cos(ang), 0.98 * np.cos(ang)], [0.6 * np.sin(ang), 0.98 * np.sin(ang)], color="white", lw=2.5)
    ax.text(0, 0.18, format_price(price), ha="center", va="center", color=TEXT, fontsize=22, fontweight="bold")
    ax.text(0, -0.05, "price units", ha="center", va="center", color=MUTED, fontsize=9)
    ax.text(-1.0, -0.17, f"{lo:,.0f}", ha="center", color=MUTED, fontsize=8)
    ax.text(1.0, -0.17, f"{hi:,.0f}", ha="center", color=MUTED, fontsize=8)
    ax.text(0, 1.14, "Position within typical dataset prices (1st-99th percentile)", ha="center", color=MUTED, fontsize=8)
    return fig


def reference_bar_figure(price: float, refs, meta: dict):
    fig, ax = _fig(6.4, 3.6)
    labels = ["Your estimate"] + [r[0] for r in refs]
    values = [price] + [r[1] for r in refs]
    colors = [MAGENTA] + [CYAN] * len(refs)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.012, i, f"{v:,.0f}", va="center", color=TEXT, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("price units")
    ax.set_title("Your estimate vs real dataset averages", loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def distribution_figure(price: float, meta: dict):
    fig, ax = _fig(6.4, 3.6)
    edges = np.array(meta["price_hist"]["edges"])
    counts = np.array(meta["price_hist"]["counts"])
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    ax.bar(centers, counts, width=widths * 0.95, color=[PURPLE if c < price else GRID for c in centers], alpha=0.9)
    ax.axvline(price, color=MAGENTA, lw=2.4)
    pct = price_percentile(price, meta)
    ax.text(price, counts.max() * 1.04, f" estimate\n {pct:.0f}% of computers cost less", color=MAGENTA, fontsize=9, va="top",
            ha="left" if price < centers[len(centers) // 2] else "right")
    ax.set_xlabel("price units")
    ax.set_ylabel("number of computers")
    ax.set_title("Where the estimate sits among 100,000 observed prices", loc="left", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return fig


FEATURE_LABELS = {"ram_gb": "RAM (GB)", "cpu_cores": "CPU cores", "cpu_tier": "CPU tier", "gpu_tier": "GPU tier",
                  "vram_gb": "GPU VRAM (GB)", "storage_gb": "Storage (GB)", "refresh_hz": "Refresh rate (Hz)",
                  "display_size_in": "Screen size (in)", "release_year": "Release year"}


def feature_percentile_figure(config: dict, meta: dict):
    cols = list(FEATURE_LABELS)
    shares = [feature_share_at_or_below(c, float(config[c]), meta) for c in cols]
    fig, ax = _fig(6.4, 3.6)
    y = np.arange(len(cols))
    ax.barh(y, [100] * len(cols), color=GRID, height=0.6)
    ax.barh(y, shares, color=GREEN, height=0.6)
    for i, (c, s) in enumerate(zip(cols, shares)):
        ax.text(101.5, i, f"{config[c]:g}  ({s:.0f}%)", va="center", color=TEXT, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[c] for c in cols], color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of dataset computers with this spec or lower")
    ax.set_title("How your specs compare with the dataset", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def model_comparison_figure(meta: dict):
    models = meta["metrics"]["all_models"]
    names = [m["model"] for m in models]
    rmse = [m["rmse_mean"] for m in models]
    fig, ax = _fig(6.4, 3.4)
    colors = [GREEN if m["model"] == meta["metrics"]["selected"]["model"] else PURPLE for m in models]
    y = np.arange(len(names))
    ax.barh(y, rmse, xerr=[m["rmse_std"] for m in models], color=colors, height=0.6,
            error_kw=dict(ecolor=MUTED, capsize=3, elinewidth=1))
    for i, v in enumerate(rmse):
        ax.text(v + max(rmse) * 0.07, i, f"{v:,.1f}", va="center", color=TEXT, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(rmse) * 1.3)
    ax.set_xlabel("mean RMSE over 10 folds (lower is better)")
    ax.set_title("All 5 algorithms compared", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def scatter_figure(meta: dict):
    fig, ax = _fig(6.4, 3.8)
    a = np.array(meta["oof_sample"]["actual"])
    p = np.array(meta["oof_sample"]["predicted"])
    ax.scatter(a, p, s=7, alpha=0.45, color=CYAN, edgecolors="none")
    lim = [min(a.min(), p.min()), max(a.max(), p.max())]
    ax.plot(lim, lim, "--", color=MAGENTA, lw=1.6)
    ax.set_xlabel("actual price")
    ax.set_ylabel("predicted price")
    ax.set_title("Validation: actual vs predicted (1,500 held-out computers)", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def importance_figure(meta: dict):
    items = meta["feature_importance_permutation"][:10][::-1]
    fig, ax = _fig(6.4, 3.6)
    ax.barh([i["feature"] for i in items], [i["rmse_increase"] for i in items], color=PURPLE, height=0.6)
    ax.tick_params(axis="y", colors=TEXT)
    ax.set_xlabel("increase in validation RMSE when the input is shuffled")
    ax.set_title("Inputs the model relies on (not a causal explanation)", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


# =================================================================================================
# Sample configurations (values are checked against the metadata in the tests)
# =================================================================================================
SAMPLE_CONFIGS = {
    "Gaming laptop": {
        "device_type": "Laptop", "brand": "ASUS", "release_year": 2024, "os": "Windows", "form_factor": "Gaming",
        "cpu_brand": "Intel", "cpu_tier": 4, "cpu_cores": 14, "cpu_threads": 28, "cpu_base_ghz": 2.6, "cpu_boost_ghz": 3.9,
        "gpu_brand": "NVIDIA", "gpu_model": "RTX 40 80", "gpu_tier": 4, "vram_gb": 10,
        "ram_gb": 32, "storage_type": "NVMe", "storage_gb": 1024, "storage_drive_count": 1,
        "display_type": "OLED", "display_size_in": 16.0, "resolution": "2560x1600", "refresh_hz": 165,
        "battery_wh": 80, "charger_watts": 180, "psu_watts": 0, "wifi": "Wi-Fi 6E", "bluetooth": 5.3,
        "weight_kg": 2.3, "warranty_months": 24,
    },
    "Workstation desktop": {
        "device_type": "Desktop", "brand": "Dell", "release_year": 2025, "os": "Windows", "form_factor": "Full-Tower",
        "cpu_brand": "AMD", "cpu_tier": 6, "cpu_cores": 26, "cpu_threads": 52, "cpu_base_ghz": 3.4, "cpu_boost_ghz": 4.3,
        "gpu_brand": "NVIDIA", "gpu_model": "RTX 40 90", "gpu_tier": 6, "vram_gb": 16,
        "ram_gb": 128, "storage_type": "NVMe", "storage_gb": 2048, "storage_drive_count": 2,
        "display_type": "IPS", "display_size_in": 32.0, "resolution": "3840x2160", "refresh_hz": 144,
        "battery_wh": 0, "charger_watts": 0, "psu_watts": 1000, "wifi": "Wi-Fi 7", "bluetooth": 5.3,
        "weight_kg": 14.0, "warranty_months": 36,
    },
    "Budget laptop": {
        "device_type": "Laptop", "brand": "Acer", "release_year": 2021, "os": "Windows", "form_factor": "Mainstream",
        "cpu_brand": "AMD", "cpu_tier": 1, "cpu_cores": 6, "cpu_threads": 12, "cpu_base_ghz": 2.0, "cpu_boost_ghz": 3.0,
        "gpu_brand": "Intel", "gpu_model": "Arc A310", "gpu_tier": 1, "vram_gb": 2,
        "ram_gb": 8, "storage_type": "SSD", "storage_gb": 256, "storage_drive_count": 1,
        "display_type": "LED", "display_size_in": 15.6, "resolution": "1920x1080", "refresh_hz": 60,
        "battery_wh": 56, "charger_watts": 65, "psu_watts": 0, "wifi": "Wi-Fi 5", "bluetooth": 5.0,
        "weight_kg": 1.8, "warranty_months": 12,
    },
}


# =================================================================================================
# HTML/CSS illustration of the gaming PC (rendered inside an iframe by Streamlit)
# =================================================================================================
def case_html(tags: dict) -> str:
    """Self-contained HTML page: glowing PC case, animated RGB fan, and live spec labels."""
    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    cpu, gpu, ram, sto = (esc(tags.get(k, "-")) for k in ("cpu", "gpu", "ram", "storage"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ --cyan:{CYAN}; --purple:{PURPLE}; --magenta:{MAGENTA}; --green:{GREEN}; --text:{TEXT}; --muted:{MUTED}; }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; background:transparent; font-family:"Chakra Petch","Segoe UI",system-ui,sans-serif; color:var(--text); }}
  .stage {{ display:grid; grid-template-columns: 1fr 190px; align-items:center; gap:14px; padding:10px 6px; min-height:400px; }}
  .case {{ position:relative; width:100%; max-width:280px; height:380px; margin:0 auto; border-radius:18px;
          background:linear-gradient(160deg,#171b26,#0b0d12 60%); border:2px solid #2b3144;
          box-shadow:0 0 0 1px #0a0c11 inset, 0 0 38px rgba(139,92,246,.35), 0 0 90px rgba(34,228,255,.12); overflow:hidden; }}
  .glass {{ position:absolute; inset:12px; border-radius:12px; border:1px solid rgba(34,228,255,.35);
           background:linear-gradient(135deg,rgba(255,255,255,.07),rgba(255,255,255,0) 40%), #080a0f; }}
  .strip {{ position:absolute; left:0; right:0; top:0; height:3px; background:linear-gradient(90deg,var(--cyan),var(--purple),var(--magenta)); }}
  .fan {{ position:absolute; left:50%; top:20px; width:120px; height:120px; margin-left:-60px; border-radius:50%;
         background:conic-gradient(from 0deg,#22e4ff,#8b5cf6,#ff2bd6,#39ff88,#22e4ff); padding:7px;
         box-shadow:0 0 24px rgba(139,92,246,.7); animation:spin 6s linear infinite; }}
  .fan::before {{ content:""; position:absolute; inset:7px; border-radius:50%; background:#0b0d12; }}
  .blades {{ position:absolute; inset:15px; border-radius:50%;
            background:repeating-conic-gradient(from 0deg,rgba(255,255,255,.55) 0 14deg,rgba(255,255,255,0) 14deg 45deg); mask:radial-gradient(circle,transparent 15%,#000 16%); -webkit-mask:radial-gradient(circle,transparent 15%,#000 16%); }}
  .hub {{ position:absolute; left:50%; top:50%; width:26px; height:26px; margin:-13px; border-radius:50%; background:#161a24; border:2px solid var(--cyan); }}
  .part {{ position:absolute; border-radius:6px; }}
  .cpu {{ left:30px; top:152px; width:52px; height:52px; border:2px solid var(--cyan); box-shadow:0 0 14px rgba(34,228,255,.7); background:#0e1b24; }}
  .cpu::after {{ content:""; position:absolute; inset:14px; border:1px solid var(--cyan); border-radius:3px; }}
  .ram {{ left:96px; top:148px; width:9px; height:62px; background:linear-gradient(var(--purple),#3b2a7a); box-shadow:0 0 10px rgba(139,92,246,.9); animation:pulse 2.6s ease-in-out infinite; }}
  .ram.r2 {{ left:110px; animation-delay:.3s; }} .ram.r3 {{ left:124px; animation-delay:.6s; }} .ram.r4 {{ left:138px; animation-delay:.9s; }}
  .gpu {{ left:28px; right:28px; top:228px; height:46px; border:2px solid var(--magenta); background:linear-gradient(90deg,#2a0f27,#14101c);
         box-shadow:0 0 18px rgba(255,43,214,.65); display:flex; align-items:center; justify-content:space-around; }}
  .gpu i {{ width:26px; height:26px; border-radius:50%; border:2px solid var(--magenta); background:radial-gradient(circle,#ff2bd6 0 18%,transparent 20%); animation:spin 3s linear infinite; }}
  .ssd {{ left:30px; right:30px; top:292px; height:14px; border:1px solid var(--green); background:#0c1a12; box-shadow:0 0 10px rgba(57,255,136,.6); }}
  .ssd.s2 {{ top:312px; width:60%; right:auto; }}
  .leds {{ position:absolute; right:22px; top:154px; width:6px; height:60px; border-radius:3px; background:linear-gradient(var(--cyan),var(--magenta)); box-shadow:0 0 10px var(--magenta); }}
  .tags {{ display:flex; flex-direction:column; gap:12px; }}
  .tag {{ border-left:3px solid var(--c); padding:6px 10px; background:rgba(255,255,255,.04); border-radius:0 8px 8px 0; box-shadow:-6px 0 18px -8px var(--c); }}
  .tag b {{ display:block; font-size:12px; letter-spacing:.04em; color:var(--c); font-weight:600; }}
  .tag span {{ display:block; font-size:13px; line-height:1.25; word-break:break-word; }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
  @keyframes pulse {{ 0%,100% {{ opacity:.65; }} 50% {{ opacity:1; }} }}
  @media (max-width:520px) {{ .stage {{ grid-template-columns:1fr; }} .tags {{ flex-direction:row; flex-wrap:wrap; }} .tag {{ flex:1 1 140px; }} }}
  @media (prefers-reduced-motion: reduce) {{ .fan,.gpu i,.ram {{ animation:none; }} }}
</style></head>
<body>
<div class="stage" role="img" aria-label="Illustration of a gaming PC with CPU, GPU, RAM and storage highlighted">
  <div class="case"><div class="strip"></div>
    <div class="glass"></div>
    <div class="fan"><div class="blades"></div><div class="hub"></div></div>
    <div class="part cpu"></div>
    <div class="part ram"></div><div class="part ram r2"></div><div class="part ram r3"></div><div class="part ram r4"></div>
    <div class="part gpu"><i></i><i></i><i></i></div>
    <div class="part ssd"></div><div class="part ssd s2"></div>
    <div class="leds"></div>
  </div>
  <div class="tags">
    <div class="tag" style="--c:var(--cyan)"><b>CPU</b><span>{cpu}</span></div>
    <div class="tag" style="--c:var(--magenta)"><b>GPU</b><span>{gpu}</span></div>
    <div class="tag" style="--c:var(--purple)"><b>RAM</b><span>{ram}</span></div>
    <div class="tag" style="--c:var(--green)"><b>Storage</b><span>{sto}</span></div>
  </div>
</div>
</body></html>"""
