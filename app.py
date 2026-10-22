"""
TechValue AI - Intelligent Computer Price Prediction (Streamlit app)

Run with:   streamlit run app.py

The app loads the saved scikit-learn pipeline (model_artifacts/model.joblib) and the dataset statistics
(model_artifacts/metadata.json). Every input below is one of the 30 features the model was trained on, and the
choices are filled from the training data - nothing is invented.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

import core

st.set_page_config(page_title="TechValue AI", page_icon="🖥️", layout="wide", initial_sidebar_state="collapsed")


# =================================================================================================
# Cached loading (runs once, not on every click)
# =================================================================================================
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return core.load_model()


@st.cache_data(show_spinner=False)
def get_meta():
    return core.load_metadata()


@st.cache_data(ttl=3600, show_spinner=False)
def get_live_rate(base, target):
    return core.fetch_live_rate(base, target)


try:
    META = get_meta()
    MODEL = get_model()
except Exception as exc:                                    # missing / incompatible artifact
    st.error("The trained model could not be loaded. Re-create it by running TechValue_AI.ipynb, then copy "
             "`model_artifacts/` next to `app.py`.")
    st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
    st.stop()

CAT = META["categories"]
NUM = META["numeric_info"]
METRICS = META["metrics"]["selected"]


# =================================================================================================
# Styling
# =================================================================================================
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,.18), transparent 60%),
                     radial-gradient(900px 500px at -10% 10%, rgba(34,228,255,.10), transparent 60%), {core.BG}; color: {core.TEXT}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1250px; }}
h1, h2, h3, .display {{ font-family: 'Chakra Petch', 'IBM Plex Sans', sans-serif !important; letter-spacing: .01em; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}

.brand {{ display:flex; align-items:center; gap:.6rem; font-family:'Chakra Petch',sans-serif; font-weight:700; font-size:1.25rem; }}
.brand .dot {{ width:12px; height:12px; border-radius:50%; background:conic-gradient({core.CYAN},{core.PURPLE},{core.MAGENTA},{core.CYAN}); box-shadow:0 0 12px {core.PURPLE}; }}
.hero-title {{ font-family:'Chakra Petch',sans-serif; font-weight:700; font-size:clamp(2.2rem,4.6vw,3.6rem); line-height:1.05; margin:.8rem 0 .8rem; }}
.hero-title span {{ background:linear-gradient(90deg,{core.CYAN},{core.PURPLE} 55%,{core.MAGENTA}); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.hero-copy {{ color:{core.MUTED}; font-size:1.05rem; max-width:34rem; line-height:1.6; }}
.chips {{ display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1.3rem; }}
.chip {{ border:1px solid #2b3144; background:rgba(255,255,255,.04); border-radius:999px; padding:.4rem .9rem; font-size:.88rem; color:{core.TEXT}; }}
.chip b {{ color:{core.CYAN}; }}

/* Component panels: st.container(key="panel_xxx") adds the CSS class st-key-panel_xxx */
[class*="st-key-panel_"] {{ background:linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
    backdrop-filter: blur(8px); border:1px solid var(--pc, #2b3144); border-radius:16px; padding:1.1rem 1.2rem .6rem;
    box-shadow:0 0 0 1px rgba(0,0,0,.4) inset, 0 0 26px -10px var(--pc, #000); margin-bottom:.6rem; }}
.st-key-panel_cpu {{ --pc:{core.CYAN}; }} .st-key-panel_gpu {{ --pc:{core.MAGENTA}; }}
.st-key-panel_ram {{ --pc:{core.PURPLE}; }} .st-key-panel_sto {{ --pc:{core.GREEN}; }}
.st-key-panel_sys {{ --pc:#4b5675; }} .st-key-panel_dsp {{ --pc:#4b5675; }}
.ptitle {{ font-family:'Chakra Petch',sans-serif; font-weight:600; font-size:1.15rem; color:var(--tc, {core.TEXT}); margin:0 0 .5rem; display:flex; align-items:center; gap:.5rem; }}
.ptitle::before {{ content:""; width:9px; height:9px; border-radius:50%; background:var(--tc, {core.TEXT}); box-shadow:0 0 10px var(--tc, {core.TEXT}); }}
.t-cyan {{ --tc:{core.CYAN}; }} .t-mag {{ --tc:{core.MAGENTA}; }} .t-pur {{ --tc:{core.PURPLE}; }} .t-grn {{ --tc:{core.GREEN}; }} .t-gry {{ --tc:#aab3c9; }}
.hint {{ color:{core.MUTED}; font-size:.82rem; margin:-.2rem 0 .6rem; }}

/* Buttons */
div.stButton > button {{ border-radius:12px; border:1px solid #2b3144; background:rgba(255,255,255,.05); color:{core.TEXT}; font-weight:600; padding:.55rem 1rem; transition:box-shadow .2s, border-color .2s; }}
div.stButton > button:hover {{ border-color:{core.CYAN}; box-shadow:0 0 14px -2px {core.CYAN}; color:{core.TEXT}; }}
div.stButton > button:focus-visible {{ outline:2px solid {core.CYAN}; outline-offset:2px; }}
.st-key-predict_btn button, .st-key-reset_btn button {{ width:100%; }}
.st-key-predict_btn button {{ background:linear-gradient(90deg,{core.CYAN},{core.PURPLE} 55%,{core.MAGENTA}) !important; color:#06070b !important;
    border:0 !important; font-family:'Chakra Petch',sans-serif; font-size:1.2rem !important; padding:.85rem 1rem !important; box-shadow:0 0 28px -6px {core.MAGENTA}; }}
.st-key-predict_btn button:hover {{ box-shadow:0 0 38px -2px {core.CYAN} !important; }}

/* Result */
.result {{ border:1px solid {core.MAGENTA}; border-radius:18px; padding:1.4rem 1.6rem; background:linear-gradient(160deg, rgba(255,43,214,.10), rgba(34,228,255,.05));
    box-shadow:0 0 40px -12px {core.MAGENTA}; }}
.result .label {{ color:{core.MUTED}; font-size:.95rem; }}
.result .price {{ font-family:'Chakra Petch',sans-serif; font-weight:700; font-size:clamp(2.4rem,5vw,3.8rem); color:{core.TEXT}; line-height:1.1; text-shadow:0 0 24px rgba(255,43,214,.45); }}
.result .unit {{ font-size:1rem; color:{core.CYAN}; margin-left:.4rem; }}
.result .note {{ color:{core.MUTED}; font-size:.85rem; margin-top:.6rem; }}
.metric-card {{ border:1px solid #2b3144; border-radius:14px; padding:.9rem 1.1rem; background:rgba(255,255,255,.04); }}
.metric-card .v {{ font-family:'Chakra Petch',sans-serif; font-size:1.7rem; font-weight:700; color:{core.GREEN}; }}
.metric-card .k {{ color:{core.MUTED}; font-size:.85rem; }}
.footer {{ color:{core.MUTED}; font-size:.85rem; border-top:1px solid #1d2230; margin-top:2.5rem; padding-top:1rem; }}
.stTabs [data-baseweb="tab"] {{ font-family:'Chakra Petch',sans-serif; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =================================================================================================
# Session-state helpers (all widget keys start with "cfg_")
# =================================================================================================
def k(name, suffix=""):
    return f"cfg_{name}" + (f"|{suffix}" if suffix != "" else "")


def clear_config():
    for key in [x for x in st.session_state.keys() if x.startswith(("cfg_", "fx_"))]:
        del st.session_state[key]
    st.session_state.pop("result", None)


def apply_sample(sample_name):
    """Callback: load a valid example into the widgets (runs before the widgets are drawn)."""
    clear_config()
    s = core.SAMPLE_CONFIGS[sample_name]
    ss = st.session_state
    ss[k("device")] = s["device_type"]
    ss[k("brand")] = s["brand"]
    ss[k("os")] = s["os"]
    ss[k("year")] = s["release_year"]
    ss[k("form", s["device_type"])] = s["form_factor"]
    ss[k("cpu_brand", s["brand"] == "Apple")] = s["cpu_brand"]
    ss[k("cpu_tier")] = s["cpu_tier"]
    ss[k("cores", s["cpu_tier"])] = s["cpu_cores"]
    ss[k("threads", f'{s["cpu_tier"]}-{s["cpu_cores"]}')] = s["cpu_threads"]
    ss[k("base", s["cpu_tier"])] = s["cpu_base_ghz"]
    ss[k("boost", f'{s["cpu_tier"]}-{s["cpu_base_ghz"]}')] = s["cpu_boost_ghz"]
    ss[k("gpu_brand")] = s["gpu_brand"]
    ss[k("gpu_model", s["gpu_brand"])] = s["gpu_model"]
    ss[k("gpu_tier", s["gpu_model"])] = s["gpu_tier"]
    ss[k("vram", s["gpu_model"])] = s["vram_gb"]
    ss[k("ram")] = s["ram_gb"]
    ss[k("sto_type")] = s["storage_type"]
    ss[k("sto_gb")] = s["storage_gb"]
    ss[k("sto_n")] = s["storage_drive_count"]
    ss[k("dsp_type")] = s["display_type"]
    ss[k("res")] = s["resolution"]
    ss[k("hz")] = s["refresh_hz"]
    ss[k("size", s["device_type"])] = s["display_size_in"]
    ss[k("battery", s["device_type"])] = s["battery_wh"]
    ss[k("charger", s["device_type"])] = s["charger_watts"]
    ss[k("psu", s["device_type"])] = s["psu_watts"]
    ss[k("wifi")] = s["wifi"]
    ss[k("bt")] = s["bluetooth"]
    ss[k("weight", s["device_type"])] = s["weight_kg"]
    ss[k("warranty")] = s["warranty_months"]


def w_select(label, options, key, preferred=None, **kw):
    """selectbox that only passes a default `index` when the user/sample has not already set a value."""
    if key in st.session_state:
        return st.selectbox(label, options, key=key, **kw)
    return st.selectbox(label, options, key=key, index=idx(options, preferred if preferred is not None else options[0]), **kw)


def w_radio(label, options, key, preferred=None, **kw):
    if key in st.session_state:
        return st.radio(label, options, key=key, **kw)
    return st.radio(label, options, key=key, index=idx(options, preferred if preferred is not None else options[0]), **kw)


def w_select_slider(label, options, key, preferred=None, **kw):
    if key in st.session_state:
        return st.select_slider(label, options, key=key, **kw)
    return st.select_slider(label, options, value=pick(options, preferred if preferred is not None else options[0]), key=key, **kw)


def w_slider(label, lo, hi, key, default, **kw):
    if key in st.session_state:
        return st.slider(label, lo, hi, key=key, **kw)
    return st.slider(label, lo, hi, default, key=key, **kw)


def val(name, default, suffix=""):
    """Current value of a widget (or its default on the first run) - used for the live illustration."""
    return st.session_state.get(k(name, suffix), default)


def nearest(options, target):
    return min(options, key=lambda o: abs(o - target))


def pick(options, preferred):
    """Default for a select box: the preferred value if it is an option, otherwise the first option."""
    return preferred if preferred in options else options[0]


def idx(options, preferred):
    return options.index(pick(options, preferred))


# =================================================================================================
# Hero
# =================================================================================================
cpu_tier_now = val("cpu_tier", 3)
ti_now = META["cpu_tier_info"][str(cpu_tier_now)]
gpu_brand_now = val("gpu_brand", "NVIDIA")
gpu_models_now = sorted(m for m, i in META["gpu_model_info"].items() if i["gpu_brand"] == gpu_brand_now)
gpu_model_now = val("gpu_model", gpu_models_now[len(gpu_models_now) // 2], gpu_brand_now)
tags = {
    "cpu": f"{val('cpu_brand', 'Intel', val('brand', 'Dell') == 'Apple')} · tier {cpu_tier_now} · "
           f"{val('cores', ti_now['cores'][len(ti_now['cores']) // 2], cpu_tier_now)} cores",
    "gpu": f"{gpu_model_now} · {val('vram', META['gpu_model_info'][gpu_model_now]['vram_mode'], gpu_model_now)} GB",
    "ram": f"{val('ram', 16)} GB",
    "storage": f"{val('sto_gb', 512)} GB {val('sto_type', 'NVMe')}",
}

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.markdown(
        f"""<div class="brand"><span class="dot"></span>TechValue AI</div>
<div class="hero-title">Build Your PC.<br><span>Discover Its Value.</span></div>
<p class="hero-copy">Choose the processor, graphics card, memory, storage and screen. A neural-network regression model
trained on {META['trained_on_rows']:,} computers estimates the price in seconds.</p>
<div class="chips">
<span class="chip"><b>{META['trained_on_rows']:,}</b> computers learned from</span>
<span class="chip">Typical error <b>±{METRICS['mae_mean']:,.0f}</b> (MAE)</span>
<span class="chip">R² <b>{METRICS['r2_mean']:.3f}</b></span>
</div>""", unsafe_allow_html=True)
with right:
    try:
        components.html(core.case_html(tags), height=420)
    except Exception:                                       # illustration is decorative; never block the app
        st.caption("PC illustration unavailable.")

st.write("")
b1, b2, b3, _ = st.columns([1.1, 1.1, 1.1, 2.7])
with b1:
    st.button("Load gaming laptop", on_click=apply_sample, args=("Gaming laptop",), key="smp1")
with b2:
    st.button("Load workstation", on_click=apply_sample, args=("Workstation desktop",), key="smp2")
with b3:
    st.button("Load budget laptop", on_click=apply_sample, args=("Budget laptop",), key="smp3")

# =================================================================================================
# Configurator
# =================================================================================================
st.markdown("## Configure your build")
cfg = {}

# ---------------------------------------------------------------- System
with st.container(key="panel_sys"):
    st.markdown('<div class="ptitle t-gry">System</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        cfg["device_type"] = w_radio("Device type", CAT["device_type"], k("device"), "Laptop", horizontal=True)
    with c2:
        cfg["brand"] = w_select("Brand", CAT["brand"], k("brand"), "Dell")
    with c3:
        cfg["os"] = w_select("Operating system", CAT["os"], k("os"), "Windows")
    with c4:
        forms = META["valid_by_device"][cfg["device_type"]]["form_factor"]
        cfg["form_factor"] = w_select("Form factor", forms, k("form", cfg["device_type"]))
    with c5:
        years = [int(v) for v in NUM["release_year"]["values"]]
        cfg["release_year"] = w_select_slider("Release year", years, k("year"), 2023)

DEV = META["valid_by_device"][cfg["device_type"]]
is_laptop = cfg["device_type"] == "Laptop"

# ---------------------------------------------------------------- CPU | GPU
col_cpu, col_gpu = st.columns(2, gap="medium")
with col_cpu, st.container(key="panel_cpu"):
    st.markdown('<div class="ptitle t-cyan">CPU</div>', unsafe_allow_html=True)
    cpu_brands = ["Apple"] if cfg["brand"] == "Apple" else [b for b in CAT["cpu_brand"] if b != "Apple"]
    x1, x2 = st.columns(2)
    with x1:
        cfg["cpu_brand"] = w_select("CPU brand", cpu_brands, k("cpu_brand", cfg["brand"] == "Apple"))
    with x2:
        tiers = [int(v) for v in NUM["cpu_tier"]["values"]]
        cfg["cpu_tier"] = w_select_slider("CPU tier (1 entry, 6 top)", tiers, k("cpu_tier"), 3)
    ti = META["cpu_tier_info"][str(cfg["cpu_tier"])]
    x1, x2 = st.columns(2)
    with x1:
        cfg["cpu_cores"] = w_select("Cores", ti["cores"], k("cores", cfg["cpu_tier"]), ti["cores"][len(ti["cores"]) // 2])
    with x2:
        threads = [t for t in ti["threads"] if t >= cfg["cpu_cores"]] or ti["threads"]
        cfg["cpu_threads"] = w_select("Threads", threads, k("threads", f'{cfg["cpu_tier"]}-{cfg["cpu_cores"]}'), 2 * cfg["cpu_cores"])
    x1, x2 = st.columns(2)
    with x1:
        cfg["cpu_base_ghz"] = w_select("Base clock (GHz)", ti["base_ghz"], k("base", cfg["cpu_tier"]))
    with x2:
        boosts = [b for b in ti["boost_ghz"] if b >= cfg["cpu_base_ghz"]] or ti["boost_ghz"]
        cfg["cpu_boost_ghz"] = w_select("Boost clock (GHz)", boosts, k("boost", f'{cfg["cpu_tier"]}-{cfg["cpu_base_ghz"]}'),
                                        boosts[len(boosts) // 2])

with col_gpu, st.container(key="panel_gpu"):
    st.markdown('<div class="ptitle t-mag">GPU</div>', unsafe_allow_html=True)
    x1, x2 = st.columns(2)
    with x1:
        cfg["gpu_brand"] = w_select("GPU brand", CAT["gpu_brand"], k("gpu_brand"), "NVIDIA")
    models = sorted(m for m, i in META["gpu_model_info"].items() if i["gpu_brand"] == cfg["gpu_brand"])
    with x2:
        cfg["gpu_model"] = w_select("GPU model", models, k("gpu_model", cfg["gpu_brand"]), models[len(models) // 2])
    gi = META["gpu_model_info"][cfg["gpu_model"]]
    x1, x2 = st.columns(2)
    with x1:
        cfg["gpu_tier"] = w_select_slider("GPU tier (1 entry, 6 top)", [1, 2, 3, 4, 5, 6], k("gpu_tier", cfg["gpu_model"]), gi["gpu_tier"])
    with x2:
        cfg["vram_gb"] = w_select("VRAM (GB)", gi["vram_options"], k("vram", cfg["gpu_model"]), gi["vram_mode"])
    st.markdown('<div class="hint">Tier and VRAM start at the most common values for the chosen GPU model.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- RAM | Storage
col_ram, col_sto = st.columns(2, gap="medium")
with col_ram, st.container(key="panel_ram"):
    st.markdown('<div class="ptitle t-pur">Memory</div>', unsafe_allow_html=True)
    ram_opts = [int(v) for v in NUM["ram_gb"]["values"]]
    cfg["ram_gb"] = w_select_slider("RAM capacity (GB)", ram_opts, k("ram"), 16)
    st.markdown('<div class="hint">Only RAM sizes that exist in the dataset are offered.</div>', unsafe_allow_html=True)

with col_sto, st.container(key="panel_sto"):
    st.markdown('<div class="ptitle t-grn">Storage</div>', unsafe_allow_html=True)
    x1, x2 = st.columns(2)
    with x1:
        cfg["storage_type"] = w_select("Storage type", CAT["storage_type"], k("sto_type"), "NVMe")
    with x2:
        n_opts = [int(v) for v in NUM["storage_drive_count"]["values"]]
        cfg["storage_drive_count"] = w_select("Number of drives", n_opts, k("sto_n"), 1)
    sg_opts = [int(v) for v in NUM["storage_gb"]["values"]]
    cfg["storage_gb"] = w_select_slider("Capacity (GB)", sg_opts, k("sto_gb"), 512)

# ---------------------------------------------------------------- Display
with st.container(key="panel_dsp"):
    st.markdown('<div class="ptitle t-gry">Display</div>', unsafe_allow_html=True)
    x1, x2, x3, x4 = st.columns(4)
    with x1:
        cfg["display_type"] = w_select("Panel type", CAT["display_type"], k("dsp_type"), "IPS")
    with x2:
        cfg["resolution"] = w_select("Resolution", CAT["resolution"], k("res"), "1920x1080")
    with x3:
        hz_opts = [int(v) for v in NUM["refresh_hz"]["values"]]
        cfg["refresh_hz"] = w_select("Refresh rate (Hz)", hz_opts, k("hz"), 60)
    with x4:
        sizes = DEV["display_size_in"]
        cfg["display_size_in"] = w_select("Screen size (in)", sizes, k("size", cfg["device_type"]), sizes[len(sizes) // 2])

# ---------------------------------------------------------------- Advanced
with st.expander("More specifications (battery, power, connectivity, weight, warranty)"):
    st.caption("Sensible defaults come from the dataset. Battery and charger apply to laptops; power supply to desktops.")
    a1, a2, a3, a4 = st.columns(4)
    d = META["defaults_by_device"][cfg["device_type"]]
    with a1:
        if is_laptop:
            lo, hi = [int(v) for v in DEV["battery_wh"]]
            cfg["battery_wh"] = w_slider("Battery (Wh)", lo, hi, k("battery", "Laptop"), int(d["battery_wh"]))
        else:
            cfg["battery_wh"] = 0
            st.markdown("Battery: none (desktop)")
    with a2:
        if is_laptop:
            lo, hi = [int(v) for v in DEV["charger_watts"]]
            cfg["charger_watts"] = w_slider("Charger (W)", lo, hi, k("charger", "Laptop"), int(d["charger_watts"]), step=5)
        else:
            cfg["charger_watts"] = 0
            st.markdown("Charger: none (desktop)")
    with a3:
        if not is_laptop:
            lo, hi = [int(v) for v in DEV["psu_watts"]]
            cfg["psu_watts"] = w_slider("Power supply (W)", lo, hi, k("psu", "Desktop"), int(d["psu_watts"]), step=50)
        else:
            cfg["psu_watts"] = 0
            st.markdown("Power supply: none (laptop)")
    with a4:
        lo, hi = DEV["weight_kg"]
        cfg["weight_kg"] = w_slider("Weight (kg)", float(lo), float(hi), k("weight", cfg["device_type"]), float(d["weight_kg"]), step=0.1)
    a1, a2, a3 = st.columns(3)
    with a1:
        cfg["wifi"] = w_select("Wi-Fi", CAT["wifi"], k("wifi"), "Wi-Fi 6")
    with a2:
        bt = [float(v) for v in NUM["bluetooth"]["values"]]
        cfg["bluetooth"] = w_select("Bluetooth version", bt, k("bt"), 5.2)
    with a3:
        wr = [int(v) for v in NUM["warranty_months"]["values"]]
        cfg["warranty_months"] = w_select("Warranty (months)", wr, k("warranty"), 24)

# =================================================================================================
# Predict / reset
# =================================================================================================
st.write("")
p1, p2 = st.columns([4, 1])
with p1:
    predict_clicked = st.button("Predict My PC Price", key="predict_btn")
with p2:
    st.button("Reset", on_click=clear_config, key="reset_btn")

if predict_clicked:
    errors, warns = core.validate_config(cfg, META)
    if errors:
        st.session_state.pop("result", None)
        for e in errors:
            st.error(e)
    else:
        try:
            row = core.build_input_frame(cfg, META)
            price = core.predict_price(MODEL, row)
            st.session_state["result"] = {"price": price, "config": dict(cfg), "warnings": warns}
        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"The model could not produce an estimate ({type(exc).__name__}). Please check the inputs and try again.")

# =================================================================================================
# Result
# =================================================================================================
res = st.session_state.get("result")
if res:
    price, rcfg = res["price"], res["config"]
    st.markdown("## Your estimate")
    for w in res["warnings"]:
        st.warning(w)

    r1, r2 = st.columns([1.15, 1], gap="large")
    with r1:
        pct = core.price_percentile(price, META)
        mae = METRICS["mae_mean"]
        st.markdown(
            f"""<div class="result"><div class="label">Estimated price (dataset price units)</div>
<div class="price">{core.format_price(price)}<span class="unit">price units</span></div>
<div class="label">Typical error of this model: about ±{mae:,.0f} (mean absolute error, validation).
About {pct:.0f}% of the computers in the dataset cost less.</div>
<div class="note">This is a machine-learning <b>estimate</b>, not a guaranteed market price. The dataset does not state a currency.</div></div>""",
            unsafe_allow_html=True)

        # ---- optional currency view (display only; the model output above never changes)
        st.markdown("#### Currency view (optional)")
        st.caption("The dataset files do not state a currency. Pick the currency you believe the prices are in "
                   "(USD is a common assumption, but it is only an assumption), then the currency to show.")
        c_a, c_b = st.columns(2)
        with c_a:
            base = st.selectbox("Dataset prices are in", ["Not specified (no conversion)"] + core.CURRENCIES,
                                index=1, key="fx_base")
        if base != "Not specified (no conversion)":
            with c_b:
                target = st.selectbox("Show estimate in", core.CURRENCIES, index=core.CURRENCIES.index("PKR"), key="fx_target")
            if base == target:
                st.info("Both currencies are the same, so no conversion is applied. Choose a different 'Show estimate in' currency.")
            else:
                live = get_live_rate(base, target)
                if live:
                    rate, source_txt = live["rate"], f"Live rate from {live['source']} - updated {live['date']}"
                else:
                    st.info("Live exchange rates are unavailable right now. Enter a rate yourself - it is only used for this display.")
                    rate = st.number_input(f"Manual rate: 1 {base} = ? {target}", min_value=0.0, value=0.0, step=0.01,
                                           format="%.4f", key="fx_manual")
                    source_txt = "Manual rate entered by you (not a live rate)"
                if rate and rate > 0:
                    st.markdown(f"**Original estimate:** {core.format_price(price)} {base} (assumed)  \n"
                                f"**Converted:** {core.format_price(core.convert(price, rate))} {target}  \n"
                                f"<span style='color:{core.MUTED};font-size:.85rem'>Exchange rate used: 1 {base} = {rate:,.4f} {target}. "
                                f"{source_txt}. Only the display changes; the model's prediction stays {core.format_price(price)} price units.</span>",
                                unsafe_allow_html=True)
                else:
                    st.caption("Enter a rate above 0 to see the converted amount. The original estimate stays unchanged.")
    with r2:
        fig = core.gauge_figure(price, META)
        st.pyplot(fig)
        plt.close(fig)

    t1, t2, t3 = st.tabs(["Compare with the market", "Where it sits", "Your specs vs the dataset"])
    with t1:
        fig = core.reference_bar_figure(price, core.reference_prices(rcfg, META), META)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Reference bars are real averages/medians of dataset computers that share one specification with your build.")
    with t2:
        fig = core.distribution_figure(price, META)
        st.pyplot(fig)
        plt.close(fig)
    with t3:
        fig = core.feature_percentile_figure(rcfg, META)
        st.pyplot(fig)
        plt.close(fig)

# =================================================================================================
# Model performance, how it works, limitations
# =================================================================================================
st.markdown("## Model performance")
m1, m2, m3, m4 = st.columns(4)
for col, key, label, fmt in [(m1, "mae_mean", "MAE (average error)", "{:,.1f}"), (m2, "rmse_mean", "RMSE", "{:,.1f}"),
                             (m3, "r2_mean", "R² score", "{:.4f}"), (m4, None, "Selected model", None)]:
    with col:
        value = METRICS["model"] if key is None else fmt.format(METRICS[key])
        size = "1.15rem" if key is None else "1.7rem"
        st.markdown(f'<div class="metric-card"><div class="v" style="font-size:{size}">{value}</div><div class="k">{label}</div></div>',
                    unsafe_allow_html=True)
st.caption(f"Measured with {META['metrics']['validation']}. RMSE = {METRICS['rmse_mean']:,.1f} "
           f"(±{METRICS['rmse_std']:,.1f} across folds), about {METRICS['rmse_pct_of_avg_price']:.1f}% of the average price.")

tabs = st.tabs(["Algorithm comparison", "Actual vs predicted", "What the model relies on"])
with tabs[0]:
    fig = core.model_comparison_figure(META)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(META["metrics"]["svr_note"])
with tabs[1]:
    fig = core.scatter_figure(META)
    st.pyplot(fig)
    plt.close(fig)
with tabs[2]:
    fig = core.importance_figure(META)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Permutation importance shows which inputs the model uses most. It does not prove that changing a part "
               "causes the price to change in the real market.")

st.markdown("## How it works")
h1, h2, h3 = st.columns(3)
with h1:
    st.markdown("**1. You choose the hardware**  \nEvery input is one of the 30 specifications the model learned from.")
with h2:
    st.markdown("**2. The pipeline prepares it**  \nThe saved scikit-learn pipeline scales numbers and one-hot encodes categories exactly as in training.")
with h3:
    st.markdown(f"**3. The model estimates the price**  \nA {METRICS['model']} trained on {META['trained_on_rows']:,} computers predicts the price.")

with st.expander("Limitations and uncertainty"):
    st.markdown(
        "- The data is a Kaggle competition dataset (Computer Prices 2025). It may be partly synthetic, so results show how the "
        "model behaves on this dataset, not on the real market.\n"
        "- The dataset files do not state a currency, so prices are shown in *price units*. The currency view is an optional assumption.\n"
        "- Only the 30 listed specifications are used. Market conditions, region, seller, condition and exact model names are not.\n"
        "- Estimates are less reliable for unusual combinations and for very expensive builds, where there is less training data.\n"
        "- Treat every result as an estimate with an error of roughly the MAE shown above - or more.")

st.markdown(
    '<div class="footer">TechValue AI is an educational project (COMSATS University Islamabad, Lahore Campus - AIC354 Machine Learning '
    'Fundamentals) by M. Ahmed Waqar, FA24-BSE-070. Data: Kaggle <i>Computer Prices 2025</i> competition. '
    'Predictions are estimates, not financial advice.</div>', unsafe_allow_html=True)