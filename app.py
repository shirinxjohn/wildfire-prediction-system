import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ══════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Wildfire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0a0b0e !important;
    color: #f0ece4 !important;
}
.stApp { background: #0a0b0e; }

section[data-testid="stSidebar"] {
    background: #111317 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #c0bdb5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f0ece4 !important;
    font-family: 'Syne', sans-serif !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #ff4a1c !important;
    border-color: #ff4a1c !important;
}

.wf-header {
    background: linear-gradient(135deg, #181b21 0%, #1e2129 100%);
    border: 1px solid rgba(255,74,28,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.wf-header-icon { font-size: 3rem; line-height: 1; }
.wf-header-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #f0ece4 !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
}
.wf-header-sub { color: #8a8a8a !important; font-size: 0.9rem; margin-top: 4px; }

.metric-card {
    background: #111317;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #f0ece4;
}
.metric-unit { font-size: 0.9rem; color: #888; }

.risk-banner-high {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.risk-banner-low {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.risk-title-high { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #ff6b6b; }
.risk-title-low  { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #4ade80; }
.risk-prob { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #888; margin-top: 6px; }

.gauge-wrap {
    margin-top: 12px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.gauge-fill-high { height: 100%; background: #ef4444; border-radius: 4px; transition: width 0.5s; }
.gauge-fill-low  { height: 100%; background: #22c55e; border-radius: 4px; transition: width 0.5s; }

.badge-low      { display:inline-block; background:rgba(34,197,94,0.15);  color:#4ade80; border:1px solid rgba(34,197,94,0.3);  border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:500; }
.badge-moderate { display:inline-block; background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3); border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:500; }
.badge-severe   { display:inline-block; background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(239,68,68,0.3);  border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:500; }

.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555;
    margin: 1.5rem 0 0.75rem;
    font-weight: 500;
}

.loc-card {
    background: #111317;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}
.loc-rank {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem; font-weight: 500; flex-shrink: 0;
}
.loc-name { font-size: 0.9rem; font-weight: 500; color: #f0ece4; }
.loc-meta { font-size: 0.75rem; color: #666; margin-top: 2px; }
.loc-pct  { font-family: 'DM Mono', monospace; font-size: 1rem; font-weight: 500; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #666 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #ff4a1c !important;
    border-bottom-color: #ff4a1c !important;
}

.stButton > button {
    background: #ff4a1c !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f0ece4 !important; }
p, li, span, div { color: #c0bdb5; }
.stMarkdown p { color: #c0bdb5 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load("xgb_wildfire_model.joblib")

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ══════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════
st.markdown("""
<div class="wf-header">
  <div class="wf-header-icon">🔥</div>
  <div>
    <div class="wf-header-title">Wildfire Risk Prediction</div>
    <div class="wf-header-sub">Environmental parameter analysis &nbsp;·&nbsp; XGBoost model &nbsp;·&nbsp; Real-time risk scoring</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ Model file `xgb_wildfire_model.joblib` not found. Showing UI preview with simulated predictions.")

# ══════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Parameters")
    st.markdown("---")

    st.markdown("**📍 Current Conditions**")
    NDVI               = st.slider("NDVI",                    0.0,   1.0,   0.5,  0.01)
    LST                = st.slider("Surface Temp (°C)",       20.0,  60.0,  35.0, 0.5)
    rainfall           = st.slider("Rainfall (mm)",           0.0,   50.0,  10.0, 0.5)
    drought_index      = st.slider("Drought Index",           0.0,   100.0, 20.0, 1.0)

    st.markdown("---")
    st.markdown("**🗺️ Location Factors**")
    population_density = st.slider("Population Density",      0,     5000,  1000, 10)
    road_proximity_km  = st.slider("Road Proximity (km)",     0.0,   20.0,  5.0,  0.1)

    st.markdown("---")
    st.markdown("**📅 3-Day Averages**")
    LST_3d_avg         = st.slider("LST 3d Avg (°C)",         20.0,  60.0,  35.0, 0.5)
    rainfall_3d_avg    = st.slider("Rainfall 3d Avg",         0.0,   50.0,  10.0, 0.5)
    NDVI_3d_avg        = st.slider("NDVI 3d Avg",             0.0,   1.0,   0.5,  0.01)

    st.markdown("---")
    st.markdown("**📆 7-Day Averages & Variability**")
    LST_7d_avg         = st.slider("LST 7d Avg (°C)",         20.0,  60.0,  35.0, 0.5)
    rainfall_7d_avg    = st.slider("Rainfall 7d Avg",         0.0,   50.0,  10.0, 0.5)
    NDVI_7d_avg        = st.slider("NDVI 7d Avg",             0.0,   1.0,   0.5,  0.01)
    LST_7d_std         = st.slider("LST Variability",         0.0,   10.0,  2.0,  0.1)
    rainfall_7d_std    = st.slider("Rainfall Variability",    0.0,   10.0,  2.0,  0.1)
    NDVI_7d_std        = st.slider("NDVI Variability",        0.0,   0.5,   0.1,  0.01)

    st.markdown("---")
    day_of_year        = st.slider("Day of Year",             1,     365,   150,  1)

# ══════════════════════════════════════════
#  INPUT DATAFRAME
# ══════════════════════════════════════════
input_data = pd.DataFrame({
    'NDVI':               [NDVI],
    'LST':                [LST],
    'rainfall':           [rainfall],
    'population_density': [population_density],
    'road_proximity_km':  [road_proximity_km],
    'LST_3d_avg':         [LST_3d_avg],
    'rainfall_3d_avg':    [rainfall_3d_avg],
    'NDVI_3d_avg':        [NDVI_3d_avg],
    'LST_7d_avg':         [LST_7d_avg],
    'rainfall_7d_avg':    [rainfall_7d_avg],
    'NDVI_7d_avg':        [NDVI_7d_avg],
    'LST_7d_std':         [LST_7d_std],
    'rainfall_7d_std':    [rainfall_7d_std],
    'NDVI_7d_std':        [NDVI_7d_std],
    'drought_index':      [drought_index],
    'day_of_year':        [day_of_year],
})

# ══════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔥 Risk Assessment", "📍 Top Risk Locations", "📈 Trend Analysis"])

# ────────────────────────────────────────
#  TAB 1 — RISK ASSESSMENT
# ────────────────────────────────────────
with tab1:

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">NDVI</div>
            <div class="metric-value">{NDVI:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Surface Temp</div>
            <div class="metric-value">{LST:.1f}<span class="metric-unit">°C</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Rainfall</div>
            <div class="metric-value">{rainfall:.1f}<span class="metric-unit">mm</span></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Drought Index</div>
            <div class="metric-value">{drought_index:.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    if st.button("🔥 Predict Wildfire Risk", use_container_width=True):

        if model_loaded:
            prediction  = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        else:
            score       = ((LST - 20) / 40 * 35) + ((1 - NDVI) * 20) + (drought_index / 100 * 25) + (max(0, (10 - rainfall) / 10) * 15)
            probability = min(0.99, max(0.01, score / 100))
            prediction  = 1 if probability >= 0.5 else 0

        pct = probability * 100

        veg_loss = max(0, (NDVI_7d_avg - NDVI) / (NDVI_7d_avg + 0.001) * 100)
        if veg_loss < 10:
            severity = "Low";      sev_class = "badge-low"
        elif veg_loss < 30:
            severity = "Moderate"; sev_class = "badge-moderate"
        else:
            severity = "Severe";   sev_class = "badge-severe"

        if prediction == 1:
            st.markdown(f"""
            <div class="risk-banner-high">
              <div class="risk-title-high">⚠️ HIGH FIRE RISK DETECTED</div>
              <div class="risk-prob">Model confidence: {pct:.1f}% probability of wildfire</div>
              <div class="gauge-wrap"><div class="gauge-fill-high" style="width:{pct:.1f}%"></div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner-low">
              <div class="risk-title-low">✅ LOW FIRE RISK</div>
              <div class="risk-prob">Safety confidence: {(100 - pct):.1f}%</div>
              <div class="gauge-wrap"><div class="gauge-fill-low" style="width:{(100-pct):.1f}%"></div></div>
            </div>""", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{pct:.0f}<span class="metric-unit">/100</span></div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Vegetation Loss</div>
                <div class="metric-value">{veg_loss:.1f}<span class="metric-unit">%</span></div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Damage Severity</div>
                <div class="metric-value" style="font-size:1.4rem">{severity}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-label">Risk Score Gauge</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#111317')
            ax.set_facecolor('#111317')
            bar_color = '#ef4444' if pct >= 50 else '#22c55e'
            ax.barh(['Risk'], [pct],  color=bar_color, height=0.45, zorder=3)
            ax.barh(['Risk'], [100],  color='#1e2129', height=0.45, zorder=2)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Risk %', color='#666', fontsize=10)
            ax.tick_params(colors='#888', labelsize=10)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            ax.text(pct + 1.5, 0, f'{pct:.1f}%', va='center', color='#f0ece4',
                    fontsize=13, fontweight='bold', fontfamily='monospace')
            ax.grid(axis='x', color=(1, 1, 1, 0.05), linewidth=0.5, zorder=1)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_right:
            st.markdown('<div class="section-label">Contributing Factors</div>', unsafe_allow_html=True)
            factors = {
                'Surface Temp':     min(100, (LST - 20) / 40 * 100),
                'Vegetation':       min(100, (1 - NDVI) * 100),
                'Drought':          drought_index,
                'Rainfall Deficit': max(0, (10 - rainfall) / 10 * 100),
                'Temp Variability': LST_7d_std / 10 * 100,
                'NDVI Variability': NDVI_7d_std / 0.5 * 100,
            }
            sorted_f   = dict(sorted(factors.items(), key=lambda x: x[1], reverse=True))
            bar_colors = ['#ff4a1c','#ff7043','#f59e0b','#fbbf24','#22c55e','#14b8a6']

            fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor='#111317')
            ax2.set_facecolor('#111317')
            y_pos = range(len(sorted_f))
            ax2.barh(list(y_pos), list(sorted_f.values()), color=bar_colors, height=0.55, zorder=3)
            ax2.set_yticks(list(y_pos))
            ax2.set_yticklabels(list(sorted_f.keys()), color='#aaa', fontsize=9)
            ax2.set_xlim(0, 110)
            ax2.tick_params(axis='x', colors='#666', labelsize=9)
            for spine in ax2.spines.values():
                spine.set_edgecolor('#333')
            for i, (k, v) in enumerate(sorted_f.items()):
                ax2.text(v + 1.5, i, f'{v:.0f}%', va='center', color='#888',
                         fontsize=8.5, fontfamily='monospace')
            ax2.grid(axis='x', color=(1, 1, 1, 0.04), linewidth=0.5, zorder=1)
            ax2.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ────────────────────────────────────────
#  TAB 2 — TOP RISK LOCATIONS
# ────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Highest risk zones — sample dataset</div>', unsafe_allow_html=True)

    locations = [
        {"rank": 1, "name": "Seshachalam Hills, AP", "meta": "13.6°N · 79.2°E · Dense forest",  "pct": 87, "sev": "Critical", "color": "#ef4444", "bg": "rgba(239,68,68,0.12)"},
        {"rank": 2, "name": "Nallamala Forest, TG",  "meta": "15.4°N · 79.0°E · Dry deciduous", "pct": 72, "sev": "High",     "color": "#f59e0b", "bg": "rgba(245,158,11,0.12)"},
        {"rank": 3, "name": "Kadapa District",        "meta": "14.5°N · 78.8°E · Scrubland",     "pct": 61, "sev": "High",     "color": "#f97316", "bg": "rgba(249,115,22,0.12)"},
        {"rank": 4, "name": "Kurnool Outskirts",      "meta": "15.8°N · 78.1°E · Semi-arid",     "pct": 43, "sev": "Moderate", "color": "#eab308", "bg": "rgba(234,179,8,0.10)"},
        {"rank": 5, "name": "Tirupati Foothills",     "meta": "13.7°N · 79.4°E · Shrubland",     "pct": 28, "sev": "Low",      "color": "#22c55e", "bg": "rgba(34,197,94,0.10)"},
    ]

    for loc in locations:
        st.markdown(f"""
        <div class="loc-card" style="border-color:{loc['color']}22">
          <div class="loc-rank" style="background:{loc['bg']}; color:{loc['color']}">{loc['rank']}</div>
          <div style="flex:1">
            <div class="loc-name">{loc['name']}</div>
            <div class="loc-meta">{loc['meta']}</div>
          </div>
          <div style="text-align:right">
            <div class="loc-pct" style="color:{loc['color']}">{loc['pct']}%</div>
            <div style="font-size:0.75rem; color:{loc['color']}; margin-top:2px">{loc['sev']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Risk comparison</div>', unsafe_allow_html=True)

    names = [l["name"].split(",")[0] for l in locations]
    pcts  = [l["pct"] for l in locations]
    cols  = [l["color"] for l in locations]

    fig3, ax3 = plt.subplots(figsize=(8, 3.5), facecolor='#111317')
    ax3.set_facecolor('#111317')
    bars = ax3.bar(names, pcts, color=cols, width=0.55, zorder=3)
    ax3.set_ylim(0, 110)
    ax3.set_ylabel('Risk %', color='#666', fontsize=10)
    ax3.tick_params(colors='#888', labelsize=9)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#2a2a2a')
    for bar, p in zip(bars, pcts):
        ax3.text(bar.get_x() + bar.get_width() / 2, p + 1.5, f'{p}%',
                 ha='center', va='bottom', color='#aaa', fontsize=9, fontfamily='monospace')
    ax3.grid(axis='y', color=(1, 1, 1, 0.04), linewidth=0.5, zorder=1)
    ax3.axhline(50, color='#ef4444', linewidth=0.8, linestyle='--', alpha=0.5, zorder=4)
    ax3.text(4.55, 52, 'high risk threshold', color='#ef4444', fontsize=8, alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ────────────────────────────────────────
#  TAB 3 — TREND ANALYSIS
# ────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-label">30-day risk trend (simulated)</div>', unsafe_allow_html=True)

    np.random.seed(42)
    days  = np.arange(1, 31)
    trend = np.clip(30 + days * 0.9 + np.random.randn(30) * 8, 5, 95)

    fig4, ax4 = plt.subplots(figsize=(9, 3.5), facecolor='#111317')
    ax4.set_facecolor('#111317')
    ax4.fill_between(days, trend, alpha=0.15, color='#ff4a1c')
    ax4.plot(days, trend, color='#ff4a1c', linewidth=2, zorder=3)
    ax4.scatter(days, trend, color='#ff4a1c', s=18, zorder=4)
    ax4.axhline(50, color='#ef4444', linewidth=0.8, linestyle='--', alpha=0.4)
    ax4.set_xlabel('Day', color='#666', fontsize=10)
    ax4.set_ylabel('Risk %', color='#666', fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.tick_params(colors='#888', labelsize=9)
    for spine in ax4.spines.values():
        spine.set_edgecolor('#2a2a2a')
    ax4.grid(color=(1, 1, 1, 0.04), linewidth=0.5, zorder=1)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model — feature importance</div>', unsafe_allow_html=True)

    feat_importance = {
        'Surface Temp (LST)':  0.88,
        'Drought Index':       0.74,
        'NDVI':                0.68,
        'Rainfall':            0.55,
        'LST 7d Average':      0.49,
        'NDVI 7d Average':     0.43,
        'Road Proximity':      0.38,
        'Rainfall 7d Avg':     0.31,
        'Population Density':  0.22,
        'Day of Year':         0.16,
    }

    fi_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(feat_importance)))

    fig5, ax5 = plt.subplots(figsize=(9, 4), facecolor='#111317')
    ax5.set_facecolor('#111317')
    y = range(len(feat_importance))
    ax5.barh(list(y), list(feat_importance.values()), color=fi_colors, height=0.6, zorder=3)
    ax5.set_yticks(list(y))
    ax5.set_yticklabels(list(feat_importance.keys()), color='#aaa', fontsize=10)
    ax5.set_xlim(0, 1.1)
    ax5.tick_params(axis='x', colors='#666', labelsize=9)
    for spine in ax5.spines.values():
        spine.set_edgecolor('#2a2a2a')
    for i, (k, v) in enumerate(feat_importance.items()):
        ax5.text(v + 0.01, i, f'{v:.2f}', va='center', color='#888',
                 fontsize=8.5, fontfamily='monospace')
    ax5.grid(axis='x', color=(1, 1, 1, 0.04), linewidth=0.5, zorder=1)
    ax5.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

# ── Footer ──
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#444; font-size:0.8rem;">Built with Python · Streamlit · XGBoost · Matplotlib</p>',
    unsafe_allow_html=True
)