# app.py
# ------------------------------------------------------------
# BSES Transformer & Distribution AI Agent Demo (Streamlit)
# ------------------------------------------------------------
# Run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
# Deploy on Streamlit Community Cloud:
#   Push this folder to GitHub and select app.py as the entry point.
# ------------------------------------------------------------

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="BSES Distribution AI Agent Demo",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1500px;}
      .stMetric {padding: 6px 10px; border-radius: 12px;}
      div[data-testid="stVerticalBlockBorderWrapper"] {padding: 10px;}
      .tight-card {padding: 13px 15px; border-radius: 16px; border: 1px solid rgba(49,51,63,0.14); background: rgba(250,250,250,0.025);}
      .muted {opacity: 0.72;}
      .small {font-size: 0.92rem;}
      .pill {display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.18); margin-right: 6px;}
      .redpill {background: rgba(255, 75, 75, 0.10);}
      .amberpill {background: rgba(255, 180, 0, 0.12);}
      .greenpill {background: rgba(0, 180, 120, 0.10);}
      .hero-title {font-size: 2.1rem; font-weight: 750; line-height: 1.1; margin-bottom: .2rem;}
      .section-note {font-size: .92rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='hero-title'>⚡ BSES Transformer & Distribution AI Agent Demo</div>", unsafe_allow_html=True)
st.caption(
    "Synthetic live telemetry for distribution transformers, feeders and LT circuits | AI anomaly detection | predictive maintenance | GenBI diagnostics."
)


# ============================================================
# Demo metadata / network model
# ============================================================
@dataclass(frozen=True)
class Asset:
    zone: str
    substation: str
    transformer: str
    capacity_kva: int
    feeder: str
    dt_count: int
    consumers: int
    baseline_loss_pct: float
    lat: float
    lon: float


ASSETS = [
    Asset("South Delhi", "Nehru Place 33/11kV", "DT-NSP-014 | 990 kVA", 990, "FDR-NP-07", 18, 5200, 7.8, 28.548, 77.251),
    Asset("West Delhi", "Janakpuri 66/11kV", "DT-JKP-031 | 630 kVA", 630, "FDR-JK-04", 22, 6100, 8.9, 28.621, 77.087),
    Asset("Central Delhi", "Patel Nagar 33/11kV", "DT-PTN-009 | 400 kVA", 400, "FDR-PN-11", 15, 3700, 10.2, 28.650, 77.169),
    Asset("East Delhi", "Laxmi Nagar 33/11kV", "DT-LXN-022 | 630 kVA", 630, "FDR-LX-03", 24, 7400, 9.6, 28.635, 77.279),
    Asset("South-West Delhi", "Dwarka 66/11kV", "DT-DWK-048 | 990 kVA", 990, "FDR-DW-15", 26, 8200, 7.2, 28.592, 77.046),
]

SCENARIOS = {
    "Normal urban load": {
        "stress": 0.75,
        "heat": 0.6,
        "tamper": 0.25,
        "fault": 0.25,
        "label": "Balanced demand with mild local noise",
    },
    "Heatwave evening peak": {
        "stress": 1.55,
        "heat": 1.65,
        "tamper": 0.45,
        "fault": 0.55,
        "label": "AC load surge, thermal stress, voltage sag risk",
    },
    "High AT&C loss pocket": {
        "stress": 1.05,
        "heat": 0.85,
        "tamper": 1.65,
        "fault": 0.75,
        "label": "Feeder energy mismatch and theft/tamper signatures",
    },
    "Cable fault developing": {
        "stress": 1.1,
        "heat": 1.0,
        "tamper": 0.55,
        "fault": 1.75,
        "label": "Rising neutral current, intermittent sag, breaker trip risk",
    },
}


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Demo Controls")
    autoplay = st.toggle("Autoplay telemetry", value=True)
    tick_ms = st.slider("Refresh speed (ms)", 150, 1500, 380, 10)

    st.divider()
    selected_asset_label = st.selectbox(
        "Select transformer / feeder",
        [f"{a.zone} · {a.substation} · {a.transformer}" for a in ASSETS],
        index=0,
    )
    selected_asset = ASSETS[[f"{a.zone} · {a.substation} · {a.transformer}" for a in ASSETS].index(selected_asset_label)]

    scenario_name = st.selectbox("Operating scenario", list(SCENARIOS.keys()), index=1)
    scenario = SCENARIOS[scenario_name]
    st.caption(scenario["label"])

    st.divider()
    st.subheader("Signal realism")
    sensor_noise = st.slider("Sensor noise", 0.0, 3.0, 0.7, 0.1)
    degradation = st.slider("Asset degradation drift", 0.0, 2.5, 0.9, 0.1)
    telemetry_drop = st.slider("Telemetry dropout %", 0, 12, 2, 1)

    st.divider()
    st.subheader("Design / operating limits")
    design_load = st.slider("Transformer safe loading %", 65, 110, 85, 1)
    oil_temp_limit = st.slider("Oil temperature limit °C", 65, 105, 85, 1)
    winding_temp_limit = st.slider("Winding temperature limit °C", 75, 130, 105, 1)
    voltage_band = st.slider("Voltage deviation tolerance %", 3.0, 12.0, 6.0, 0.5)
    loss_baseline = st.slider("Allowed feeder loss %", 4.0, 18.0, selected_asset.baseline_loss_pct, 0.1)


# ============================================================
# Synthetic telemetry generation
# ============================================================
def make_distribution_series(asset: Asset, scenario: dict, n: int = 420, noise: float = 0.7, degradation: float = 0.9, dropout: int = 2):
    seed = abs(hash((asset.transformer, scenario_name))) % (10**7)
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Urban electricity load profile: morning bump + evening AC peak + short oscillations.
    evening_peak = np.exp(-((t - 250) / 65) ** 2)
    morning_peak = 0.45 * np.exp(-((t - 110) / 55) ** 2)
    cyclic = 0.08 * np.sin(2 * np.pi * t / 48) + 0.05 * np.sin(2 * np.pi * t / 17)

    load_frac = 0.48 + morning_peak + 0.62 * evening_peak + cyclic
    load_frac += scenario["stress"] * 0.055 * rng.normal(0, 1, n)
    load_frac = np.clip(load_frac, 0.22, 1.38)

    # Environmental heat and degradation profiles.
    ambient_c = 31 + 6.5 * evening_peak + scenario["heat"] * 2.8 + rng.normal(0, noise * 0.45, n)
    insulation_aging = np.clip(degradation * 0.0017 * t, 0, 0.92)
    oil_degradation = np.clip(degradation * 0.0014 * t, 0, 0.88)
    bushing_degradation = np.clip(degradation * 0.0012 * t, 0, 0.82)

    # Partial maintenance reset to keep charts dynamic.
    for k in range(145, n, 165):
        insulation_aging[k:] = np.clip(insulation_aging[k:] - 0.16, 0, 1)
        oil_degradation[k:] = np.clip(oil_degradation[k:] - 0.12, 0, 1)
        bushing_degradation[k:] = np.clip(bushing_degradation[k:] - 0.10, 0, 1)

    kva = asset.capacity_kva * load_frac
    phase_imbalance = np.clip(1.8 + 7.5 * bushing_degradation + 3.0 * scenario["fault"] * np.maximum(load_frac - 0.82, 0) + rng.normal(0, noise * 0.45, n), 0.5, 18)
    thd_pct = np.clip(2.5 + 3.0 * scenario["stress"] * np.maximum(load_frac - 0.7, 0) + 4.5 * bushing_degradation + rng.normal(0, noise * 0.35, n), 1, 16)
    pf = np.clip(0.96 - 0.055 * np.maximum(load_frac - 0.75, 0) - 0.015 * thd_pct / 10 + rng.normal(0, noise * 0.006, n), 0.78, 0.99)

    oil_temp = 45 + 27 * (load_frac ** 1.75) + 0.48 * (ambient_c - 32) + 13 * oil_degradation + rng.normal(0, noise * 0.8, n)
    winding_temp = oil_temp + 13 + 28 * np.maximum(load_frac - 0.72, 0) + 16 * insulation_aging + rng.normal(0, noise * 1.1, n)
    oil_level_pct = np.clip(96 - 10.5 * oil_degradation - 2.5 * scenario["heat"] * evening_peak + rng.normal(0, noise * 0.55, n), 78, 101)
    dissolved_gas_ppm = np.clip(80 + 260 * insulation_aging + 110 * np.maximum(winding_temp - 95, 0) / 35 + rng.normal(0, noise * 8, n), 35, 720)

    # Feeder and LT metrics.
    voltage_dev_pct = np.clip(1.2 + 5.7 * np.maximum(load_frac - 0.82, 0) + 2.8 * scenario["fault"] * bushing_degradation + rng.normal(0, noise * 0.35, n), 0.2, 14)
    current_a = np.clip((kva * 1000) / (math.sqrt(3) * 11000 * np.maximum(pf, 0.8)) + rng.normal(0, noise * 2.5, n), 12, 95)
    neutral_current_a = np.clip(8 + 34 * phase_imbalance / 10 + 17 * scenario["fault"] * bushing_degradation + rng.normal(0, noise * 2.2, n), 2, 95)
    breaker_trip_prob = np.clip(2 + 34 * np.maximum(load_frac - 1.0, 0) + 1.4 * voltage_dev_pct + 0.25 * neutral_current_a + rng.normal(0, noise * 1.5, n), 0, 98)

    # Energy-in vs billing/out signature for losses.
    input_mwh = np.cumsum(np.clip(kva / 1000 * 0.25, 0.03, 0.50))
    expected_loss = asset.baseline_loss_pct + 2.2 * np.maximum(load_frac - 0.9, 0)
    tamper_burst = scenario["tamper"] * (2.5 * np.exp(-((t - 205) / 45) ** 2) + 2.0 * np.exp(-((t - 315) / 55) ** 2))
    feeder_loss_pct = np.clip(expected_loss + tamper_burst + 1.2 * bushing_degradation + rng.normal(0, noise * 0.35, n), 3.5, 26)
    billed_mwh = input_mwh * (1 - feeder_loss_pct / 100)

    # Smart meter telemetry health.
    meter_online_pct = np.clip(98.5 - telemetry_drop - 1.8 * scenario["fault"] * np.maximum(load_frac - 0.9, 0) + rng.normal(0, noise * 0.2, n), 88, 100)

    return {
        "t": t,
        "load_frac": load_frac,
        "kva": kva,
        "loading_pct": load_frac * 100,
        "ambient_c": ambient_c,
        "oil_temp": oil_temp,
        "winding_temp": winding_temp,
        "oil_level_pct": oil_level_pct,
        "dissolved_gas_ppm": dissolved_gas_ppm,
        "phase_imbalance": phase_imbalance,
        "thd_pct": thd_pct,
        "pf": pf,
        "voltage_dev_pct": voltage_dev_pct,
        "current_a": current_a,
        "neutral_current_a": neutral_current_a,
        "breaker_trip_prob": breaker_trip_prob,
        "input_mwh": input_mwh,
        "billed_mwh": billed_mwh,
        "feeder_loss_pct": feeder_loss_pct,
        "meter_online_pct": meter_online_pct,
        "insulation_aging": insulation_aging,
        "oil_degradation": oil_degradation,
        "bushing_degradation": bushing_degradation,
    }


def severity_label(score: float) -> str:
    if score >= 75:
        return "ALERT"
    if score >= 45:
        return "WATCH"
    return "NORMAL"


def compute_agent(asset: Asset, x: dict, cursor: int, design_load: float, oil_limit: float, winding_limit: float, voltage_band: float, loss_baseline: float):
    loading = float(x["loading_pct"][cursor])
    oil = float(x["oil_temp"][cursor])
    winding = float(x["winding_temp"][cursor])
    oil_level = float(x["oil_level_pct"][cursor])
    dga = float(x["dissolved_gas_ppm"][cursor])
    imbalance = float(x["phase_imbalance"][cursor])
    thd = float(x["thd_pct"][cursor])
    pf = float(x["pf"][cursor])
    vdev = float(x["voltage_dev_pct"][cursor])
    neutral = float(x["neutral_current_a"][cursor])
    loss = float(x["feeder_loss_pct"][cursor])
    trip = float(x["breaker_trip_prob"][cursor])
    online = float(x["meter_online_pct"][cursor])

    overload_norm = np.clip((loading - design_load) / 35, 0, 1)
    oil_norm = np.clip((oil - oil_limit) / 24, 0, 1)
    winding_norm = np.clip((winding - winding_limit) / 35, 0, 1)
    oil_level_norm = np.clip((92 - oil_level) / 12, 0, 1)
    dga_norm = np.clip((dga - 220) / 320, 0, 1)
    imbalance_norm = np.clip((imbalance - 4.0) / 10, 0, 1)
    thd_norm = np.clip((thd - 5.0) / 8, 0, 1)
    pf_norm = np.clip((0.93 - pf) / 0.11, 0, 1)
    vdev_norm = np.clip((vdev - voltage_band) / 7, 0, 1)
    neutral_norm = np.clip((neutral - 35) / 45, 0, 1)
    loss_norm = np.clip((loss - loss_baseline) / 9, 0, 1)
    trip_norm = np.clip(trip / 75, 0, 1)
    online_norm = np.clip((96 - online) / 8, 0, 1)

    anomaly_score = float(np.clip(100 * (
        0.18 * overload_norm + 0.13 * oil_norm + 0.16 * winding_norm +
        0.10 * vdev_norm + 0.09 * imbalance_norm + 0.08 * neutral_norm +
        0.08 * thd_norm + 0.06 * loss_norm + 0.06 * trip_norm + 0.06 * online_norm
    ), 0, 100))

    maintenance_score = float(np.clip(100 * (
        0.18 * overload_norm + 0.16 * winding_norm + 0.13 * oil_norm +
        0.15 * dga_norm + 0.11 * oil_level_norm + 0.10 * neutral_norm +
        0.08 * imbalance_norm + 0.05 * thd_norm + 0.04 * trip_norm
    ), 0, 100))

    efficiency_score = float(np.clip(100 * (
        0.32 * loss_norm + 0.18 * pf_norm + 0.16 * thd_norm + 0.14 * vdev_norm +
        0.10 * imbalance_norm + 0.10 * online_norm
    ), 0, 100))

    # RUL in days: sharper drop when thermal + DGA risk rises.
    composite = max(maintenance_score, 0.72 * anomaly_score + 0.28 * efficiency_score)
    rul_days = float(np.clip(90 * (1 - (composite / 100) ** 1.42), 2, 90))

    findings = []
    actions = []
    root = "Stable operation"

    if overload_norm > 0.45:
        findings.append(f"Transformer loading is {loading:.0f}% versus safe band {design_load:.0f}% — overload signature active.")
        actions.append("Shift peak load / check feeder reconfiguration to reduce transformer loading.")
        root = "Overload / thermal stress"
    if winding_norm > 0.45 or oil_norm > 0.45:
        findings.append(f"Thermal anomaly: oil {oil:.1f}°C, winding {winding:.1f}°C; heat rise is above asset baseline.")
        actions.append("Inspect cooling, oil circulation, ventilation and load balance; plan thermography.")
        root = "Thermal rise / insulation stress"
    if dga_norm > 0.45:
        findings.append(f"DGA proxy is {dga:.0f} ppm — internal heating / insulation ageing probability rising.")
        actions.append("Prioritize oil sampling/DGA validation and insulation health check.")
        root = "Insulation ageing / dissolved gas rise"
    if oil_level_norm > 0.45:
        findings.append(f"Oil level has dropped to {oil_level:.1f}% — possible leak, evaporation or sensor drift.")
        actions.append("Verify oil level sensor and inspect transformer tank/gasket for leakage.")
    if vdev_norm > 0.45:
        findings.append(f"Voltage deviation is {vdev:.1f}% — LT side voltage quality anomaly detected.")
        actions.append("Check tap position, feeder loading, capacitor banks and local voltage compensation.")
        root = "Voltage sag / feeder stress"
    if imbalance_norm > 0.45 or neutral_norm > 0.45:
        findings.append(f"Phase imbalance {imbalance:.1f}% with neutral current {neutral:.0f}A — unbalanced LT load signature.")
        actions.append("Run phase balancing diagnostics at DT/LT feeder level.")
        root = "Unbalanced load / neutral stress"
    if thd_norm > 0.45:
        findings.append(f"THD is {thd:.1f}% and PF is {pf:.2f} — harmonic / reactive power inefficiency signature.")
        actions.append("Check nonlinear loads, capacitor health and harmonic filter requirement.")
    if loss_norm > 0.45:
        findings.append(f"Feeder loss is {loss:.1f}% versus baseline {loss_baseline:.1f}% — commercial/technical loss anomaly.")
        actions.append("Compare feeder input energy with meter clusters; trigger theft/tamper field verification.")
        root = "AT&C loss / tamper cluster"
    if trip_norm > 0.55:
        findings.append(f"Breaker trip probability is {trip:.0f}% — cable/feeder fault risk is escalating.")
        actions.append("Dispatch cable fault localization and preventive patrolling before outage escalates.")
        root = "Cable fault / breaker trip risk"
    if online_norm > 0.55:
        findings.append(f"Meter telemetry online rate has dropped to {online:.1f}% — data quality gap affects diagnostics confidence.")
        actions.append("Check modem/SIM/DCU health for smart-meter telemetry restoration.")

    if not findings:
        findings.append("No high-severity anomaly is active; metrics are inside configured design bands.")
    if not actions:
        actions.append("Continue normal monitoring; keep transformer in watchlist during evening peak.")

    return {
        "anomaly_score": anomaly_score,
        "maintenance_score": maintenance_score,
        "efficiency_score": efficiency_score,
        "rul_days": rul_days,
        "status_anomaly": severity_label(anomaly_score),
        "status_maint": severity_label(maintenance_score),
        "status_eff": severity_label(efficiency_score),
        "root": root,
        "findings": findings,
        "actions": list(dict.fromkeys(actions))[:6],
    }


def make_network_figure(asset: Asset, agent: dict):
    # Relative schematic coordinates; lat/lon retained in asset model for future map integration.
    nodes = pd.DataFrame(
        [
            ["Grid Infeed", 0, 2.5, "220/66 kV"],
            [asset.substation, 2, 2.5, "33/11 kV Substation"],
            [asset.transformer, 4, 2.5, "Distribution Transformer"],
            [asset.feeder, 6, 2.5, "11 kV Feeder"],
            ["LT Circuit A", 8, 3.35, f"{asset.consumers//3:,} consumers"],
            ["LT Circuit B", 8, 2.5, f"{asset.consumers//3:,} consumers"],
            ["LT Circuit C", 8, 1.65, f"{asset.consumers - 2*(asset.consumers//3):,} consumers"],
        ],
        columns=["name", "x", "y", "meta"],
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6)]
    fig = go.Figure()
    for i, j in edges:
        width = 5 if (i, j) in [(1, 2), (2, 3)] else 3
        fig.add_trace(go.Scatter(
            x=[nodes.loc[i, "x"], nodes.loc[j, "x"]],
            y=[nodes.loc[i, "y"], nodes.loc[j, "y"]],
            mode="lines",
            line=dict(width=width),
            hoverinfo="skip",
            showlegend=False,
        ))
    size = [24, 28, 34, 30, 24, 24, 24]
    fig.add_trace(go.Scatter(
        x=nodes["x"], y=nodes["y"], mode="markers+text",
        marker=dict(size=size, line=dict(width=2)),
        text=nodes["name"], textposition="bottom center",
        customdata=nodes["meta"],
        hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
        showlegend=False,
    ))
    fig.add_annotation(
        x=4, y=3.35,
        text=f"AI status: {agent['status_anomaly']} · Root: {agent['root']}",
        showarrow=False,
        borderpad=5,
        bgcolor="rgba(255,255,255,0.65)",
    )
    fig.update_layout(
        height=360,
        margin=dict(l=5, r=5, t=15, b=5),
        xaxis=dict(visible=False, range=[-0.5, 8.6]),
        yaxis=dict(visible=False, range=[1.0, 3.8]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def genbi_answer(q: str, asset: Asset, x: dict, cursor: int, agent: dict, due_str: str):
    ql = q.strip().lower()
    if not ql:
        return None, None

    def metric_value(key, label, unit=""):
        value = float(x[key][cursor])
        suffix = f" {unit}" if unit else ""
        return f"Current **{label}** is **{value:.2f}{suffix}**."

    if any(k in ql for k in ["current risk", "maintenance risk", "rul", "remaining"]):
        return (
            f"Current maintenance risk is **{agent['maintenance_score']:.0f}/100** (**{agent['status_maint']}**). "
            f"Predicted RUL is **{agent['rul_days']:.0f} days** and next service should be planned by **{due_str}**.",
            None,
        )
    if "anomaly" in ql or "alert" in ql:
        return f"AI agent anomaly score is **{agent['anomaly_score']:.0f}/100** (**{agent['status_anomaly']}**). Primary root cause: **{agent['root']}**.", None
    if "efficiency" in ql or "loss" in ql or "at&c" in ql or "atc" in ql:
        return (
            f"Efficiency leakage score is **{agent['efficiency_score']:.0f}/100** (**{agent['status_eff']}**). "
            f"Feeder loss is **{x['feeder_loss_pct'][cursor]:.1f}%** versus configured baseline **{loss_baseline:.1f}%**.",
            None,
        )
    if "root" in ql or "cause" in ql or "why" in ql:
        bullets = "\n".join(f"- {f}" for f in agent["findings"][:6])
        return f"Primary root cause: **{agent['root']}**.\n\n{bullets}", None
    if "action" in ql or "recommend" in ql or "dispatch" in ql or "diagnostic" in ql:
        bullets = "\n".join(f"- {a}" for a in agent["actions"][:6])
        return f"Recommended diagnostic / field actions:\n{bullets}", None
    if "consumer" in ql or "dt" in ql or "feeder" in ql or "asset" in ql:
        return (
            f"Asset scope: **{asset.zone}**, **{asset.substation}**, **{asset.transformer}**, feeder **{asset.feeder}**, "
            f"**{asset.dt_count} DT/LT points** and about **{asset.consumers:,} consumers** in this simulated pocket.",
            None,
        )

    metric_map = {
        "loading": ("loading_pct", "Transformer Loading", "%"),
        "load": ("loading_pct", "Transformer Loading", "%"),
        "oil": ("oil_temp", "Oil Temperature", "°C"),
        "winding": ("winding_temp", "Winding Temperature", "°C"),
        "dga": ("dissolved_gas_ppm", "DGA Proxy", "ppm"),
        "gas": ("dissolved_gas_ppm", "DGA Proxy", "ppm"),
        "voltage": ("voltage_dev_pct", "Voltage Deviation", "%"),
        "imbalance": ("phase_imbalance", "Phase Imbalance", "%"),
        "neutral": ("neutral_current_a", "Neutral Current", "A"),
        "thd": ("thd_pct", "THD", "%"),
        "pf": ("pf", "Power Factor", ""),
        "power factor": ("pf", "Power Factor", ""),
        "trip": ("breaker_trip_prob", "Breaker Trip Probability", "%"),
        "meter": ("meter_online_pct", "Smart Meter Online Rate", "%"),
    }

    m = re.search(r"last\s+(\d+)\s+(ticks|points|minutes)", ql)
    n = int(m.group(1)) if m else 90
    n = int(np.clip(n, 20, 240))
    start = max(0, cursor - n)

    chosen = None
    for token, descriptor in metric_map.items():
        if token in ql:
            chosen = descriptor
            break

    if chosen:
        key, label, unit = chosen
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x["t"][start:cursor + 1], y=x[key][start:cursor + 1], mode="lines", name=label))
        fig.add_vline(x=x["t"][cursor], line_width=2)
        fig.update_layout(height=285, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Telemetry tick", yaxis_title=unit or label)
        return f"Showing last **{cursor - start}** telemetry ticks for **{label}**. {metric_value(key, label, unit)}", fig

    return (
        "Try questions like: **current risk**, **why is anomaly high**, **show last 80 ticks oil trend**, "
        "**feeder loss efficiency**, **recommended diagnostics**, or **asset scope**.",
        None,
    )


# ============================================================
# Session state and telemetry cursor
# ============================================================
series = make_distribution_series(selected_asset, scenario, noise=sensor_noise, degradation=degradation, dropout=telemetry_drop)

if "cursor" not in st.session_state:
    st.session_state.cursor = 0
if "last_selection" not in st.session_state:
    st.session_state.last_selection = None

selection_key = (selected_asset_label, scenario_name)
if st.session_state.last_selection != selection_key:
    st.session_state.cursor = 0
    st.session_state.last_selection = selection_key

cursor = int(np.clip(st.session_state.cursor, 0, len(series["t"]) - 1))
st.session_state.cursor = cursor

agent = compute_agent(
    selected_asset,
    series,
    cursor,
    design_load=design_load,
    oil_limit=oil_temp_limit,
    winding_limit=winding_temp_limit,
    voltage_band=voltage_band,
    loss_baseline=loss_baseline,
)

service_due = datetime.now() + timedelta(days=agent["rul_days"])
service_due_str = service_due.strftime("%d %b %Y")
confidence = float(np.clip(91 - sensor_noise * 4 - telemetry_drop * 1.3 + min(8, agent["anomaly_score"] / 12), 58, 96))

if autoplay and st.session_state.cursor < len(series["t"]) - 1:
    st.session_state.cursor = min(st.session_state.cursor + 2, len(series["t"]) - 1)
    time.sleep(tick_ms / 1000.0)
    st.rerun()
elif autoplay:
    st.caption("Autoplay reached the latest telemetry tick. Use **Advance +12 ticks** or change scenario/asset to continue.")


# ============================================================
# Main layout
# ============================================================
left, right = st.columns([1.18, 1.0], gap="large")

with left:
    st.subheader("🗺️ Distribution Network View")
    st.plotly_chart(make_network_figure(selected_asset, agent), use_container_width=True)

    st.markdown("<div class='tight-card'>", unsafe_allow_html=True)
    st.markdown("### 📌 Executive Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Agent State", agent["status_anomaly"])
    c2.metric("Anomaly Score", f"{agent['anomaly_score']:.0f}/100")
    c3.metric("Maint. Risk", f"{agent['maintenance_score']:.0f}/100")
    c4.metric("RUL", f"{agent['rul_days']:.0f} days")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Efficiency Leakage", f"{agent['efficiency_score']:.0f}/100")
    c6.metric("Loading", f"{series['loading_pct'][cursor]:.0f}%")
    c7.metric("Loss", f"{series['feeder_loss_pct'][cursor]:.1f}%")
    c8.metric("Trip Prob.", f"{series['breaker_trip_prob'][cursor]:.0f}%")

    st.markdown(
        f"<span class='pill {'redpill' if agent['status_anomaly']=='ALERT' else 'amberpill' if agent['status_anomaly']=='WATCH' else 'greenpill'}'>Root: {agent['root']}</span>"
        f"<span class='pill'>Next service by: {service_due_str}</span>"
        f"<span class='pill'>Confidence: {confidence:.0f}%</span>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🧠 AI Agent Findings")
    for finding in agent["findings"][:5]:
        st.write(f"- {finding}")

    st.markdown("#### 🔧 Recommended Actions")
    for action in agent["actions"][:5]:
        st.write(f"- {action}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 💬 GenBI Quick Query")
    q_quick = st.text_input(
        "Ask the distribution GenBI assistant",
        placeholder="e.g., why is anomaly high? show last 80 ticks oil trend; feeder loss efficiency",
        key="quick_genbi",
    )
    if q_quick:
        answer, fig = genbi_answer(q_quick, selected_asset, series, cursor, agent, service_due_str)
        if answer:
            st.info(answer)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📟 Live Distribution Telemetry")

    r1, r2, r3 = st.columns(3)
    r1.metric("Transformer kVA", f"{series['kva'][cursor]:.0f}")
    r2.metric("Oil Temp", f"{series['oil_temp'][cursor]:.1f}°C")
    r3.metric("Winding Temp", f"{series['winding_temp'][cursor]:.1f}°C")

    r4, r5, r6 = st.columns(3)
    r4.metric("Voltage Dev.", f"{series['voltage_dev_pct'][cursor]:.1f}%")
    r5.metric("Phase Imbalance", f"{series['phase_imbalance'][cursor]:.1f}%")
    r6.metric("Neutral Current", f"{series['neutral_current_a'][cursor]:.0f}A")

    r7, r8, r9 = st.columns(3)
    r7.metric("THD", f"{series['thd_pct'][cursor]:.1f}%")
    r8.metric("Power Factor", f"{series['pf'][cursor]:.2f}")
    r9.metric("Meter Online", f"{series['meter_online_pct'][cursor]:.1f}%")

    tabs = st.tabs(["📈 Telemetry", "🧠 Agent Diagnostics", "📊 GenBI Workspace"])

    with tabs[0]:
        window = 150
        start = max(0, cursor - window)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series["t"][start:cursor + 1], y=series["loading_pct"][start:cursor + 1], mode="lines", name="Loading %"))
        fig.add_trace(go.Scatter(x=series["t"][start:cursor + 1], y=series["oil_temp"][start:cursor + 1], mode="lines", name="Oil °C", yaxis="y2"))
        fig.add_trace(go.Scatter(x=series["t"][start:cursor + 1], y=series["feeder_loss_pct"][start:cursor + 1], mode="lines", name="Loss %", yaxis="y3"))
        fig.add_vline(x=series["t"][cursor], line_width=2)
        fig.add_hline(y=design_load, line_width=1)
        fig.update_layout(
            height=388,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Telemetry tick",
            yaxis=dict(title="Loading %"),
            yaxis2=dict(title="Oil °C", overlaying="y", side="right"),
            yaxis3=dict(title="Loss %", overlaying="y", side="right", position=0.97, showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        cA, cB = st.columns([1, 2])
        with cA:
            if st.button("⏩ Advance telemetry"):
                st.session_state.cursor = min(st.session_state.cursor + 12, len(series["t"]) - 1)
                st.rerun()
        with cB:
            st.progress(int((cursor / (len(series["t"]) - 1)) * 100))

    with tabs[1]:
        c1, c2 = st.columns([0.9, 1.1])
        with c1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=agent["rul_days"],
                number={"suffix": " days"},
                gauge={"axis": {"range": [0, 90]}, "bar": {"thickness": 0.35}},
                title={"text": "Remaining Useful Life"},
            ))
            gauge.update_layout(height=285, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        with c2:
            s = max(0, cursor - 150)
            anomaly, maint, eff = [], [], []
            for i in range(s, cursor + 1):
                a = compute_agent(selected_asset, series, i, design_load, oil_temp_limit, winding_temp_limit, voltage_band, loss_baseline)
                anomaly.append(a["anomaly_score"])
                maint.append(a["maintenance_score"])
                eff.append(a["efficiency_score"])
            risk_fig = go.Figure()
            risk_fig.add_trace(go.Scatter(x=series["t"][s:cursor + 1], y=anomaly, mode="lines", name="Anomaly"))
            risk_fig.add_trace(go.Scatter(x=series["t"][s:cursor + 1], y=maint, mode="lines", name="Maintenance"))
            risk_fig.add_trace(go.Scatter(x=series["t"][s:cursor + 1], y=eff, mode="lines", name="Efficiency"))
            risk_fig.add_hline(y=45, line_width=1)
            risk_fig.add_hline(y=75, line_width=1)
            risk_fig.add_vline(x=series["t"][cursor], line_width=2)
            risk_fig.update_layout(height=285, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Score 0-100", xaxis_title="Tick", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(risk_fig, use_container_width=True)

        st.markdown("### Diagnostic Work Order Preview")
        wo = pd.DataFrame(
            [
                ["Priority", agent["status_maint"]],
                ["Likely Root Cause", agent["root"]],
                ["Service Due By", service_due_str],
                ["Crew Skill", "Transformer + Feeder diagnostics" if agent["maintenance_score"] > 45 else "Routine inspection"],
                ["Suggested Checks", "; ".join(agent["actions"][:3])],
            ],
            columns=["Field", "Recommendation"],
        )
        st.dataframe(wo, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("### GenBI Workspace")
        st.caption("Plain-English, offline rule-based layer for demo. Can be connected to enterprise BI/LLM later.")
        q = st.text_input("Your question", placeholder="e.g., current risk; show last 100 ticks voltage trend; recommended diagnostics", key="full_genbi")
        if q:
            ans, fig = genbi_answer(q, selected_asset, series, cursor, agent, service_due_str)
            if ans:
                st.info(ans)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Exportable KPI table")
        kpi_df = pd.DataFrame(
            {
                "Metric": [
                    "Transformer Loading %", "Oil Temp °C", "Winding Temp °C", "DGA Proxy ppm", "Voltage Deviation %",
                    "Phase Imbalance %", "THD %", "Power Factor", "Feeder Loss %", "Meter Online %",
                    "Anomaly Score", "Maintenance Score", "Efficiency Leakage Score", "RUL Days",
                ],
                "Current Value": [
                    f"{series['loading_pct'][cursor]:.1f}", f"{series['oil_temp'][cursor]:.1f}", f"{series['winding_temp'][cursor]:.1f}",
                    f"{series['dissolved_gas_ppm'][cursor]:.0f}", f"{series['voltage_dev_pct'][cursor]:.1f}", f"{series['phase_imbalance'][cursor]:.1f}",
                    f"{series['thd_pct'][cursor]:.1f}", f"{series['pf'][cursor]:.2f}", f"{series['feeder_loss_pct'][cursor]:.1f}",
                    f"{series['meter_online_pct'][cursor]:.1f}", f"{agent['anomaly_score']:.0f}", f"{agent['maintenance_score']:.0f}",
                    f"{agent['efficiency_score']:.0f}", f"{agent['rul_days']:.0f}",
                ],
                "Interpretation": [
                    "Load stress on DT", "Oil thermal condition", "Insulation thermal stress", "Internal fault/ageing proxy", "Voltage quality",
                    "Unbalanced LT loading", "Harmonic distortion", "Reactive power efficiency", "Technical/commercial loss", "Telemetry reliability",
                    "AI agent anomaly index", "Predictive maintenance urgency", "Loss/quality/efficiency leakage", "Predicted service horizon",
                ],
            }
        )
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)
