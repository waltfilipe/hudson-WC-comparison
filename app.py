import re
import os
from pathlib import Path
from io import BytesIO

import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
import mplcursors

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Cicala — Pass Comparison")

# OPTIONAL DOCX IMPORT
DOCX_AVAILABLE = True
try:
    from docx import Document
except Exception:
    DOCX_AVAILABLE = False

# STYLE
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .player-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #eef1f7;
        margin-bottom: 0.15rem;
    }
    .player-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    .map-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #c7cdda;
        margin: 0.6rem 0 0.25rem 0;
    }
  </style>
    """,
    unsafe_allow_html=True,
)

# CONSTANTS
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
GOAL_X = 120.0
GOAL_Y = 40.0
FIG_W, FIG_H = 7.0, 4.7
FIG_DPI = 180
COLOR_SUCCESS = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.07
PASS_TONES = ["#5b9bd5", "#3b82f6", "#1d4ed8"]
PLAYER_TONES = {
    "Hudson Cicala": "#5b9bd5",
    "MacAllister": "#70ad47",
    "Bouaddi": "#d4a843",
}
CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)
NX_XT, NY_XT = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60
LATERAL_MIN_DIST = 12.0

HUDSON_DOCX = "Passes - Hudson Cicala.docx"
WORLD_CUP_DOCX = "Passes World Cup.docx"
MACALLISTER_KEY = "MacAllister (vs Argelia)"
BOUADDI_KEY = "Bouaddi (vs Brasil)"
WORLD_CUP_MINUTES = 90.0

SGA_RANGE_METRICS = {
    "xt_p90": "1.7 – 2.0",
    "pos_pct": "40% – 45%",
}
SGA_RANGE_LABEL = "SGA Range"
BENCHMARK_EUR_KEY = "TOP 5 - EUR"
BENCHMARK_POSITIONS = ("RDMF", "RCMF", "LDMF", "LCMF", "DMF")
BENCHMARK_FILES = {"MLS": "MLS 1.xlsx", BENCHMARK_EUR_KEY: "TOP 5 - UE.xlsx"}
BENCHMARK_MINUTES_RATIO = 0.50

CARD_TITLE_TEXT = "14px"
CARD_LABEL_TEXT = "16px"
CARD_SUBTEXT = "13px"
CARD_INNER_BORDER = "rgba(107,114,128,0.45)"
CARD_MUTED_TEXT = "#94a3b8"

C_GREEN_LIGHT = "#86efac"
C_GREEN_STRONG = "#15803d"
C_ORANGE_LIGHT = "#fdba74"
C_RED_DARK = "#7f1d1d"
C_NEUTRAL = "#94a3b8"


def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def is_progressive_pass(x_start, y_start, x_end, y_end):
    if x_start < 35:
        return False
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    if start_dist == 0:
        return False
    return ((start_dist - end_dist) / start_dist) >= 0.25


def classify_pass_direction(x_start, y_start, x_end, y_end):
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:
        return "forward"
    if angle_deg >= 135.0:
        return "backward"
    if dist > LATERAL_MIN_DIST:
        return "lateral_right" if dy > 0 else "lateral_left"
    return "forward" if dx >= 0 else "backward"


def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))


@st.cache_data(show_spinner=False)
def compute_xt_grid(NX=16, NY=12, sub=24):
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    xp = 0.01 + (Xc / FIELD_X) * 0.99
    yc = 1.0 - np.abs((Yc / FIELD_Y) - 0.5) * 2.0
    base = xp * (0.8 + 0.2 * yc)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12)
    XT = base.copy()
    XT = (XT - XT.min()) / (XT.max() - XT.min() + 1e-12)
    XTc = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            XTc[iy, ix] = XT[iy * sub:(iy + 1) * sub, ix * sub:(ix + 1) * sub].mean()
    XTc = (XTc - XTc.min()) / (XTc.max() - XTc.min() + 1e-12)
    return XTc


XT_GRID = compute_xt_grid()


def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * NX_XT, 0, NX_XT - 1))
    iy = int(np.clip((y / FIELD_Y) * NY_XT, 0, NY_XT - 1))
    return float(XT_GRID[iy, ix])


def apply_date_mapping(name: str) -> str:
    mapping = {
        "Real Salt Lake": "Real Salt Lake (04-26)",
        "Real Futbol": "Real Futbol (05-23)",
        "San Jose": "San Jose (05-24)",
        "Houston Dynamo": "Houston Dynamo (05-26)",
    }
    for k, v in mapping.items():
        if k.lower() == name.lower().strip():
            return v
    return name


def get_match_minutes(match_name: str, hudson_matches: list[str] | None = None) -> float:
    if match_name in (MACALLISTER_KEY, BOUADDI_KEY):
        return WORLD_CUP_MINUTES
    name_lower = match_name.lower()
    if "houston" in name_lower:
        return 63.0
    if "vardar" in name_lower:
        return 65.0
    return 90.0


def read_docx_text(docx_path: Path) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed.")
    doc = Document(str(docx_path))
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())


def parse_hudson_docx(raw_text: str) -> dict:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    matches = {}
    current_match = None
    current_state = None
    re_match = re.compile(r"^Vs\s+(.+)$", re.IGNORECASE)
    re_success = re.compile(r"^Sucesso$", re.IGNORECASE)
    re_fail = re.compile(r"^Errado[s]?$", re.IGNORECASE)
    re_arrow = re.compile(
        r"^Seta\s+\d+:\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\s*->\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)$",
        re.IGNORECASE,
    )
    for ln in lines:
        m_match = re_match.match(ln)
        if m_match:
            current_match = apply_date_mapping(m_match.group(1).strip())
            matches.setdefault(current_match, [])
            current_state = None
            continue
        if re_success.match(ln):
            current_state = "PASS WON"
            continue
        if re_fail.match(ln):
            current_state = "PASS LOST"
            continue
        m_arrow = re_arrow.match(ln)
        if m_arrow and current_match and current_state:
            x1, y1, x2, y2 = map(float, m_arrow.groups())
            matches[current_match].append((current_state, x1, y1, x2, y2, None))
    return {k: v for k, v in matches.items() if len(v) > 0}


def parse_world_cup_docx(raw_text: str) -> dict:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    players = {}
    current_player = None
    current_state = "PASS WON"
    re_player = re.compile(r"^Passes\s+Totais\s*\|\s*(.+)$", re.IGNORECASE)
    re_fail_section = re.compile(r"^Passes\s+Errados:?$", re.IGNORECASE)
    re_arrow = re.compile(
        r"^Seta\s+\d+:\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\s*->\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)$",
        re.IGNORECASE,
    )
    for ln in lines:
        m_player = re_player.match(ln)
        if m_player:
            current_player = m_player.group(1).strip()
            players.setdefault(current_player, [])
            current_state = "PASS WON"
            continue
        if re_fail_section.match(ln):
            current_state = "PASS LOST"
            continue
        m_arrow = re_arrow.match(ln)
        if m_arrow and current_player:
            x1, y1, x2, y2 = map(float, m_arrow.groups())
            players[current_player].append((current_state, x1, y1, x2, y2, None))
    return {k: v for k, v in players.items() if len(v) > 0}


def events_to_dataframe(events: list, match_name: str) -> pd.DataFrame:
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False)
    dfm["progressive"] = dfm.apply(
        lambda r: r["is_won"] and is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1,
    )
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1,
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"].isin(["lateral_left", "lateral_right"])
    dfm["pass_distance"] = np.sqrt((dfm["x_end"] - dfm["x_start"]) ** 2 + (dfm["y_end"] - dfm["y_start"]) ** 2)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(dfm["is_won"], dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["dist_bonus"] = distance_bonus(dfm["pass_distance"].values)
    dfm["delta_xt_adj"] = np.where(dfm["is_won"], dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]), 0.0)
    return dfm


@st.cache_data(show_spinner=False)
def load_all_pass_data() -> tuple[dict, dict]:
    hudson_path = Path(HUDSON_DOCX)
    wc_path = Path(WORLD_CUP_DOCX)
    if not hudson_path.exists() or not wc_path.exists():
        return {}, {}
    hudson_raw = parse_hudson_docx(read_docx_text(hudson_path))
    wc_raw = parse_world_cup_docx(read_docx_text(wc_path))
    hudson_dfs = {name: events_to_dataframe(events, name) for name, events in hudson_raw.items()}
    wc_dfs = {name: events_to_dataframe(events, name) for name, events in wc_raw.items()}
    return hudson_dfs, wc_dfs


def compute_stats(df: pd.DataFrame, match_name: str) -> dict:
    total = len(df)
    mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    if total == 0:
        return {
            "total_passes": 0,
            "successful_passes": 0,
            "unsuccessful_passes": 0,
            "accuracy_pct": 0.0,
            "progressive_attempted": 0,
            "progressive_successful": 0,
            "progressive_accuracy_pct": 0.0,
            "to_final_third_total": 0,
            "to_final_third_success": 0,
            "to_final_third_accuracy_pct": 0.0,
            "fwd": 0,
            "fwd_pct": 0.0,
            "bwd": 0,
            "bwd_pct": 0.0,
            "lat": 0,
            "lat_pct": 0.0,
            "pos_count": 0,
            "pos_pct": 0.0,
            "high_xt_pct": 0.0,
            "sum_dxt": 0.0,
            "total_p90": 0.0,
            "prog_p90": 0.0,
            "f3_p90": 0.0,
            "xt_p90": 0.0,
            "neg_xt_p90": 0.0,
            "minutes": mins,
            "long_acc_pct": 0.0,
            "high_xt_p90": 0.0,
            "dz_p90": 0.0,
            "advanced_passes_p90": 0.0,
            "advanced_accuracy_pct": 0.0,
        }
    successful = int(df["is_won"].sum())
    unsuccessful = total - successful
    accuracy = successful / total * 100.0
    progressive_total = int(df["progressive"].sum())
    progressive_unsuccessful = int(
        (
            ~df["is_won"]
            & df.apply(
                lambda r: is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
                axis=1,
            )
        ).sum()
    )
    progressive_attempted = progressive_total + progressive_unsuccessful
    progressive_accuracy = (progressive_total / progressive_attempted * 100.0) if progressive_attempted else 0.0
    to_final_third = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
    to_final_third_total = int(to_final_third.sum())
    to_final_third_success = int((to_final_third & df["is_won"]).sum())
    to_final_third_accuracy = (to_final_third_success / to_final_third_total * 100.0) if to_final_third_total else 0.0
    long_passes = df[df["pass_distance"] > 25.0]
    long_total = len(long_passes)
    long_success = int(long_passes["is_won"].sum())
    long_acc_pct = (long_success / long_total * 100.0) if long_total > 0 else 0.0
    dz_mask = df["is_won"] & (
        (df["x_end"] >= 100.0)
        | (
            (df["x_end"] >= 80.0)
            & (df["x_end"] < 100.0)
            & (df["y_end"] >= LANE_RIGHT_MAX)
            & (df["y_end"] < LANE_LEFT_MIN)
        )
    )
    dz_passes = int(dz_mask.sum())
    fwd = int(df["is_forward"].sum())
    bwd = int(df["is_backward"].sum())
    lat = int(df["is_lateral"].sum())
    pos_count = int((df["is_won"] & (df["delta_xt_adj"] > 0)).sum())
    pos_pct = (pos_count / total * 100.0) if total > 0 else 0.0
    high_xt = int((df["delta_xt_adj"] > 0.1).sum())
    sum_dxt = float(df.loc[df["is_won"], "delta_xt_adj"].sum())
    neg_xt = float(df.loc[df["is_won"] & (df["delta_xt_adj"] < 0), "delta_xt_adj"].sum())
    advanced_successful = progressive_total + to_final_third_success
    advanced_attempted = progressive_attempted + to_final_third_total
    advanced_accuracy_pct = (advanced_successful / advanced_attempted * 100.0) if advanced_attempted else 0.0
    advanced_passes_p90 = round((progressive_total + to_final_third_success) * p90_factor, 2)
    return {
        "total_passes": total,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "progressive_attempted": progressive_attempted,
        "progressive_successful": progressive_total,
        "progressive_accuracy_pct": round(progressive_accuracy, 2),
        "to_final_third_total": to_final_third_total,
        "to_final_third_success": to_final_third_success,
        "to_final_third_accuracy_pct": round(to_final_third_accuracy, 2),
        "fwd": fwd,
        "fwd_pct": round(fwd / total * 100.0, 1),
        "bwd": bwd,
        "bwd_pct": round(bwd / total * 100.0, 1),
        "lat": lat,
        "lat_pct": round(lat / total * 100.0, 1),
        "pos_count": pos_count,
        "pos_pct": round(pos_pct, 1),
        "high_xt_pct": round(high_xt / total * 100.0, 1),
        "sum_dxt": round(sum_dxt, 3),
        "total_p90": round(total * p90_factor, 1),
        "prog_p90": round(progressive_total * p90_factor, 2),
        "f3_p90": round(to_final_third_success * p90_factor, 2),
        "xt_p90": round(sum_dxt * p90_factor, 3),
        "neg_xt_p90": round(neg_xt * p90_factor, 3),
        "minutes": mins,
        "long_acc_pct": round(long_acc_pct, 1),
        "high_xt_p90": round(high_xt * p90_factor, 2),
        "dz_p90": round(dz_passes * p90_factor, 2),
        "advanced_passes_p90": round(advanced_passes_p90, 1),
        "advanced_accuracy_pct": round(advanced_accuracy_pct, 2),
    }


def _lerp_channel(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * max(0.0, min(1.0, t))))


def _lerp_hex(hex_a: str, hex_b: str, t: float) -> str:
    ha, hb = hex_a.lstrip("#"), hex_b.lstrip("#")
    r = _lerp_channel(int(ha[0:2], 16), int(hb[0:2], 16), t)
    g = _lerp_channel(int(ha[2:4], 16), int(hb[2:4], 16), t)
    b = _lerp_channel(int(ha[4:6], 16), int(hb[4:6], 16), t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _diff_gradient_color(diff_pct: float) -> tuple:
    if abs(diff_pct) < 0.5:
        return C_NEUTRAL, "rgba(148,163,184,0.15)"
    if diff_pct > 0:
        t = min(diff_pct / 15.0, 1.0)
        color = _lerp_hex(C_GREEN_LIGHT, C_GREEN_STRONG, t)
        return color, f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.18)"
    t = min(abs(diff_pct) / 15.0, 1.0)
    color = _lerp_hex(C_ORANGE_LIGHT, C_RED_DARK, t)
    return color, f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.18)"


def _target_pct_diff(val: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return ((val - target) / target) * 100.0


def _rand_target(base: float, key: str, is_pct: bool = False, decimals: int = 1) -> float:
    rng = np.random.default_rng(2026 + (hash(key) % 10000))
    sign = int(rng.choice([-1, 1]))
    if is_pct:
        return float(np.clip(base + sign * rng.uniform(2.0, 8.5), 0.0, 100.0))
    if base == 0:
        return round(rng.uniform(0.5, 2.5), decimals)
    target = max(0.0, base + sign * base * rng.uniform(0.06, 0.16))
    return round(target, decimals)


@st.cache_data(show_spinner=False)
def load_benchmark_targets(source: str) -> dict | None:
    filename = BENCHMARK_FILES.get(source)
    if not filename:
        return None
    path = Path(filename)
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path)
    except ImportError:
        return None
    if "Position" not in df.columns or "Minutes played" not in df.columns:
        return None
    df = df[df["Position"].isin(BENCHMARK_POSITIONS)].copy()
    max_mins = float(df["Minutes played"].max()) if len(df) else 0.0
    if max_mins <= 0:
        return None
    min_mins = max_mins * BENCHMARK_MINUTES_RATIO
    df = df[df["Minutes played"] >= min_mins]
    if df.empty:
        return None
    prog_p90 = df["Progressive passes per 90"].fillna(0)
    f3_p90 = df["Passes to final third per 90"].fillna(0)
    prog_acc = df["Accurate progressive passes, %"].fillna(0)
    f3_acc = df["Accurate passes to final third, %"].fillna(0)
    prog_attempted_p90 = np.where(prog_acc > 0, prog_p90 / (prog_acc / 100.0), 0.0)
    f3_success_p90 = f3_p90 * (f3_acc / 100.0)
    advanced_passes_series = prog_p90 + f3_success_p90
    advanced_attempted_series = prog_attempted_p90 + f3_p90
    advanced_success_series = prog_p90 + f3_success_p90
    advanced_accuracy_series = np.where(
        advanced_attempted_series > 0,
        advanced_success_series / advanced_attempted_series * 100.0,
        0.0,
    )
    return {
        "total_p90": round(float(df["Passes per 90"].mean()), 1),
        "accuracy_pct": round(float(df["Accurate passes, %"].mean()), 1),
        "advanced_passes_p90": round(float(advanced_passes_series.mean()), 1),
        "advanced_accuracy_pct": round(float(advanced_accuracy_series.mean()), 1),
        "sample_size": int(len(df)),
        "minutes_threshold": round(min_mins, 0),
    }


def build_metric_targets(pass_base: dict, benchmark_source: str = "MLS") -> dict:
    bench = load_benchmark_targets(benchmark_source)
    return {
        "total_p90": bench["total_p90"] if bench else _rand_target(pass_base["total_p90"], "total_p90"),
        "accuracy_pct": bench["accuracy_pct"] if bench else _rand_target(pass_base["accuracy_pct"], "accuracy_pct", is_pct=True),
        "advanced_passes_p90": bench["advanced_passes_p90"] if bench else _rand_target(pass_base.get("advanced_passes_p90", 0), "advanced_passes_p90"),
        "advanced_accuracy_pct": bench["advanced_accuracy_pct"] if bench else _rand_target(pass_base.get("advanced_accuracy_pct", 0), "advanced_accuracy_pct", is_pct=True),
        "xt_p90": _rand_target(pass_base["xt_p90"], "xt_p90", decimals=2 if pass_base["xt_p90"] < 5 else 1),
        "pos_pct": _rand_target(pass_base["pos_pct"], "pos_pct", is_pct=True),
    }


def _fmt_target_value(key: str, targets: dict) -> str:
    v = targets[key]
    if key in ("accuracy_pct", "advanced_accuracy_pct", "pos_pct"):
        return f"{v:.1f}%"
    if key == "xt_p90":
        return f"{v:.1f}"
    return f"{v:.1f}"


def build_metric_item(label: str, val: float, disp_val: str, key: str, extra: str = ""):
    if key in SGA_RANGE_METRICS:
        return (label, float(val), disp_val, "sga", SGA_RANGE_METRICS[key], "", extra)
    return (
        label,
        float(val),
        disp_val,
        "league",
        _fmt_target_value(key, T_MLS),
        _fmt_target_value(key, T_TOP_EUR),
        extra,
    )


def _sga_range_html(range_str: str) -> str:
    return (
        f'<div style="font-size:{CARD_SUBTEXT};color:{CARD_MUTED_TEXT};margin-top:4px;">'
        f"{SGA_RANGE_LABEL}: {range_str}"
        f"</div>"
    )


def _targets_line_html(disp_mls: str, disp_top_eur: str) -> str:
    return (
        f'<div style="font-size:{CARD_SUBTEXT};color:{CARD_MUTED_TEXT};margin-top:4px;">'
        f"MLS: {disp_mls} · TOP 5 EUR: {disp_top_eur}"
        f"</div>"
    )


def _item_reference_line(item) -> str:
    if item[3] == "sga":
        return _sga_range_html(item[4])
    return _targets_line_html(item[4], item[5])


def _item_sep(idx, total):
    return "" if idx == total - 1 else f"margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid {CARD_INNER_BORDER};"


def _accent_rgb(border_color):
    h = border_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _combined_body_scoreboard(border_color, items):
    body = ""
    for idx, item in enumerate(items):
        label, disp_val = item[0], item[2]
        extra = item[6] if len(item) > 6 and item[6] else ""
        body += f'<div style="{_item_sep(idx, len(items))}">'
        body += (
            '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;">'
            f'<span style="font-size:{CARD_LABEL_TEXT};color:#c7cdda;font-weight:600;">{label}</span>'
            f'<span style="font-size:28px;color:#ffffff;font-weight:700;line-height:1;">{disp_val}</span>'
            "</div>"
        )
        body += _item_reference_line(item)
        if extra:
            body += f'<div style="font-size:{CARD_SUBTEXT};color:{CARD_MUTED_TEXT};margin-top:4px;">{extra}</div>'
        body += "</div>"
    return body


def _target_card_shell_html(title, border_color, body_html):
    r, g, b = _accent_rgb(border_color)
    grad = (
        f"linear-gradient(150deg, rgba({r},{g},{b},0.18) 0%, "
        f"rgba(24,24,38,0.55) 55%, rgba(16,16,26,0.82) 100%)"
    )
    html = (
        f'<div style="background:{grad};border:1px solid rgba({r},{g},{b},0.55);'
        f'border-radius:14px;padding:18px 20px 14px 20px;margin-bottom:12px;">'
    )
    html += (
        f'<div style="border-bottom:2.5px solid rgb({r},{g},{b});padding-bottom:8px;margin-bottom:12px;">'
        f'<span style="font-size:{CARD_TITLE_TEXT};color:#eef1f7;font-weight:700;letter-spacing:0.04em;">'
        f"{title.upper()}</span></div>"
    )
    html += body_html
    html += "</div>"
    return html


def target_section_card(title, border_color, items):
    inner = _combined_body_scoreboard(border_color, items)
    st.markdown(_target_card_shell_html(title, border_color, inner), unsafe_allow_html=True)


def _base_pitch(bg="#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg)
    fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#ffffff", lw=1.2, alpha=0.40, linestyle="--")
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.7, alpha=0.12, linestyle="--")
    return fig, ax, pitch


def _attack_arrow(fig, has_cbar=False):
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(
        FancyArrowPatch(
            (0.44 + ox, 0.045),
            (0.56 + ox, 0.045),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.6,
            color="#aaaaaa",
        )
    )
    fig.text(
        0.50 + ox,
        0.012,
        "Attacking Direction",
        ha="center",
        va="bottom",
        transform=fig.transFigure,
        fontsize=7.5,
        color="#aaaaaa",
    )


def _save_fig(fig):
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def draw_pass_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        is_lost = not row["is_won"]
        is_prog = bool(row["progressive"])
        if is_lost:
            color, alpha = COLOR_FAIL, 0.72
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS
        pitch.arrows(
            row["x_start"],
            row["y_start"],
            row["x_end"],
            row["y_end"],
            color=color,
            width=1.3,
            headwidth=2.0,
            headlength=2.0,
            ax=ax,
            zorder=3,
            alpha=alpha,
        )
        pitch.scatter(
            row["x_start"],
            row["y_start"],
            s=32,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=0.6,
            ax=ax,
            zorder=6,
            alpha=alpha,
        )
    leg = ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completed", alpha=0.65),
            Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
            Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incomplete", alpha=0.90),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.5,
        labelspacing=0.35,
        borderpad=0.4,
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_corridor_heatmap(df):
    df_s = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {
        "left": (LANE_LEFT_MIN, FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right": (0.0, LANE_RIGHT_MAX),
    }
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            arr[i] = int(
                ((df_s["x_end"] >= x0_) & (df_s["x_end"] < x1_) & (df_s["y_end"] >= y0) & (df_s["y_end"] < y1)).sum()
            )
        counts[cname] = arr
    all_vals = np.concatenate([counts[c] for c in counts])
    vmax = max(1, int(all_vals.max()))
    cmap = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)
    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            value = counts[cname][i]
            ax.add_patch(
                Rectangle(
                    (x0_, y0),
                    x1_ - x0_,
                    y1 - y0,
                    facecolor=cmap(norm(value)),
                    edgecolor=(1, 1, 1, 0.12),
                    lw=0.5,
                    alpha=0.95,
                    zorder=2,
                )
            )
            ax.text(
                (x0_ + x1_) / 2,
                (y0 + y1) / 2,
                str(value),
                ha="center",
                va="center",
                color="#000000" if value <= threshold else "#ffffff",
                fontsize=9,
                fontweight="700" if value >= vmax * 0.5 else "600",
                zorder=4,
            )
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def _draw_comet_arrow(ax, x0, y0, x1, y1, color):
    segs = 12
    ts = np.linspace(0.0, 1.0, segs + 1)
    for i in range(segs):
        t0, t1 = ts[i], ts[i + 1]
        xa = x0 + (x1 - x0) * t0
        ya = y0 + (y1 - y0) * t0
        xb = x0 + (x1 - x0) * t1
        yb = y0 + (y1 - y0) * t1
        alpha = 0.85 * (0.15 + 0.85 * t1)
        lw = 2.5 * (0.80 + 0.20 * t1)
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=4, solid_capstyle="round")
    ax.scatter(x0, y0, s=20, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, zorder=5, alpha=0.85)
    ax.scatter(x1, y1, s=32, marker="o", facecolors=color, edgecolors="white", linewidths=0.9, zorder=6, alpha=0.85)


def draw_top_xt_map(df, top_n=10):
    fig, ax, pitch = _base_pitch()
    top_passes = (
        df[(df["is_won"]) & (df["delta_xt_adj"] > 0)]
        .sort_values("delta_xt_adj", ascending=False)
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )
    cursor_points = []
    if not top_passes.empty:
        for _, row in top_passes.iterrows():
            val = float(row["delta_xt_adj"])
            color = CMAP_TOP10(NORM_TOP10(np.clip(val, 0.05, 0.40)))
            _draw_comet_arrow(
                ax,
                float(row["x_start"]),
                float(row["y_start"]),
                float(row["x_end"]),
                float(row["y_end"]),
                color,
            )
            match_name = row.get("match", "")
            pt = ax.scatter(
                float(row["x_start"]),
                float(row["y_start"]),
                s=20,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                zorder=5,
                alpha=0,
                visible=False,
            )
            cursor_points.append((pt, f"xT: {val:.3f}\nMatch: {match_name}"))
        crs = mplcursors.cursor([p[0] for p in cursor_points], hover=True)

        @crs.connect("add")
        def _(sel):
            sel.annotation.set_text(cursor_points[sel.index][1])
            sel.annotation.get_bbox_patch().set(fc="#1a1a2e", ec="#5b9bd5", alpha=0.95)
            sel.annotation.arrow_patch.set(connectionstyle="arc3,rad=0.2", fc="#1a1a2e", ec="#5b9bd5")

    sm = plt.cm.ScalarMappable(cmap=CMAP_TOP10, norm=NORM_TOP10)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
    cbar.set_label("Pass Impact", color="#ffffff", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig


def render_player_maps(df: pd.DataFrame):
    img_pm, fig_pm = draw_pass_map(df)
    plt.close(fig_pm)
    st.markdown('<div class="map-label">Pass Map</div>', unsafe_allow_html=True)
    st.image(img_pm, use_container_width=True)

    img_ht, fig_ht = draw_corridor_heatmap(df)
    plt.close(fig_ht)
    st.markdown('<div class="map-label">Zone Heatmap (Destination)</div>', unsafe_allow_html=True)
    st.image(img_ht, use_container_width=True)

    img_xt, fig_xt = draw_top_xt_map(df, top_n=10)
    plt.close(fig_xt)
    st.markdown('<div class="map-label">Top 10 Pass Impact</div>', unsafe_allow_html=True)
    st.image(img_xt, use_container_width=True)


def render_player_cards(stats: dict, tone: str):
    target_section_card(
        "Overview",
        tone,
        [
            build_metric_item("Total Passes Per Game", stats["total_p90"], f"{stats['total_p90']:.1f}", "total_p90"),
            build_metric_item("% Accuracy", stats["accuracy_pct"], f"{stats['accuracy_pct']:.1f}%", "accuracy_pct"),
        ],
    )
    target_section_card(
        "Progressive",
        tone,
        [
            build_metric_item(
                "Progressive Passes Per Game",
                stats["advanced_passes_p90"],
                f"{stats['advanced_passes_p90']:.1f}",
                "advanced_passes_p90",
            ),
            build_metric_item(
                "% Progressive Accuracy",
                stats["advanced_accuracy_pct"],
                f"{stats['advanced_accuracy_pct']:.1f}%",
                "advanced_accuracy_pct",
            ),
        ],
    )
    target_section_card(
        "Impact",
        tone,
        [
            build_metric_item("Pass Impact Value Per Game", stats["xt_p90"], f"{stats['xt_p90']:.1f}", "xt_p90"),
            build_metric_item("% Positive Impact", stats["pos_pct"], f"{stats['pos_pct']:.1f}%", "pos_pct"),
        ],
    )


# ── DATA LOAD ──────────────────────────────────────────────────
hudson_dfs, wc_dfs = load_all_pass_data()

if not hudson_dfs or MACALLISTER_KEY not in wc_dfs or BOUADDI_KEY not in wc_dfs:
    st.error(
        "Não foi possível carregar os dados. Verifique se os arquivos "
        f"'{HUDSON_DOCX}' e '{WORLD_CUP_DOCX}' estão no diretório do app."
    )
    st.stop()

hudson_match_names = list(hudson_dfs.keys())
macallister_df = wc_dfs[MACALLISTER_KEY]
bouaddi_df = wc_dfs[BOUADDI_KEY]

_pass_base = {}
all_hudson_stats = [compute_stats(hudson_dfs[m], m) for m in hudson_match_names]
if all_hudson_stats:
    for k in all_hudson_stats[0].keys():
        if isinstance(all_hudson_stats[0][k], (int, float)):
            _pass_base[k] = sum(s[k] for s in all_hudson_stats) / len(all_hudson_stats)

T_MLS = build_metric_targets(_pass_base, "MLS")
T_TOP_EUR = build_metric_targets(_pass_base, BENCHMARK_EUR_KEY)

# ── SIDEBAR (sem customização) ─────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center;">
      <h2 style="margin:0;color:#eef1f7;">Pass Comparison</h2>
      <p style="color:#94a3b8;font-size:0.9rem;">Hudson Cicala vs World Cup</p>
    </div>
    """,
    unsafe_allow_html=True,
)

img_path = "Captura de tela 2026-06-02 154425.png"
if os.path.exists(img_path):
    st.sidebar.image(img_path, use_container_width=True)

st.sidebar.markdown(
    """
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.5;">
      Comparação de passes por partida.<br>
      MacAllister e Bouaddi: jogos fixos da Copa do Mundo.<br>
      Hudson: selecione o jogo na área principal.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── MAIN LAYOUT ────────────────────────────────────────────────
st.markdown("## Passes — Comparação de Jogadores")
st.caption("Hudson Cicala vs MacAllister (vs Argélia) vs Bouaddi (vs Brasil)")

selected_hudson_match = st.selectbox(
    "Selecione o jogo de Hudson Cicala para comparar",
    options=hudson_match_names,
    index=0,
    key="hudson_match_selector",
)

hudson_df = hudson_dfs[selected_hudson_match]
hudson_stats = compute_stats(hudson_df, selected_hudson_match)
macallister_stats = compute_stats(macallister_df, MACALLISTER_KEY)
bouaddi_stats = compute_stats(bouaddi_df, BOUADDI_KEY)

players = [
    {
        "name": "Hudson Cicala",
        "subtitle": selected_hudson_match,
        "df": hudson_df,
        "stats": hudson_stats,
        "tone": PLAYER_TONES["Hudson Cicala"],
    },
    {
        "name": "MacAllister",
        "subtitle": "Copa do Mundo — vs Argélia",
        "df": macallister_df,
        "stats": macallister_stats,
        "tone": PLAYER_TONES["MacAllister"],
    },
    {
        "name": "Bouaddi",
        "subtitle": "Copa do Mundo — vs Brasil",
        "df": bouaddi_df,
        "stats": bouaddi_stats,
        "tone": PLAYER_TONES["Bouaddi"],
    },
]

st.markdown("---")
st.markdown("### Mapas de Passe")

map_cols = st.columns(3)
for col, player in zip(map_cols, players):
    with col:
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="player-sub">{player["subtitle"]}</div>', unsafe_allow_html=True)
        render_player_maps(player["df"])

st.markdown("---")
st.markdown("### Estatísticas (Scoreboard — inline targets)")

stat_cols = st.columns(3)
for col, player in zip(stat_cols, players):
    with col:
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="player-sub">{player["subtitle"]}</div>', unsafe_allow_html=True)
        render_player_cards(player["stats"], player["tone"])
