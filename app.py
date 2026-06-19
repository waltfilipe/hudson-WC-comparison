import re
import os
from pathlib import Path
from io import BytesIO

import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen
from databallpy.events.base_event import OPEN_PLAY_XT
from databallpy.models.utils import get_xt_prediction as databallpy_get_xt_prediction
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
COLOR_HIGHLY_PROGRESSIVE = "#1B44A8"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.07
PASS_TONES = ["#5b9bd5", "#3b82f6", "#1d4ed8"]
PLAYER_TONES = {
    "Hudson Cicala": "#5b9bd5",
    "Bentancur": "#70ad47",
    "Bouaddi": "#d4a843",
}
CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NX_XT, NY_XT = 16, 12
DATABALLPY_PITCH_LENGTH = 106.0
DATABALLPY_PITCH_WIDTH = 68.0
LATERAL_MIN_DIST = 12.0
XT_MODEL_HEURISTIC = "heuristic"
XT_MODEL_HEURISTIC_V2 = "heuristic_v2"
XT_MODEL_STATSBOMB = "statsbomb"
XT_MODEL_DATABALLPY = "databallpy"
XT_MODEL_LABELS = {
    XT_MODEL_HEURISTIC: "Heurístico (v1)",
    XT_MODEL_HEURISTIC_V2: "Heurístico v2 (construção)",
    XT_MODEL_STATSBOMB: "StatsBomb (Karun Singh)",
    XT_MODEL_DATABALLPY: "DataBallPy (open play)",
}
XT_MODEL_DESCRIPTIONS = {
    XT_MODEL_HEURISTIC: "Grade sintética normalizada (0–1) por proximidade ao gol.",
    XT_MODEL_HEURISTIC_V2: "Zonas por terço, pico na grande área, progressivo ΔxT >0.15 / >0.35.",
    XT_MODEL_STATSBOMB: "Markov chain — FA WSL 2019/20, probabilidade de gol por célula.",
    XT_MODEL_DATABALLPY: "Modelo pré-treinado open play (264×196), coords convertidas de StatsBomb.",
}
WYSCOUT_PROG_OWN_HALF = 30.0
WYSCOUT_PROG_CROSS_HALF = 15.0
WYSCOUT_PROG_OPP_HALF = 10.0
OPT_ATTACKING_TWO_THIRDS_X = 40.0
OPT_PROGRESS_PCT = 0.25
XT_PROGRESS_ATTACKING_PCT = 0.15
XT_PROGRESS_DEFENSIVE_PCT = 0.20
XT_HIGH_PCT = 0.30
XT_MIN_PASS_DISTANCE = 9.5
XT_EPS = 1e-9
XT_V2_MIN_ABS_DELTA = 0.008
XT_V2_PROG_DELTA = 0.15
XT_V2_HIGH_DELTA = 0.35
XT_V2_NEG_PENALTY_FACTOR = 0.55
XT_V2_PRESSURE_ESCAPE_BONUS = 0.005
XT_V2_PRESSURE_X_MAX = 50.0
XT_V2_WIDE_FRAC = 0.60
XT_V2_FINE_NX = 96
XT_V2_FINE_NY = 64

PROG_MODEL_WYSCOUT = "wyscout"
PROG_MODEL_OPTA = "opta"
PROG_MODEL_XT = "xt"
PROG_MODEL_LABELS = {
    PROG_MODEL_WYSCOUT: "Wyscout",
    PROG_MODEL_OPTA: "Opta",
    PROG_MODEL_XT: "delta xT",
}
PROG_MODEL_DESCRIPTIONS = {
    PROG_MODEL_WYSCOUT: "Metros até o gol por zona (30 m / 15 m / 10 m).",
    PROG_MODEL_OPTA: "Passe completado nos 2/3 ofensivos que aproxima ≥25% do gol.",
    PROG_MODEL_XT: "xT +20% fim def. / +15% fim ataq., dist. >9.5 m; >30% altamente prog.",
}

HUDSON_DOCX = "Passes - Hudson Cicala.docx"
WORLD_CUP_DOCX = "Passes World Cup.docx"
BENTANCUR_KEY = "Bentancur (vs Saudi Arabia)"
BOUADDI_KEY = "Bouaddi (vs Brasil)"
INVERTED_WC_PLAYERS = {BENTANCUR_KEY, BOUADDI_KEY}
WORLD_CUP_MINUTES = 90.0

BENTANCUR_RAW_DATA = """
All Passes – Bentancur (vs Saudi Arabia)

Seta 1: (55.02, 27.93) -> (42.16, 25.77)
Seta 2: (32.29, 46.06) -> (33.70, 59.96)
Seta 3: (48.26, 47.65) -> (65.82, 43.05)
Seta 4: (46.01, 58.08) -> (35.49, 57.33)
Seta 5: (47.32, 59.21) -> (56.90, 58.36)
Seta 6: (46.29, 59.49) -> (56.06, 76.11)
Seta 7: (69.67, 60.80) -> (74.09, 61.74)
Seta 8: (63.29, 42.96) -> (49.76, 75.36)
Seta 9: (62.91, 23.52) -> (69.77, 27.46)
Seta 10: (72.96, 17.98) -> (61.88, 15.16)
Seta 1: (12.10, 19.57) -> (3.27, 48.69)
Seta 2: (34.45, 24.27) -> (27.60, 19.29)
Seta 3: (62.07, 10.37) -> (76.53, 4.55)
Seta 4: (62.82, 43.52) -> (65.17, 57.61)
Seta 5: (67.42, 33.94) -> (68.73, 37.79)
Seta 6: (66.95, 26.62) -> (60.19, 36.29)
Seta 7: (69.96, 36.76) -> (70.61, 26.24)
Seta 8: (69.86, 48.88) -> (88.36, 36.38)
Seta 9: (71.27, 45.96) -> (72.40, 65.40)
Seta 10: (75.12, 32.16) -> (75.78, 18.82)
Seta 11: (79.07, 27.93) -> (85.92, 32.06)
Seta 1: (32.01, 25.49) -> (35.21, 43.90)
Seta 2: (41.69, 58.92) -> (27.97, 56.29)
Seta 3: (51.45, 30.94) -> (35.02, 25.68)
Seta 4: (69.11, 50.85) -> (74.56, 32.72)
Seta 5: (73.81, 39.20) -> (63.47, 47.94)
Seta 6: (72.96, 64.18) -> (80.47, 55.36)
Seta 7: (85.83, 46.06) -> (81.60, 51.88)
Seta 8: (80.85, 73.48) -> (97.47, 61.55)
Seta 9: (92.59, 52.54) -> (88.18, 47.56)
Seta 10: (93.34, 66.63) -> (105.83, 45.02)
Seta 11: (106.68, 59.68) -> (109.31, 74.61)
Seta 1: (78.50, 31.22) -> (62.25, 70.95)
Seta 2: (77.00, 29.72) -> (70.61, 21.64)
Seta 3: (68.73, 48.31) -> (83.39, 48.31)
Seta 4: (60.94, 42.21) -> (74.93, 56.86)
Seta 5: (52.11, 40.61) -> (47.23, 18.07)
Seta 6: (51.64, 48.22) -> (71.27, 41.17)
Seta 7: (72.49, 32.25) -> (46.85, 21.17)
Seta 8: (50.33, 13.56) -> (57.46, 27.18)
Seta 9: (47.70, 11.78) -> (52.20, 25.68)
Seta 10: (46.19, 9.43) -> (30.04, 5.58)
Seta 11: (43.75, 17.88) -> (53.71, 20.04)
Seta 12: (39.90, 16.00) -> (36.90, 21.92)
Seta 13: (27.97, 16.57) -> (20.84, 4.64)
Seta 14: (35.58, 28.50) -> (24.69, 30.37)
Seta 15: (29.38, 29.72) -> (35.30, 39.30)
Seta 16: (34.92, 30.94) -> (20.65, 25.58)
Seta 17: (35.58, 45.96) -> (27.32, 24.93)
Seta 1: (82.16, 47.00) -> (73.05, 62.40)
Seta 2: (80.10, 57.33) -> (97.19, 46.81)
Seta 3: (66.01, 51.32) -> (66.10, 26.80)
Seta 4: (52.11, 50.57) -> (58.50, 37.42)
Seta 5: (46.38, 35.45) -> (29.10, 7.18)
Seta 6: (45.63, 49.25) -> (44.79, 60.61)
Seta 7: (38.87, 56.48) -> (21.40, 73.20)
Seta 8: (33.70, 49.16) -> (16.80, 70.95)
Seta 9: (31.36, 48.03) -> (28.26, 61.18)
Seta 10: (35.02, 47.09) -> (37.46, 40.80)
Seta 1: (78.60, 24.55) -> (63.19, 46.25)
Seta 2: (63.10, 15.72) -> (73.15, 47.00)
Seta 3: (55.12, 34.98) -> (48.26, 8.58)
Seta 4: (41.87, 48.97) -> (41.87, 28.68)
Seta 5: (41.97, 34.98) -> (0.83, 19.85)
Seta 6: (40.37, 21.17) -> (45.35, 20.51)
Seta 7: (39.06, 18.92) -> (28.35, 7.08)
Seta 8: (36.52, 20.23) -> (33.05, 42.77)
Seta 9: (32.86, 16.47) -> (32.20, 33.00)
Seta 10: (30.23, 14.88) -> (14.26, 27.18)
Seta 11: (22.15, 9.90) -> (25.16, 30.19)
Seta 12: (24.50, 30.47) -> (18.96, 12.25)
Seta 1: (87.52, 69.54) -> (108.84, 53.76)
Seta 2: (79.91, 41.93) -> (65.54, 63.81)
Seta 3: (71.46, 47.18) -> (68.83, 29.53)
Seta 4: (70.99, 25.58) -> (73.81, 51.04)
Seta 5: (41.78, 53.29) -> (42.34, 28.59)
Seta 6: (36.90, 49.72) -> (31.17, 37.89)
Seta 7: (34.45, 39.67) -> (33.05, 52.73)
Seta 8: (38.68, 35.45) -> (38.12, 56.67)
Seta 9: (39.81, 31.50) -> (29.85, 33.10)

Passes Errados
Seta 1: (36.43, 33.75) -> (0.46, 18.82)
Seta 2: (30.79, 12.06) -> (13.61, 25.96)
Seta 3: (69.58, 33.00) -> (71.55, 38.83)
Seta 4: (72.96, 63.90) -> (78.13, 65.78)
Seta 5: (64.23, 20.51) -> (70.05, 23.99)
Seta 6: (89.02, 45.87) -> (83.95, 53.29)
Seta 7: (93.90, 51.04) -> (90.15, 46.43)
Seta 8: (78.88, 23.89) -> (74.09, 17.13)
Seta 9: (78.78, 25.30) -> (85.26, 28.78)
Seta 10: (48.17, 28.50) -> (34.08, 22.20)
Seta 11: (34.45, 17.69) -> (29.29, 15.16)
Seta 12: (34.92, 26.15) -> (19.71, 23.42)
"""

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


def is_progressive_wyscout(x_start, y_start, x_end, y_end) -> bool:
    """Wyscout progressive pass: distance-to-goal reduction by pitch zone."""
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    progress = start_dist - end_dist
    if progress <= 0:
        return False
    start_own = x_start < HALF_LINE_X
    end_own = x_end < HALF_LINE_X
    start_opp = x_start >= HALF_LINE_X
    end_opp = x_end >= HALF_LINE_X
    if start_own and end_own:
        return progress >= WYSCOUT_PROG_OWN_HALF
    if start_opp and end_opp:
        return progress >= WYSCOUT_PROG_OPP_HALF
    return progress >= WYSCOUT_PROG_CROSS_HALF


def is_progressive_opta(x_start, y_start, x_end, y_end) -> bool:
    """Opta progressive pass: completed pass from attacking 2/3, ≥25% closer to goal."""
    if x_start < OPT_ATTACKING_TWO_THIRDS_X:
        return False
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    if start_dist <= 0:
        return False
    return (start_dist - end_dist) / start_dist >= OPT_PROGRESS_PCT


def xt_relative_increase(xt_start: float, xt_end: float) -> float:
    if xt_end <= xt_start:
        return 0.0
    if xt_start <= XT_EPS:
        return float("inf")
    return (xt_end - xt_start) / xt_start


def classify_xt_progressive(
    xt_start: float,
    xt_end: float,
    x_end: float,
    pass_distance: float,
) -> str:
    """Returns 'none', 'progressive', or 'highly'."""
    if pass_distance <= XT_MIN_PASS_DISTANCE:
        return "none"
    if xt_end <= xt_start:
        return "none"
    pct = xt_relative_increase(xt_start, xt_end)
    min_pct = XT_PROGRESS_DEFENSIVE_PCT if x_end < HALF_LINE_X else XT_PROGRESS_ATTACKING_PCT
    if pct < min_pct:
        return "none"
    if pct == float("inf") or pct > XT_HIGH_PCT:
        return "highly"
    return "progressive"


def classify_xt_progressive_v2(
    xt_start: float,
    xt_end: float,
    x_end: float,
    pass_distance: float,
) -> str:
    """v2: progressivo por ΔxT absoluto (>0.15 / >0.35)."""
    if pass_distance <= XT_MIN_PASS_DISTANCE:
        return "none"
    delta_abs = xt_end - xt_start
    if delta_abs <= XT_V2_PROG_DELTA:
        return "none"
    if delta_abs > XT_V2_HIGH_DELTA:
        return "highly"
    return "progressive"


def classify_xt_progressive_for_model(
    xt_start: float,
    xt_end: float,
    x_end: float,
    pass_distance: float,
    xt_model: str,
) -> str:
    if xt_model == XT_MODEL_HEURISTIC_V2:
        return classify_xt_progressive_v2(xt_start, xt_end, x_end, pass_distance)
    return classify_xt_progressive(xt_start, xt_end, x_end, pass_distance)


def is_progressive_xt_attempt(xt_start: float, xt_end: float, x_end: float, pass_distance: float) -> bool:
    return classify_xt_progressive(xt_start, xt_end, x_end, pass_distance) in ("progressive", "highly")


def is_progressive_xt_attempt_for_model(
    xt_start: float, xt_end: float, x_end: float, pass_distance: float, xt_model: str,
) -> bool:
    return classify_xt_progressive_for_model(
        xt_start, xt_end, x_end, pass_distance, xt_model,
    ) in ("progressive", "highly")


def is_progressive_attempt(row, model: str, xt_model: str = XT_MODEL_HEURISTIC) -> bool:
    if model == PROG_MODEL_WYSCOUT:
        return is_progressive_wyscout(row.x_start, row.y_start, row.x_end, row.y_end)
    if model == PROG_MODEL_OPTA:
        return is_progressive_opta(row.x_start, row.y_start, row.x_end, row.y_end)
    return is_progressive_xt_attempt_for_model(
        row.xt_start, row.xt_end, row.x_end, row.pass_distance, xt_model,
    )


def apply_progressive_model(
    df: pd.DataFrame, model: str, xt_model: str = XT_MODEL_HEURISTIC,
) -> pd.DataFrame:
    df = df.copy()
    progressive_flags = []
    highly_flags = []
    for row in df.itertuples(index=False):
        if model == PROG_MODEL_WYSCOUT:
            attempt = is_progressive_wyscout(row.x_start, row.y_start, row.x_end, row.y_end)
            highly = False
        elif model == PROG_MODEL_OPTA:
            attempt = is_progressive_opta(row.x_start, row.y_start, row.x_end, row.y_end)
            highly = False
        else:
            xt_cat = classify_xt_progressive_for_model(
                row.xt_start, row.xt_end, row.x_end, row.pass_distance, xt_model,
            )
            attempt = xt_cat in ("progressive", "highly")
            highly = xt_cat == "highly"
        progressive_flags.append(row.is_won and attempt)
        highly_flags.append(row.is_won and highly)
    df["progressive"] = progressive_flags
    df["highly_progressive"] = highly_flags
    return df


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


@st.cache_data(show_spinner=False)
def compute_heuristic_xt_grid(NX=16, NY=12, sub=24):
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


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _zone_x_threat(x: np.ndarray) -> np.ndarray:
    """Threat by pitch third; attacking profile peaks near the box, fades at the goal line."""
    x = np.clip(x, 0.0, FIELD_X)
    threat = np.zeros_like(x, dtype=float)
    def_mask = x < OPT_ATTACKING_TWO_THIRDS_X
    mid_mask = (x >= OPT_ATTACKING_TWO_THIRDS_X) & (x < FINAL_THIRD_LINE_X)
    att_mask = x >= FINAL_THIRD_LINE_X

    threat[def_mask] = 0.12 * (x[def_mask] / OPT_ATTACKING_TWO_THIRDS_X)
    mid_t = (x[mid_mask] - OPT_ATTACKING_TWO_THIRDS_X) / (
        FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X
    )
    threat[mid_mask] = 0.12 + 0.34 * _smoothstep(mid_t)

    att_t = (x[att_mask] - FINAL_THIRD_LINE_X) / (FIELD_X - FINAL_THIRD_LINE_X)
    # Peak near penalty area (t≈0.55 → x≈102), smooth decay toward the byline (t→1)
    rise = _smoothstep(att_t / 0.58)
    byline_fade = 1.0 - 0.48 * _smoothstep((att_t - 0.52) / 0.48)
    threat[att_mask] = 0.48 + 0.28 * rise * byline_fade
    return threat


def _centrality(y: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs((y / FIELD_Y) - 0.5) * 2.0


def _location_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Centrality bonus only in the attacking third (flat across width in build-up zones)."""
    cent = _centrality(y)
    att_t = np.clip((x - FINAL_THIRD_LINE_X) / (FIELD_X - FINAL_THIRD_LINE_X), 0.0, 1.0)
    att_gate = _smoothstep(att_t)
    return 0.93 + 0.07 * cent * att_gate


def _lateral_frac(y: float) -> float:
    return float(abs(y - GOAL_Y) / (FIELD_Y / 2.0))


@st.cache_data(show_spinner=False)
def compute_heuristic_v2_fine_grid(nx: int = XT_V2_FINE_NX, ny: int = XT_V2_FINE_NY) -> np.ndarray:
    """Idea 2: high-resolution grid for bilinear lookup."""
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    base = _zone_x_threat(Xc) * _location_factor(Xc, Yc)
    return (base - base.min()) / (base.max() - base.min() + 1e-12)


def xt_value_bilinear(x: float, y: float, fine_grid: np.ndarray) -> float:
    nx, ny = fine_grid.shape[1], fine_grid.shape[0]
    fx = float(np.clip(x / FIELD_X * (nx - 1), 0.0, nx - 1))
    fy = float(np.clip(y / FIELD_Y * (ny - 1), 0.0, ny - 1))
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    tx, ty = fx - x0, fy - y0
    v00, v10 = fine_grid[y0, x0], fine_grid[y0, x1]
    v01, v11 = fine_grid[y1, x0], fine_grid[y1, x1]
    return float(
        (1 - tx) * (1 - ty) * v00
        + tx * (1 - ty) * v10
        + (1 - tx) * ty * v01
        + tx * ty * v11
    )


@st.cache_data(show_spinner=False)
def compute_heuristic_v2_xt_grid() -> np.ndarray:
    """16×12 display grid sampled via bilinear interpolation."""
    fine = compute_heuristic_v2_fine_grid()
    grid = np.zeros((NY_XT, NX_XT))
    x_bins = np.linspace(0, FIELD_X, NX_XT + 1)
    y_bins = np.linspace(0, FIELD_Y, NY_XT + 1)
    for iy in range(NY_XT):
        for ix in range(NX_XT):
            cx = (x_bins[ix] + x_bins[ix + 1]) / 2.0
            cy = (y_bins[iy] + y_bins[iy + 1]) / 2.0
            grid[iy, ix] = xt_value_bilinear(cx, cy, fine)
    return grid


def _adjust_heuristic_v2_pass_delta(row) -> float:
    """Idea 5: softer penalty on useful recycle / pressure escape."""
    if not row.is_won:
        return 0.0
    raw = float(row.xt_end - row.xt_start)
    if raw >= 0:
        return raw
    lat_start = _lateral_frac(row.y_start)
    lat_end = _lateral_frac(row.y_end)
    adjusted = raw * (XT_V2_NEG_PENALTY_FACTOR if lat_end < lat_start else 1.0)
    if (
        row.x_start < XT_V2_PRESSURE_X_MAX
        and lat_start > XT_V2_WIDE_FRAC
        and lat_end < lat_start - 0.12
    ):
        adjusted += XT_V2_PRESSURE_ESCAPE_BONUS
    return adjusted


def apply_heuristic_v2_xt(df: pd.DataFrame) -> pd.DataFrame:
    fine = compute_heuristic_v2_fine_grid()
    out = df.copy()
    out["xt_start"] = out.apply(lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1)
    out["xt_end"] = out.apply(lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1)
    out["delta_xt"] = out.apply(_adjust_heuristic_v2_pass_delta, axis=1)
    return out


@st.cache_data(show_spinner=False)
def compute_statsbomb_xt_grid():
    """Karun Singh xT on StatsBomb open data (FA WSL 2019/20)."""
    parser = Sbopen()
    pitch = Pitch(line_zorder=2)
    bins = (NX_XT, NY_XT)
    df_match = parser.match(competition_id=37, season_id=42)
    cols = [
        "match_id", "id", "type_name", "x", "y", "end_x", "end_y", "outcome_name",
    ]
    frames = []
    for match_id in df_match.match_id.unique():
        event = parser.event(match_id)[0]
        event = event.loc[event.type_name.isin(["Carry", "Shot", "Pass"]), cols].copy()
        event["goal"] = event["outcome_name"] == "Goal"
        event["shoot"] = event["type_name"] == "Shot"
        event["move"] = event["type_name"] != "Shot"
        frames.append(event)
    event = pd.concat(frames, ignore_index=True)

    shot_probability = pitch.bin_statistic(
        event["x"], event["y"], values=event["shoot"], statistic="mean", bins=bins,
    )
    move_probability = pitch.bin_statistic(
        event["x"], event["y"], values=event["move"], statistic="mean", bins=bins,
    )
    goal_probability = pitch.bin_statistic(
        event.loc[event["shoot"], "x"],
        event.loc[event["shoot"], "y"],
        event.loc[event["shoot"], "goal"],
        statistic="mean",
        bins=bins,
    )

    move = event[event["move"]].copy()
    bin_start_locations = pitch.bin_statistic(move["x"], move["y"], bins=bins)
    move = move[bin_start_locations["inside"]].copy()
    bin_end_locations = pitch.bin_statistic(move["end_x"], move["end_y"], bins=bins)
    move_success = move[(bin_end_locations["inside"]) & (move["outcome_name"].isnull())].copy()

    bin_success_start = pitch.bin_statistic(move_success["x"], move_success["y"], bins=bins)
    bin_success_end = pitch.bin_statistic(move_success["end_x"], move_success["end_y"], bins=bins)
    df_bin = pd.DataFrame({
        "x": bin_success_start["binnumber"][0],
        "y": bin_success_start["binnumber"][1],
        "end_x": bin_success_end["binnumber"][0],
        "end_y": bin_success_end["binnumber"][1],
    })
    bin_counts = df_bin.value_counts().reset_index(name="bin_counts")

    num_y, num_x = shot_probability["statistic"].shape
    move_transition_matrix = np.zeros((num_y, num_x, num_y, num_x))
    move_transition_matrix[
        bin_counts["y"], bin_counts["x"], bin_counts["end_y"], bin_counts["end_x"]
    ] = bin_counts.bin_counts.values

    bin_start_stat = pitch.bin_statistic(move["x"], move["y"], bins=bins)
    bin_start_stat = np.expand_dims(bin_start_stat["statistic"], (2, 3))
    move_transition_matrix = np.divide(
        move_transition_matrix,
        bin_start_stat,
        out=np.zeros_like(move_transition_matrix),
        where=bin_start_stat != 0,
    )

    move_transition_matrix = np.nan_to_num(move_transition_matrix)
    shot_probability_matrix = np.nan_to_num(shot_probability["statistic"])
    move_probability_matrix = np.nan_to_num(move_probability["statistic"])
    goal_probability_matrix = np.nan_to_num(goal_probability["statistic"])

    xt = np.multiply(shot_probability_matrix, goal_probability_matrix)
    while np.any(np.abs(xt - (xt_copy := xt.copy())) > 0.00001):
        xt = np.multiply(shot_probability_matrix, goal_probability_matrix) + np.multiply(
            move_probability_matrix,
            np.multiply(move_transition_matrix, np.expand_dims(xt, axis=(0, 1))).sum(axis=(2, 3)),
        )
    return xt


def statsbomb_to_databallpy(x_sb: float, y_sb: float) -> tuple[float, float]:
    """Convert StatsBomb 120×80 (attack → right) to DataBallPy 106×68 (centered, goal at +x)."""
    x_db = (x_sb / FIELD_X) * DATABALLPY_PITCH_LENGTH - (DATABALLPY_PITCH_LENGTH / 2.0)
    y_db = (y_sb / FIELD_Y) * DATABALLPY_PITCH_WIDTH - (DATABALLPY_PITCH_WIDTH / 2.0)
    return x_db, y_db


def databallpy_xt_value(x_sb: float, y_sb: float) -> float:
    x_db, y_db = statsbomb_to_databallpy(x_sb, y_sb)
    return float(databallpy_get_xt_prediction(x_db, y_db, OPEN_PLAY_XT))


@st.cache_data(show_spinner=False)
def compute_databallpy_xt_grid() -> np.ndarray:
    """Resample DataBallPy open-play xT onto the app 16×12 StatsBomb grid."""
    grid = np.zeros((NY_XT, NX_XT))
    x_bins = np.linspace(0, FIELD_X, NX_XT + 1)
    y_bins = np.linspace(0, FIELD_Y, NY_XT + 1)
    for iy in range(NY_XT):
        for ix in range(NX_XT):
            cx = (x_bins[ix] + x_bins[ix + 1]) / 2.0
            cy = (y_bins[iy] + y_bins[iy + 1]) / 2.0
            grid[iy, ix] = databallpy_xt_value(cx, cy)
    return grid


def get_xt_grid(xt_model: str) -> np.ndarray:
    if xt_model == XT_MODEL_STATSBOMB:
        return compute_statsbomb_xt_grid()
    if xt_model == XT_MODEL_DATABALLPY:
        return compute_databallpy_xt_grid()
    if xt_model == XT_MODEL_HEURISTIC_V2:
        return compute_heuristic_v2_xt_grid()
    return compute_heuristic_xt_grid()


def apply_xt_model(df: pd.DataFrame, xt_model: str) -> pd.DataFrame:
    if xt_model == XT_MODEL_HEURISTIC_V2:
        return apply_heuristic_v2_xt(df)
    if xt_model == XT_MODEL_DATABALLPY:
        out = df.copy()
        out["xt_start"] = out.apply(lambda r: databallpy_xt_value(r["x_start"], r["y_start"]), axis=1)
        out["xt_end"] = out.apply(lambda r: databallpy_xt_value(r["x_end"], r["y_end"]), axis=1)
        out["delta_xt"] = np.where(out["is_won"], out["xt_end"] - out["xt_start"], 0.0)
        return out
    return apply_xt_grid(df, get_xt_grid(xt_model))


def xt_value_from_grid(x: float, y: float, grid: np.ndarray) -> float:
    ix = int(np.clip((x / FIELD_X) * NX_XT, 0, NX_XT - 1))
    iy = int(np.clip((y / FIELD_Y) * NY_XT, 0, NY_XT - 1))
    return float(grid[iy, ix])


def apply_xt_grid(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["xt_start"] = out.apply(lambda r: xt_value_from_grid(r["x_start"], r["y_start"], grid), axis=1)
    out["xt_end"] = out.apply(lambda r: xt_value_from_grid(r["x_end"], r["y_end"], grid), axis=1)
    out["delta_xt"] = np.where(out["is_won"], out["xt_end"] - out["xt_start"], 0.0)
    return out


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
    if match_name in INVERTED_WC_PLAYERS:
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


def invert_pitch_coords(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Rotate pitch coordinates 180° clockwise (same as rotating the pass map)."""
    return FIELD_X - x1, FIELD_Y - y1, FIELD_X - x2, FIELD_Y - y2


def _pass_match_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) + np.max(np.abs(a - b)))


def reconcile_failed_passes(
    totais_coords: list[np.ndarray],
    errados_coords: list[np.ndarray],
) -> list[tuple]:
    """Match failed passes to the closest Totais entry and keep only one LOST event."""
    used_totais: set[int] = set()
    used_errados: set[int] = set()
    lost_by_index: dict[int, np.ndarray] = {}

    candidate_pairs: list[tuple[float, int, int]] = []
    for err_idx, err in enumerate(errados_coords):
        for tot_idx, tot in enumerate(totais_coords):
            candidate_pairs.append((_pass_match_distance(err, tot), err_idx, tot_idx))
    candidate_pairs.sort(key=lambda item: item[0])

    for _, err_idx, tot_idx in candidate_pairs:
        if err_idx in used_errados or tot_idx in used_totais:
            continue
        used_errados.add(err_idx)
        used_totais.add(tot_idx)
        lost_by_index[tot_idx] = errados_coords[err_idx]

    for err_idx, err in enumerate(errados_coords):
        if err_idx in used_errados:
            continue
        best_idx = None
        best_dist = float("inf")
        for tot_idx, tot in enumerate(totais_coords):
            if tot_idx in used_totais:
                continue
            dist = _pass_match_distance(err, tot)
            if dist < best_dist:
                best_dist = dist
                best_idx = tot_idx
        if best_idx is not None:
            used_errados.add(err_idx)
            used_totais.add(best_idx)
            lost_by_index[best_idx] = err

    events = []
    for tot_idx, tot in enumerate(totais_coords):
        if tot_idx in lost_by_index:
            err = lost_by_index[tot_idx]
            events.append(("PASS LOST", float(err[0]), float(err[1]), float(err[2]), float(err[3]), None))
        else:
            events.append(("PASS WON", float(tot[0]), float(tot[1]), float(tot[2]), float(tot[3]), None))
    return events


def apply_player_orientation(player: str, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    if player in INVERTED_WC_PLAYERS:
        return invert_pitch_coords(x1, y1, x2, y2)
    return x1, y1, x2, y2


def parse_player_passes_text(raw_text: str) -> dict:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    players_totais: dict[str, list[np.ndarray]] = {}
    players_errados: dict[str, list[np.ndarray]] = {}
    current_player = None
    current_mode = "totais"
    re_player_tot = re.compile(r"^Passes\s+Totais\s*\|\s*(.+)$", re.IGNORECASE)
    re_player_all = re.compile(r"^All\s+Passes\s*[–-]\s*(.+)$", re.IGNORECASE)
    re_fail_section = re.compile(r"^Passes\s+Errados:?$", re.IGNORECASE)
    re_arrow = re.compile(
        r"^Seta\s+\d+:\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\s*->\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)$",
        re.IGNORECASE,
    )

    for ln in lines:
        m_player = re_player_tot.match(ln) or re_player_all.match(ln)
        if m_player:
            current_player = m_player.group(1).strip()
            players_totais.setdefault(current_player, [])
            players_errados.setdefault(current_player, [])
            current_mode = "totais"
            continue
        if not current_player:
            continue
        if re_fail_section.match(ln):
            current_mode = "errados"
            continue
        m_arrow = re_arrow.match(ln)
        if not m_arrow:
            continue
        coords = np.array(list(map(float, m_arrow.groups())))
        if current_mode == "totais":
            players_totais[current_player].append(coords)
        else:
            players_errados[current_player].append(coords)

    players = {}
    for player, totais in players_totais.items():
        if not totais:
            continue
        errados = players_errados.get(player, [])
        events = reconcile_failed_passes(totais, errados)
        events = [
            (
                state,
                *apply_player_orientation(player, x1, y1, x2, y2),
                video,
            )
            for state, x1, y1, x2, y2, video in events
        ]
        players[player] = events
    return players


def parse_world_cup_docx(raw_text: str) -> dict:
    return parse_player_passes_text(raw_text)


def events_to_dataframe(events: list, match_name: str) -> pd.DataFrame:
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].eq("PASS WON")
    dfm["progressive"] = False
    dfm["highly_progressive"] = False
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1,
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"].isin(["lateral_left", "lateral_right"])
    dfm["pass_distance"] = np.sqrt((dfm["x_end"] - dfm["x_start"]) ** 2 + (dfm["y_end"] - dfm["y_start"]) ** 2)
    dfm["xt_start"] = 0.0
    dfm["xt_end"] = 0.0
    dfm["delta_xt"] = 0.0
    return dfm


def prepare_player_df(df: pd.DataFrame, xt_model: str, prog_model: str) -> pd.DataFrame:
    return apply_progressive_model(apply_xt_model(df, xt_model), prog_model, xt_model)


def load_all_pass_data() -> tuple[dict, dict]:
    hudson_path = Path(HUDSON_DOCX)
    if not hudson_path.exists():
        return {}, {}

    hudson_raw = parse_hudson_docx(read_docx_text(hudson_path))
    wc_raw: dict[str, list] = {}

    bentancur_parsed = parse_player_passes_text(BENTANCUR_RAW_DATA)
    if BENTANCUR_KEY in bentancur_parsed:
        wc_raw[BENTANCUR_KEY] = bentancur_parsed[BENTANCUR_KEY]

    wc_path = Path(WORLD_CUP_DOCX)
    if wc_path.exists():
        wc_doc = parse_player_passes_text(read_docx_text(wc_path))
        if BOUADDI_KEY in wc_doc:
            wc_raw[BOUADDI_KEY] = wc_doc[BOUADDI_KEY]

    hudson_dfs = {name: events_to_dataframe(events, name) for name, events in hudson_raw.items()}
    wc_dfs = {name: events_to_dataframe(events, name) for name, events in wc_raw.items()}
    return hudson_dfs, wc_dfs


def compute_stats(
    df: pd.DataFrame,
    match_name: str,
    prog_model: str = PROG_MODEL_WYSCOUT,
    xt_model: str = XT_MODEL_HEURISTIC,
) -> dict:
    df = apply_progressive_model(df, prog_model, xt_model)
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
            "highly_progressive": 0,
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
    highly_progressive = int(df["highly_progressive"].sum())
    progressive_unsuccessful = int(
        (~df["is_won"] & df.apply(lambda r: is_progressive_attempt(r, prog_model, xt_model), axis=1)).sum()
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
    pos_count = int((df["is_won"] & (df["delta_xt"] > 0)).sum())
    pos_pct = (pos_count / total * 100.0) if total > 0 else 0.0
    high_xt_thresh = 0.1 if df["delta_xt"].abs().max() > 0.05 else 0.02
    high_xt = int((df["delta_xt"] > high_xt_thresh).sum())
    sum_dxt = float(df.loc[df["is_won"], "delta_xt"].sum())
    neg_xt = float(df.loc[df["is_won"] & (df["delta_xt"] < 0), "delta_xt"].sum())
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
        "highly_progressive": highly_progressive,
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


def _item_sep(idx: int, total: int) -> str:
    return "" if idx == total - 1 else f"margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid {CARD_INNER_BORDER};"


def _accent_rgb(border_color: str) -> tuple[int, int, int]:
    h = border_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _stats_card_shell_html(title: str, border_color: str, body_html: str) -> str:
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


def _simple_body_scoreboard(items: list[tuple[str, str]]) -> str:
    body = ""
    for idx, (label, disp_val) in enumerate(items):
        body += f'<div style="{_item_sep(idx, len(items))}">'
        body += (
            '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;">'
            f'<span style="font-size:{CARD_LABEL_TEXT};color:#c7cdda;font-weight:600;">{label}</span>'
            f'<span style="font-size:28px;color:#ffffff;font-weight:700;line-height:1;">{disp_val}</span>'
            "</div>"
        )
        body += "</div>"
    return body


def stats_section_card(title: str, border_color: str, items: list[tuple[str, str]]) -> None:
    inner = _simple_body_scoreboard(items)
    st.markdown(_stats_card_shell_html(title, border_color, inner), unsafe_allow_html=True)


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


def draw_pass_map(df, prog_model: str = PROG_MODEL_WYSCOUT):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        is_lost = not row["is_won"]
        is_highly = bool(row.get("highly_progressive", False))
        is_prog = bool(row["progressive"])
        if is_lost:
            color, alpha = COLOR_FAIL, 0.72
        elif is_highly:
            color, alpha = COLOR_HIGHLY_PROGRESSIVE, 0.95
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
    legend_handles = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completed", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incomplete", alpha=0.90),
    ]
    if prog_model == PROG_MODEL_XT:
        legend_handles.insert(
            2,
            Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=2.0, label="Highly Progressive", alpha=0.95),
        )
    leg = ax.legend(
        handles=legend_handles,
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


def draw_top_xt_map(df, top_n=5):
    fig, ax, pitch = _base_pitch()
    top_passes = (
        df[(df["is_won"]) & (df["delta_xt"] > 0)]
        .sort_values("delta_xt", ascending=False)
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )
    cursor_points = []
    if not top_passes.empty:
        vmax = max(float(top_passes["delta_xt"].max()), 0.01)
        norm = Normalize(vmin=0.0, vmax=vmax)
        for _, row in top_passes.iterrows():
            val = float(row["delta_xt"])
            color = CMAP_TOP10(norm(np.clip(val / vmax, 0.05, 1.0)))
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

    sm = plt.cm.ScalarMappable(
        cmap=CMAP_TOP10,
        norm=Normalize(
            vmin=0.0,
            vmax=max(float(top_passes["delta_xt"].max()), 0.01) if not top_passes.empty else 0.1,
        ),
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
    cbar.set_label("Pass Impact", color="#ffffff", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    g = grid.astype(float)
    return (g - g.min()) / (g.max() - g.min() + 1e-12)


def draw_xt_grid_map(grid: np.ndarray, title: str, value_fmt: str = ".2f", as_percent: bool = False):
    fig, ax, pitch = _base_pitch()
    x_bins = np.linspace(0, FIELD_X, NX_XT + 1)
    y_bins = np.linspace(0, FIELD_Y, NY_XT + 1)
    vmax = max(float(grid.max()), 1e-6)
    cmap = LinearSegmentedColormap.from_list("xt", ["#1a1a2e", "#3b82f6", "#fbbf24", "#ef4444"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = vmax * 0.45
    for iy in range(NY_XT):
        for ix in range(NX_XT):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=cmap(norm(value)),
                    edgecolor=(1, 1, 1, 0.15),
                    lw=0.4,
                    alpha=0.95,
                    zorder=2,
                )
            )
            label = f"{value * 100:.1f}%" if as_percent else f"{value:{value_fmt}}"
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, label,
                ha="center", va="center",
                color="#000000" if value <= threshold else "#ffffff",
                fontsize=5.5, fontweight="600", zorder=4,
            )
    ax.set_title(title, color="#eef1f7", fontsize=10, pad=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.55)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig


def draw_xt_diff_map(grid_a: np.ndarray, grid_b: np.ndarray, title: str):
    """Heatmap of cell-wise difference (grid_a normalized − grid_b normalized)."""
    diff = _normalize_grid(grid_a) - _normalize_grid(grid_b)
    fig, ax, pitch = _base_pitch()
    x_bins = np.linspace(0, FIELD_X, NX_XT + 1)
    y_bins = np.linspace(0, FIELD_Y, NY_XT + 1)
    vmax = max(float(np.abs(diff).max()), 0.05)
    cmap = LinearSegmentedColormap.from_list(
        "diff", ["#2563eb", "#1a1a2e", "#1a1a2e", "#ef4444"],
    )
    norm = Normalize(vmin=-vmax, vmax=vmax)
    for iy in range(NY_XT):
        for ix in range(NX_XT):
            value = float(diff[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=cmap(norm(value)),
                    edgecolor=(1, 1, 1, 0.12),
                    lw=0.4,
                    alpha=0.95,
                    zorder=2,
                )
            )
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, f"{value:+.2f}",
                ha="center", va="center",
                color="#ffffff" if abs(value) > vmax * 0.35 else "#c7cdda",
                fontsize=5.5, fontweight="600", zorder=4,
            )
    ax.set_title(title, color="#eef1f7", fontsize=9, pad=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.55)
    cbar.set_label("Δ normalizado", color="#ffffff", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig


    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig


def draw_heuristic_v1_v2_impact_chart(player_rows: list[dict]):
    names = [p["name"] for p in player_rows]
    v1_vals = [p["v1_impact"] for p in player_rows]
    v2_vals = [p["v2_impact"] for p in player_rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, v1_vals, w, label="Heurístico v1", color="#5b9bd5", alpha=0.92)
    ax.bar(x + w / 2, v2_vals, w, label="Heurístico v2", color="#70ad47", alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="#c7cdda", fontsize=9)
    ax.set_ylabel("Pass Impact (Σ ΔxT)", color="#c7cdda", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_title("Pass Impact por jogador — v1 vs v2", color="#eef1f7", fontsize=11, pad=10)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444466", labelcolor="#eef1f7", fontsize=8)
    ax.axhline(0, color="#444466", lw=0.8)
    for spine in ax.spines.values():
        spine.set_color("#444466")
    fig.tight_layout()
    return _save_fig(fig), fig


def draw_pass_delta_scatter(df_v1: pd.DataFrame, df_v2: pd.DataFrame, title: str):
    left = df_v1[df_v1["is_won"]][["number", "delta_xt"]].rename(columns={"delta_xt": "v1"})
    right = df_v2[df_v2["is_won"]][["number", "delta_xt"]].rename(columns={"delta_xt": "v2"})
    merged = left.merge(right, on="number", how="inner")
    fig, ax = plt.subplots(figsize=(4.8, 4.5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    if not merged.empty:
        lim = max(merged[["v1", "v2"]].abs().max().max(), 0.02)
        ax.scatter(merged["v1"], merged["v2"], c="#d4a843", alpha=0.75, s=28, edgecolors="white", linewidths=0.4)
        ax.plot([-lim, lim], [-lim, lim], "--", color="#64748b", lw=1.0, alpha=0.8)
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_ylim(-lim * 1.05, lim * 1.05)
        corr = merged["v1"].corr(merged["v2"])
        ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, color="#eef1f7", fontsize=9, va="top")
    ax.set_xlabel("ΔxT v1", color="#c7cdda", fontsize=9)
    ax.set_ylabel("ΔxT v2", color="#c7cdda", fontsize=9)
    ax.set_title(title, color="#eef1f7", fontsize=10, pad=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#444466")
    fig.tight_layout()
    return _save_fig(fig), fig


def render_heuristic_v1_v2_comparison(
    hudson_base: pd.DataFrame,
    bentancur_base: pd.DataFrame,
    bouaddi_base: pd.DataFrame,
    prog_model: str,
    hudson_label: str,
):
    grid_v1 = compute_heuristic_xt_grid()
    grid_v2 = compute_heuristic_v2_xt_grid()

    st.markdown("---")
    st.markdown("### Comparativo gráfico — xT Heurístico v1 vs v2")
    st.caption(
        "v2: pico de ameaça na grande área, centralidade só no terço ofensivo, "
        "progressivo ΔxT >0.15 / altamente >0.35."
    )

    gcols = st.columns(3)
    grid_panels = [
        (grid_v1, "Heurístico v1"),
        (grid_v2, "Heurístico v2"),
        (grid_v2, "Δ v2 − v1"),
    ]
    for col, (grid, label) in zip(gcols, grid_panels[:2]):
        with col:
            img, fig = draw_xt_grid_map(grid, label, value_fmt=".2f")
            plt.close(fig)
            st.image(img, use_container_width=True)
    with gcols[2]:
        img, fig = draw_xt_diff_map(grid_v2, grid_v1, "Δ v2 − v1 (normalizados)")
        plt.close(fig)
        st.image(img, use_container_width=True)

    bases = [
        ("Hudson Cicala", hudson_base, hudson_label),
        ("Bentancur", bentancur_base, "Copa — vs Arábia Saudita"),
        ("Bouaddi", bouaddi_base, "Copa — vs Brasil"),
    ]
    impact_rows = []
    for name, base_df, _ in bases:
        df_v1 = prepare_player_df(base_df, XT_MODEL_HEURISTIC, prog_model)
        df_v2 = prepare_player_df(base_df, XT_MODEL_HEURISTIC_V2, prog_model)
        impact_rows.append({
            "name": name,
            "v1_impact": float(df_v1.loc[df_v1["is_won"], "delta_xt"].sum()),
            "v2_impact": float(df_v2.loc[df_v2["is_won"], "delta_xt"].sum()),
            "df_v1": df_v1,
            "df_v2": df_v2,
        })

    chart_col, scatter_col = st.columns([1.4, 1.0])
    with chart_col:
        img, fig = draw_heuristic_v1_v2_impact_chart(impact_rows)
        plt.close(fig)
        st.image(img, use_container_width=True)
    with scatter_col:
        bent = next(r for r in impact_rows if r["name"] == "Bentancur")
        img, fig = draw_pass_delta_scatter(
            bent["df_v1"], bent["df_v2"], "Passes Bentancur — ΔxT v1 vs v2",
        )
        plt.close(fig)
        st.image(img, use_container_width=True)

    st.markdown("#### Resumo numérico (jogo atual)")
    summary_rows = []
    for row in impact_rows:
        df_v1, df_v2 = row["df_v1"], row["df_v2"]
        won_v1, won_v2 = df_v1[df_v1["is_won"]], df_v2[df_v2["is_won"]]
        summary_rows.append({
            "Jogador": row["name"],
            "ΣΔxT v1": round(float(won_v1["delta_xt"].sum()), 3),
            "ΣΔxT v2": round(float(won_v2["delta_xt"].sum()), 3),
            "% Δ>0 v1": round((won_v1["delta_xt"] > 0).mean() * 100, 1),
            "% Δ>0 v2": round((won_v2["delta_xt"] > 0).mean() * 100, 1),
            "Prog. v1": int(df_v1["progressive"].sum()),
            "Prog. v2": int(df_v2["progressive"].sum()),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _pdf_draw_image_row(c, images_labels, page_w, page_h, y_top, row_h, margin_x=36, gap=10):
    from reportlab.lib.utils import ImageReader

    n = len(images_labels)
    usable_w = page_w - 2 * margin_x - gap * (n - 1)
    panel_w = usable_w / n
    y0 = y_top - row_h
    for idx, (img, label) in enumerate(images_labels):
        x0 = margin_x + idx * (panel_w + gap)
        c.setFillColorRGB(0.93, 0.95, 0.97)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x0, y_top + 4, label)
        img_buf = BytesIO()
        img.save(img_buf, format="PNG")
        img_buf.seek(0)
        aspect = img.width / img.height
        draw_h = row_h
        draw_w = draw_h * aspect
        if draw_w > panel_w:
            draw_w = panel_w
            draw_h = draw_w / aspect
        c.drawImage(
            ImageReader(img_buf), x0, y0,
            width=draw_w, height=draw_h,
            preserveAspectRatio=True, anchor="sw",
        )


def build_xt_comparison_pdf() -> bytes:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas

    h_grid = compute_heuristic_xt_grid()
    sb_grid = compute_statsbomb_xt_grid()
    db_grid = compute_databallpy_xt_grid()

    panels = [
        (draw_xt_grid_map(h_grid, "xT Heurístico (0–1)", value_fmt=".2f")[0], "Heurístico"),
        (draw_xt_grid_map(sb_grid, "xT StatsBomb", value_fmt=".3f")[0], "StatsBomb"),
        (draw_xt_grid_map(db_grid, "xT DataBallPy", value_fmt=".3f")[0], "DataBallPy"),
        (
            draw_xt_diff_map(
                h_grid, sb_grid,
                "Δ Heurístico − StatsBomb (normalizados)",
            )[0],
            "Δ H − SB",
        ),
        (
            draw_xt_diff_map(
                h_grid, db_grid,
                "Δ Heurístico − DataBallPy (normalizados)",
            )[0],
            "Δ H − DB",
        ),
        (
            draw_xt_diff_map(
                sb_grid, db_grid,
                "Δ StatsBomb − DataBallPy (normalizados)",
            )[0],
            "Δ SB − DB",
        ),
    ]
    plt.close("all")

    buf = BytesIO()
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(buf, pagesize=landscape(A4))

    def _draw_page_header(subtitle: str):
        c.setFillColorRGB(0.1, 0.1, 0.16)
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
        c.setFillColorRGB(0.93, 0.95, 0.97)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(36, page_h - 36, "Comparativo de Modelos xT — Grade 16×12")
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.58, 0.64, 0.72)
        c.drawString(36, page_h - 52, subtitle)

    _draw_page_header(
        "Grades por modelo | StatsBomb 120×80 · DataBallPy convertido para 106×68 centrado",
    )
    _pdf_draw_image_row(
        c, panels[:3], page_w, page_h,
        y_top=page_h - 68, row_h=page_h - 100,
    )
    c.showPage()

    _draw_page_header(
        "Diferenças normalizadas (0–1) | Vermelho = modelo da esquerda maior na subtração",
    )
    _pdf_draw_image_row(
        c, panels[3:], page_w, page_h,
        y_top=page_h - 68, row_h=page_h - 100,
    )
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner="Gerando PDF comparativo xT…")
def get_xt_comparison_pdf_bytes() -> bytes:
    return build_xt_comparison_pdf()


def render_player_maps(df: pd.DataFrame, prog_model: str):
    img_pm, fig_pm = draw_pass_map(df, prog_model)
    plt.close(fig_pm)
    st.markdown('<div class="map-label">Pass Map</div>', unsafe_allow_html=True)
    st.image(img_pm, use_container_width=True)

    img_ht, fig_ht = draw_corridor_heatmap(df)
    plt.close(fig_ht)
    st.markdown('<div class="map-label">Zone Heatmap (Destination)</div>', unsafe_allow_html=True)
    st.image(img_ht, use_container_width=True)

    img_xt, fig_xt = draw_top_xt_map(df, top_n=5)
    plt.close(fig_xt)
    st.markdown('<div class="map-label">Top 5 Pass Impact</div>', unsafe_allow_html=True)
    st.image(img_xt, use_container_width=True)


def render_player_cards(stats: dict, tone: str, prog_model: str):
    progressive_items = [
        ("Progressive Passes", f"{stats['progressive_successful']:.0f}"),
        ("% Progressive Accuracy", f"{stats['progressive_accuracy_pct']:.1f}%"),
    ]
    if prog_model == PROG_MODEL_XT:
        progressive_items.insert(
            1,
            ("Highly Progressive", f"{stats['highly_progressive']:.0f}"),
        )
    stats_section_card(
        "Overview",
        tone,
        [
            ("Total Passes", f"{stats['total_passes']:.0f}"),
            ("% Accuracy", f"{stats['accuracy_pct']:.1f}%"),
        ],
    )
    stats_section_card("Progressive", tone, progressive_items)
    stats_section_card(
        "Impact",
        tone,
        [
            ("Pass Impact Value", f"{stats['sum_dxt']:.2f}"),
            ("% Positive Impact", f"{stats['pos_pct']:.1f}%"),
        ],
    )


# ── DATA LOAD ──────────────────────────────────────────────────
hudson_dfs, wc_dfs = load_all_pass_data()

if not hudson_dfs or BENTANCUR_KEY not in wc_dfs or BOUADDI_KEY not in wc_dfs:
    st.error(
        "Não foi possível carregar os dados. Verifique se os arquivos "
        f"'{HUDSON_DOCX}' e '{WORLD_CUP_DOCX}' estão no diretório do app."
    )
    st.stop()

hudson_match_names = list(hudson_dfs.keys())

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
      Bentancur e Bouaddi: jogos fixos da Copa do Mundo.<br>
      Hudson: selecione o jogo na área principal.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo de xT**")
xt_model = st.sidebar.radio(
    "Modelo de xT",
    options=list(XT_MODEL_LABELS.keys()),
    format_func=lambda k: XT_MODEL_LABELS[k],
    key="xt_model_selector",
    label_visibility="collapsed",
)
st.sidebar.caption(XT_MODEL_DESCRIPTIONS[xt_model])

st.sidebar.markdown("---")
st.sidebar.markdown("**Critério de Progressive Pass**")
prog_model = st.sidebar.radio(
    "Modelo de avaliação",
    options=list(PROG_MODEL_LABELS.keys()),
    format_func=lambda k: PROG_MODEL_LABELS[k],
    key="prog_model_selector",
    label_visibility="collapsed",
)
st.sidebar.caption(PROG_MODEL_DESCRIPTIONS[prog_model])

st.sidebar.markdown("---")
pdf_bytes = get_xt_comparison_pdf_bytes()
st.sidebar.download_button(
    label="Exportar PDF — Mapas xT",
    data=pdf_bytes,
    file_name="comparativo_xt_mapas.pdf",
    mime="application/pdf",
    use_container_width=True,
)
st.sidebar.caption("PDF: 3 grades (H / SB / DB) + 3 mapas de diferença normalizada.")

# ── MAIN LAYOUT ────────────────────────────────────────────────
st.markdown("## Passes — Comparação de Jogadores")
st.caption("Hudson Cicala vs Bentancur (vs Arábia Saudita) vs Bouaddi (vs Brasil)")

selected_hudson_match = st.selectbox(
    "Selecione o jogo de Hudson Cicala para comparar",
    options=hudson_match_names,
    index=0,
    key="hudson_match_selector",
)

hudson_df = prepare_player_df(hudson_dfs[selected_hudson_match], xt_model, prog_model)
bentancur_df = prepare_player_df(wc_dfs[BENTANCUR_KEY], xt_model, prog_model)
bouaddi_df = prepare_player_df(wc_dfs[BOUADDI_KEY], xt_model, prog_model)

hudson_stats = compute_stats(hudson_df, selected_hudson_match, prog_model, xt_model)
bentancur_stats = compute_stats(bentancur_df, BENTANCUR_KEY, prog_model, xt_model)
bouaddi_stats = compute_stats(bouaddi_df, BOUADDI_KEY, prog_model, xt_model)

players = [
    {
        "name": "Hudson Cicala",
        "subtitle": selected_hudson_match,
        "df": hudson_df,
        "stats": hudson_stats,
        "tone": PLAYER_TONES["Hudson Cicala"],
    },
    {
        "name": "Bentancur",
        "subtitle": "Copa do Mundo — vs Arábia Saudita",
        "df": bentancur_df,
        "stats": bentancur_stats,
        "tone": PLAYER_TONES["Bentancur"],
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
st.markdown(
    f"### Mapas de Passe — {PROG_MODEL_LABELS[prog_model]} · xT: {XT_MODEL_LABELS[xt_model]}"
)

map_cols = st.columns(3)
for col, player in zip(map_cols, players):
    with col:
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="player-sub">{player["subtitle"]}</div>', unsafe_allow_html=True)
        render_player_maps(player["df"], prog_model)

st.markdown("---")
st.markdown(
    f"### Estatísticas do jogo — {PROG_MODEL_LABELS[prog_model]} · xT: {XT_MODEL_LABELS[xt_model]}"
)

stat_cols = st.columns(3)
for col, player in zip(stat_cols, players):
    with col:
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="player-sub">{player["subtitle"]}</div>', unsafe_allow_html=True)
        render_player_cards(player["stats"], player["tone"], prog_model)

render_heuristic_v1_v2_comparison(
    hudson_dfs[selected_hudson_match],
    wc_dfs[BENTANCUR_KEY],
    wc_dfs[BOUADDI_KEY],
    prog_model,
    selected_hudson_match,
)
