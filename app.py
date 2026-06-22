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
    "Vitinha": "#d4a843",
}
CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NX_XT, NY_XT = 16, 12
DATABALLPY_PITCH_LENGTH = 106.0
DATABALLPY_PITCH_WIDTH = 68.0
LATERAL_MIN_DIST = 12.0
XT_MODEL_HEURISTIC = "heuristic"
XT_MODEL_HEURISTIC_V2 = "heuristic_v2"
XT_MODEL_HEURISTIC_V3 = "heuristic_v3"
XT_MODEL_HEURISTIC_V4 = "heuristic_v4"
XT_MODEL_HEURISTIC_V5 = "heuristic_v5"
XT_MODEL_STATSBOMB = "statsbomb"
XT_MODEL_DATABALLPY = "databallpy"
XT_MODEL_LABELS = {
    XT_MODEL_HEURISTIC: "Heurístico (v1)",
    XT_MODEL_HEURISTIC_V2: "Heurístico v2 (construção)",
    XT_MODEL_HEURISTIC_V3: "Heurístico v3 (zonas)",
    XT_MODEL_HEURISTIC_V4: "Heurístico v4 (v1 + xG)",
    XT_MODEL_HEURISTIC_V5: "Heurístico v5 (suave + coerente)",
    XT_MODEL_STATSBOMB: "StatsBomb (Karun Singh)",
    XT_MODEL_DATABALLPY: "DataBallPy (open play)",
}
XT_MODEL_DESCRIPTIONS = {
    XT_MODEL_HEURISTIC: "Grade sintética normalizada (0–1) por proximidade ao gol.",
    XT_MODEL_HEURISTIC_V2: "Zonas por terço, pico na grande área, progressivo ΔxT >0.15 / >0.35.",
    XT_MODEL_HEURISTIC_V3: "Terços monotônicos, escala por zona, alas com menor xT a partir dos 2/3.",
    XT_MODEL_HEURISTIC_V4: "v3 + centralidade v1 nos 2/3 + zona de finalização suave (lógica xG).",
    XT_MODEL_HEURISTIC_V5: "v5 + limiares calibrados (P75/P92 por zona, ref. StatsBomb).",
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
XT_V3_FINE_NX = 96
XT_V3_FINE_NY = 64
XT_V3_DISPLAY_SUB = 24
XT_V3_DEF_MAX = 0.25
XT_V3_MID_MAX = 0.60
XT_V3_PROG_SCALE = 0.15
XT_V3_HIGH_SCALE = 0.35
XT_V3_PROG_FLOOR = 0.08
XT_V3_HIGH_FLOOR = 0.18
XT_V3_NEG_PENALTY_FACTOR = 0.55
XT_V3_PRESSURE_ESCAPE_BONUS = 0.02
XT_V3_PRESSURE_X_MAX = 50.0
XT_V3_WIDE_FRAC = 0.60
XT_V3_NEG_RECYCLE_X_MAX = 60.0
XT_V3_MAX_PASS_DELTA = 0.45
XT_V3_SHORT_PASS_DIST = 5.0
XT_V3_SHORT_PASS_FACTOR = 0.7
XT_V3_LAT_DISC_MAX = 0.16
XT_V3_LAT_CURVE_POWER = 1.0
XT_V4_FINE_NX = 96
XT_V4_FINE_NY = 64
XT_V4_DISPLAY_SUB = 24
XT_V4_V1_WING_BASE = 0.80
XT_V4_V1_CENT_MULT = 1.00
XT_V4_BOX_X_START = 90.0
XT_V4_BOX_X_FULL = 112.0
XT_V4_CORNER_LAT_ON = 0.58
XT_V4_CORNER_PENALTY = 0.10
XT_V4_CENTRAL_PREMIUM = 0.06
XT_V4_SHORT_PASS_DIST = 8.0
XT_V4_SHORT_PASS_FACTOR = 0.55
XT_V5_FINE_NX = 96
XT_V5_FINE_NY = 64
XT_V5_DISPLAY_SUB = 24
XT_V5_ZONE_BLEND_WIDTH = 26.0
XT_V5_SURFACE_SMOOTH_SIGMA = 1.85
XT_V5_SHORT_PASS_DIST = 8.0
XT_V5_SHORT_PASS_FACTOR = 0.55
XT_V5_SHORT_PASS_BLEND = 4.0
XT_V5_MAX_DELTA_DEF = 0.28
XT_V5_MAX_DELTA_MID = 0.36
XT_V5_MAX_DELTA_ATT = 0.42
XT_V5_MAX_DELTA_BOX = 0.52
XT_V5_DELTA_CAP_BLEND = 12.0
XT_V5_CALIB_PROG_PCT = 75.0
XT_V5_CALIB_HIGH_PCT = 92.0

PASS_MAP_FILTER_ALL = "all"
PASS_MAP_FILTER_SUPER = "super"
PASS_MAP_FILTER_LABELS = {
    PASS_MAP_FILTER_ALL: "Todos os passes",
    PASS_MAP_FILTER_SUPER: "Somente super progressivos",
}

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
BENTANCUR_KEY = "Bentancur (vs Saudi Arabia)"
VITINHA_KEY = "Vitinha"
INVERTED_WC_PLAYERS = {BENTANCUR_KEY}
Y_FLIPPED_WC_PLAYERS = {VITINHA_KEY}
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

VITINHA_RAW_DATA = """
All Passes – Vitinha

Seta 1: (46.2, 26.4) -> (53.16, 36.8)
Seta 2: (41.64, 57.2) -> (39.72, 22.08)
Seta 3: (53.64, 67.68) -> (38.88, 47.28)
Seta 4: (40.8, 9.12) -> (45.12, 24.96)
Seta 5: (54.48, 40.56) -> (65.16, 12.56)
Seta 6: (53.88, 18.96) -> (72.36, 9.6)
Seta 7: (53.64, 42.56) -> (48.0, 55.68)
Seta 8: (76.32, 29.84) -> (74.4, 51.84)
Seta 9: (77.16, 55.2) -> (81.6, 55.92)
Seta 10: (75.36, 58.16) -> (85.08, 71.84)
Seta 11: (76.92, 66.72) -> (78.24, 54.32)
Seta 12: (76.08, 49.76) -> (80.28, 54.32)
Seta 13: (79.56, 48.8) -> (87.72, 60.96)
Seta 14: (79.92, 50.96) -> (107.52, 4.64)
Seta 15: (75.24, 57.36) -> (86.04, 78.32)
Seta 16: (79.92, 59.76) -> (97.32, 56.96)
Seta 17: (83.52, 59.6) -> (81.6, 49.92)
Seta 18: (84.72, 45.92) -> (98.28, 71.52)
Seta 19: (28.2, 7.44) -> (27.48, 2.24)
Seta 20: (77.28, 54.32) -> (90.0, 76.88)
Seta 21: (75.48, 66.0) -> (85.32, 63.6)
Seta 22: (72.48, 61.28) -> (52.32, 56.96)
Seta 23: (71.28, 73.76) -> (85.8, 77.52)
Seta 24: (67.68, 70.56) -> (74.64, 69.2)
Seta 25: (61.68, 67.68) -> (53.88, 75.68)
Seta 26: (63.72, 53.84) -> (51.36, 24.32)
Seta 27: (46.08, 70.08) -> (68.64, 77.76)
Seta 28: (49.56, 66.72) -> (37.56, 44.0)
Seta 29: (14.64, 55.44) -> (38.64, 58.88)
Seta 30: (20.04, 51.84) -> (33.0, 41.2)
Seta 31: (36.6, 36.72) -> (48.36, 49.44)
Seta 32: (54.84, 36.24) -> (56.4, 13.76)
Seta 33: (45.72, 27.6) -> (48.72, 70.32)
Seta 34: (48.12, 43.68) -> (48.6, 8.72)
Seta 35: (49.8, 39.36) -> (49.56, 56.64)
Seta 36: (81.48, 39.36) -> (93.96, 5.36)
Seta 37: (41.52, 15.68) -> (41.64, 5.12)
Seta 38: (27.48, 30.96) -> (43.2, 75.36)
Seta 39: (50.4, 43.84) -> (63.36, 15.6)
Seta 40: (73.8, 25.04) -> (76.08, 17.36)
Seta 41: (64.56, 44.72) -> (64.68, 19.04)
Seta 42: (49.56, 34.08) -> (49.08, 60.48)
Seta 43: (76.08, 48.48) -> (80.04, 41.6)
Seta 44: (76.2, 47.6) -> (79.92, 42.48)
Seta 45: (76.44, 44.96) -> (75.36, 14.0)
Seta 46: (72.48, 18.72) -> (67.44, 9.2)
Seta 47: (67.08, 60.8) -> (78.36, 77.36)
Seta 48: (72.24, 63.6) -> (81.48, 77.36)
Seta 49: (74.4, 49.04) -> (82.2, 67.68)
Seta 50: (74.16, 51.92) -> (81.0, 72.08)
Seta 51: (68.88, 52.32) -> (63.12, 21.84)
Seta 52: (45.36, 23.6) -> (55.44, 4.88)
Seta 53: (41.16, 32.0) -> (46.8, 70.56)
Seta 54: (39.96, 51.68) -> (42.6, 61.92)
Seta 55: (85.08, 64.8) -> (77.04, 44.48)
Seta 56: (86.04, 62.0) -> (73.68, 46.4)
Seta 57: (76.8, 51.36) -> (81.72, 76.56)
Seta 58: (74.76, 62.4) -> (64.08, 32.4)
Seta 59: (45.6, 67.92) -> (39.48, 74.4)
Seta 60: (35.88, 56.64) -> (10.68, 41.6)
Seta 61: (37.2, 66.0) -> (42.36, 58.16)
Seta 62: (36.96, 59.52) -> (43.68, 69.84)
Seta 63: (36.96, 21.44) -> (35.04, 2.72)
Seta 64: (35.16, 9.12) -> (35.28, 3.84)
Seta 65: (31.56, 43.76) -> (33.84, 18.0)
Seta 66: (46.2, 21.6) -> (50.88, 71.12)
Seta 67: (45.24, 48.56) -> (57.24, 31.44)
Seta 68: (63.12, 33.68) -> (72.72, 11.76)
Seta 69: (84.48, 8.0) -> (81.72, 17.04)
Seta 70: (80.88, 20.72) -> (79.68, 21.84)
Seta 71: (59.88, 46.32) -> (53.04, 33.68)
Seta 72: (32.76, 50.48) -> (41.76, 46.8)
Seta 73: (36.6, 48.56) -> (45.36, 43.76)
Seta 74: (44.52, 49.04) -> (50.04, 73.76)
Seta 75: (44.52, 53.36) -> (39.84, 34.16)
Seta 76: (54.84, 71.84) -> (67.08, 77.28)
Seta 77: (46.8, 63.68) -> (34.56, 37.68)
Seta 78: (75.36, 65.12) -> (68.52, 50.88)
Seta 79: (69.12, 56.16) -> (69.36, 76.32)
Seta 80: (54.6, 68.4) -> (35.4, 48.56)
Seta 81: (50.04, 68.72) -> (45.6, 69.68)
Seta 82: (49.2, 66.8) -> (44.4, 47.12)
Seta 83: (42.6, 32.4) -> (45.24, 62.4)
Seta 84: (40.08, 48.8) -> (53.76, 31.04)
Seta 85: (28.92, 38.16) -> (5.76, 38.0)
Seta 86: (11.76, 15.04) -> (3.6, 37.68)
Seta 87: (34.56, 38.24) -> (40.92, 9.6)
Seta 88: (44.16, 22.56) -> (28.56, 41.52)
Seta 89: (54.12, 54.0) -> (50.04, 73.28)
Seta 90: (76.32, 39.6) -> (82.32, 44.72)
Seta 91: (72.84, 19.68) -> (75.48, 29.36)
Seta 92: (50.64, 59.6) -> (57.72, 77.76)
Seta 93: (45.96, 68.88) -> (35.28, 45.6)
Seta 94: (39.96, 71.12) -> (58.56, 77.6)
Seta 95: (40.32, 69.44) -> (35.04, 14.64)
Seta 96: (64.2, 55.28) -> (101.4, 12.96)
Seta 97: (72.84, 60.32) -> (64.56, 42.72)
Seta 98: (75.24, 30.08) -> (83.52, 77.28)
Seta 99: (64.8, 49.2) -> (68.04, 14.4)
Seta 100: (47.04, 31.28) -> (87.6, 20.64)
Seta 101: (21.6, 40.16) -> (36.36, 5.52)
Seta 102: (45.12, 11.76) -> (38.4, 32.16)
Seta 103: (74.4, 48.48) -> (82.2, 12.72)
Seta 104: (81.6, 18.24) -> (109.56, 52.64)
Seta 105: (82.32, 60.8) -> (103.44, 66.0)
Seta 106: (45.72, 51.36) -> (48.96, 39.92)
Seta 107: (78.72, 40.56) -> (78.96, 13.68)
Seta 108: (13.68, 55.52) -> (51.84, 53.28)
Seta 109: (18.96, 39.68) -> (54.12, 77.84)
Seta 110: (43.44, 41.28) -> (43.68, 9.68)
Seta 111: (37.56, 34.56) -> (45.6, 18.32)
Seta 112: (54.36, 34.8) -> (54.0, 64.88)
Seta 113: (65.16, 56.24) -> (65.04, 74.48)
Seta 114: (36.6, 56.16) -> (35.64, 42.32)
Seta 115: (62.4, 27.44) -> (64.92, 52.8)
Seta 116: (77.64, 35.76) -> (88.92, 4.16)
Seta 117: (81.6, 23.52) -> (96.48, 7.52)
Seta 118: (74.16, 53.04) -> (79.56, 18.72)
Seta 119: (55.32, 40.4) -> (64.2, 31.28)
Seta 120: (81.48, 47.28) -> (75.6, 68.16)
Seta 121: (63.48, 58.08) -> (56.76, 29.52)
Seta 122: (65.64, 22.56) -> (79.8, 6.8)
Seta 123: (69.24, 17.12) -> (87.72, 4.32)
Seta 124: (87.24, 33.68) -> (93.48, 37.2)
Seta 125: (17.04, 42.48) -> (39.96, 36.32)
Seta 126: (75.0, 20.72) -> (72.36, 59.6)
Seta 127: (71.64, 51.6) -> (78.36, 72.24)
Seta 128: (73.56, 59.76) -> (90.24, 75.68)
Seta 129: (42.36, 35.52) -> (47.88, 13.68)

Passes Errados
Seta 1: (76.08, 49.76) -> (80.28, 54.32)
Seta 2: (80.88, 20.72) -> (79.68, 21.84)
Seta 3: (50.04, 68.72) -> (45.6, 69.68)
Seta 4: (40.08, 48.8) -> (53.76, 31.04)
Seta 5: (76.32, 39.6) -> (82.32, 44.72)
Seta 6: (47.04, 31.28) -> (87.6, 20.64)
Seta 7: (81.6, 18.24) -> (109.56, 52.64)
Seta 8: (13.68, 55.52) -> (51.84, 53.28)
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


def classify_xt_progressive_v3(
    xt_start: float,
    xt_end: float,
    x_end: float,
    pass_distance: float,
) -> str:
    """v3/v4: limiar de ΔxT relativo à posição no campo."""
    if pass_distance <= XT_MIN_PASS_DISTANCE:
        return "none"
    delta_abs = xt_end - xt_start
    prog_thresh = max(XT_V3_PROG_FLOOR, XT_V3_PROG_SCALE * (1.0 - xt_start))
    high_thresh = max(XT_V3_HIGH_FLOOR, XT_V3_HIGH_SCALE * (1.0 - xt_start))
    if delta_abs <= prog_thresh:
        return "none"
    if delta_abs > high_thresh:
        return "highly"
    return "progressive"


def classify_xt_progressive_v4(
    xt_start: float,
    xt_end: float,
    x_end: float,
    pass_distance: float,
) -> str:
    return classify_xt_progressive_v3(xt_start, xt_end, x_end, pass_distance)


def classify_xt_progressive_v5(
    xt_start: float,
    delta_xt: float,
    x_end: float,
    pass_distance: float,
    x_start: float | None = None,
) -> str:
    """v5: adjusted delta_xt + zone thresholds calibrated on StatsBomb reference passes."""
    if pass_distance <= XT_MIN_PASS_DISTANCE:
        return "none"
    if delta_xt <= 0:
        return "none"
    prog_thresh, high_thresh = _v5_zone_thresholds(x_start if x_start is not None else xt_start * FIELD_X)
    if delta_xt <= prog_thresh:
        return "none"
    if delta_xt > high_thresh:
        return "highly"
    return "progressive"


def _x_zone_bucket(x_start: float) -> str:
    x = float(x_start)
    if x >= XT_V4_BOX_X_START:
        return "box"
    if x >= FINAL_THIRD_LINE_X:
        return "att"
    if x >= OPT_ATTACKING_TWO_THIRDS_X:
        return "mid"
    return "def"


@st.cache_data(show_spinner=False)
def compute_v5_reference_thresholds() -> dict[str, dict[str, float]]:
    """P75/P92 of positive v5 ΔxT by pitch zone from StatsBomb FA WSL completed passes."""
    parser = Sbopen()
    fine = compute_heuristic_v5_fine_grid()
    deltas: dict[str, list[float]] = {z: [] for z in ("def", "mid", "att", "box")}
    df_match = parser.match(competition_id=37, season_id=42)
    for match_id in df_match.match_id.unique():
        event = parser.event(match_id)[0]
        passes = event[(event["type_name"] == "Pass") & (event["outcome_name"].isnull())]
        for _, row in passes.iterrows():
            dist = float(np.hypot(row["end_x"] - row["x"], row["end_y"] - row["y"]))
            if dist <= XT_MIN_PASS_DISTANCE:
                continue
            xs = xt_value_bilinear(float(row["x"]), float(row["y"]), fine)
            xe = xt_value_bilinear(float(row["end_x"]), float(row["end_y"]), fine)
            delta = xe - xs
            if delta <= 0:
                continue
            deltas[_x_zone_bucket(float(row["x"]))].append(float(delta))
    thresholds: dict[str, dict[str, float]] = {}
    for zone, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        if arr.size < 50:
            thresholds[zone] = {
                "prog": max(XT_V3_PROG_FLOOR, XT_V3_PROG_SCALE * 0.6),
                "high": max(XT_V3_HIGH_FLOOR, XT_V3_HIGH_SCALE * 0.6),
            }
        else:
            thresholds[zone] = {
                "prog": float(np.percentile(arr, XT_V5_CALIB_PROG_PCT)),
                "high": float(np.percentile(arr, XT_V5_CALIB_HIGH_PCT)),
            }
    return thresholds


def _v5_zone_thresholds(x_start: float) -> tuple[float, float]:
    th = compute_v5_reference_thresholds()[_x_zone_bucket(x_start)]
    return th["prog"], th["high"]


def _xt_progressive_category(
    xt_start: float,
    xt_end: float,
    x_start: float,
    x_end: float,
    pass_distance: float,
    xt_model: str,
    delta_xt: float,
) -> str:
    if xt_model == XT_MODEL_HEURISTIC_V5:
        return classify_xt_progressive_v5(xt_start, delta_xt, x_end, pass_distance, x_start=x_start)
    return classify_xt_progressive_for_model(
        xt_start, xt_end, x_end, pass_distance, xt_model, delta_xt=delta_xt,
    )


def classify_xt_progressive_for_model(
    xt_start: float,
    xt_end: float,
    x_end: float,
    pass_distance: float,
    xt_model: str,
    delta_xt: float | None = None,
    x_start: float | None = None,
) -> str:
    if xt_model == XT_MODEL_HEURISTIC_V2:
        return classify_xt_progressive_v2(xt_start, xt_end, x_end, pass_distance)
    if xt_model == XT_MODEL_HEURISTIC_V5:
        use_delta = delta_xt if delta_xt is not None else xt_end - xt_start
        return classify_xt_progressive_v5(
            xt_start, use_delta, x_end, pass_distance,
            x_start=x_start if x_start is not None else xt_start * FIELD_X,
        )
    if xt_model in (XT_MODEL_HEURISTIC_V3, XT_MODEL_HEURISTIC_V4):
        return classify_xt_progressive_v3(xt_start, xt_end, x_end, pass_distance)
    return classify_xt_progressive(xt_start, xt_end, x_end, pass_distance)


def is_progressive_xt_attempt(xt_start: float, xt_end: float, x_end: float, pass_distance: float) -> bool:
    return classify_xt_progressive(xt_start, xt_end, x_end, pass_distance) in ("progressive", "highly")


def is_progressive_xt_attempt_for_model(
    xt_start: float, xt_end: float, x_end: float, pass_distance: float, xt_model: str,
    delta_xt: float | None = None,
    x_start: float | None = None,
) -> bool:
    return classify_xt_progressive_for_model(
        xt_start, xt_end, x_end, pass_distance, xt_model,
        delta_xt=delta_xt, x_start=x_start,
    ) in ("progressive", "highly")


def is_progressive_attempt(row, model: str, xt_model: str = XT_MODEL_HEURISTIC) -> bool:
    if model == PROG_MODEL_WYSCOUT:
        return is_progressive_wyscout(row.x_start, row.y_start, row.x_end, row.y_end)
    if model == PROG_MODEL_OPTA:
        return is_progressive_opta(row.x_start, row.y_start, row.x_end, row.y_end)
    raw_delta = getattr(row, "raw_delta_xt", row.xt_end - row.xt_start)
    return is_progressive_xt_attempt_for_model(
        row.xt_start, row.xt_end, row.x_end, row.pass_distance, xt_model,
        delta_xt=raw_delta, x_start=row.x_start,
    )


def _spatial_progressive_attempt(x_start, y_start, x_end, y_end, model: str) -> bool:
    if model == PROG_MODEL_WYSCOUT:
        return is_progressive_wyscout(x_start, y_start, x_end, y_end)
    if model == PROG_MODEL_OPTA:
        return is_progressive_opta(x_start, y_start, x_end, y_end)
    return False


def apply_progressive_model(
    df: pd.DataFrame, model: str, xt_model: str = XT_MODEL_HEURISTIC,
) -> pd.DataFrame:
    df = df.copy()
    if "raw_delta_xt" not in df.columns:
        df["raw_delta_xt"] = df["xt_end"] - df["xt_start"]
    progressive_attempt_flags = []
    progressive_flags = []
    highly_xt_flags = []
    for row in df.itertuples(index=False):
        if model in (PROG_MODEL_WYSCOUT, PROG_MODEL_OPTA):
            attempt = _spatial_progressive_attempt(
                row.x_start, row.y_start, row.x_end, row.y_end, model,
            )
        else:
            attempt = is_progressive_xt_attempt_for_model(
                row.xt_start, row.xt_end, row.x_end, row.pass_distance, xt_model,
                delta_xt=row.raw_delta_xt, x_start=row.x_start,
            )
        success_delta = row.delta_xt if xt_model == XT_MODEL_HEURISTIC_V5 else row.raw_delta_xt
        xt_cat_success = _xt_progressive_category(
            row.xt_start, row.xt_end, row.x_start, row.x_end, row.pass_distance,
            xt_model, success_delta,
        )
        progressive_attempt_flags.append(attempt)
        if model in (PROG_MODEL_WYSCOUT, PROG_MODEL_OPTA):
            progressive_flags.append(row.is_won and attempt)
        else:
            progressive_flags.append(row.is_won and xt_cat_success in ("progressive", "highly"))
        highly_xt_flags.append(row.is_won and xt_cat_success == "highly")
    df["progressive_attempt"] = progressive_attempt_flags
    df["progressive"] = progressive_flags
    df["highly_progressive"] = highly_xt_flags
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


def _attach_xt_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["raw_delta_xt"] = out["xt_end"] - out["xt_start"]
    return out


def apply_heuristic_v2_xt(df: pd.DataFrame) -> pd.DataFrame:
    fine = compute_heuristic_v2_fine_grid()
    out = df.copy()
    out["xt_start"] = out.apply(lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1)
    out["xt_end"] = out.apply(lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1)
    out["delta_xt"] = out.apply(_adjust_heuristic_v2_pass_delta, axis=1)
    return out


def _zone_x_threat_v3_raw(x: np.ndarray) -> np.ndarray:
    """Per-third 0→1 profile; monotonic in the attacking third."""
    x = np.clip(x, 0.0, FIELD_X)
    threat = np.zeros_like(x, dtype=float)
    def_mask = x < OPT_ATTACKING_TWO_THIRDS_X
    mid_mask = (x >= OPT_ATTACKING_TWO_THIRDS_X) & (x < FINAL_THIRD_LINE_X)
    att_mask = x >= FINAL_THIRD_LINE_X

    threat[def_mask] = x[def_mask] / OPT_ATTACKING_TWO_THIRDS_X
    mid_t = (x[mid_mask] - OPT_ATTACKING_TWO_THIRDS_X) / (
        FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X
    )
    threat[mid_mask] = _smoothstep(mid_t)
    att_t = (x[att_mask] - FINAL_THIRD_LINE_X) / (FIELD_X - FINAL_THIRD_LINE_X)
    threat[att_mask] = _smoothstep(att_t)
    return threat


def _map_zonal_threat(raw: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map each third to fixed absolute ranges: def 0–0.25, mid 0.25–0.60, att 0.60–1.0."""
    result = np.zeros_like(raw, dtype=float)
    def_mask = x < OPT_ATTACKING_TWO_THIRDS_X
    mid_mask = (x >= OPT_ATTACKING_TWO_THIRDS_X) & (x < FINAL_THIRD_LINE_X)
    att_mask = x >= FINAL_THIRD_LINE_X
    result[def_mask] = XT_V3_DEF_MAX * raw[def_mask]
    result[mid_mask] = XT_V3_DEF_MAX + (XT_V3_MID_MAX - XT_V3_DEF_MAX) * raw[mid_mask]
    result[att_mask] = XT_V3_MID_MAX + (1.0 - XT_V3_MID_MAX) * raw[att_mask]
    return result


def _lateral_relative_position(y: np.ndarray) -> np.ndarray:
    """0 at the center line, 1 at the touchline."""
    return np.abs(y - GOAL_Y) / (FIELD_Y / 2.0)


def _location_factor_v3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lateral discount from attacking 2/3 (x≥40): wings lower than center, smooth & symmetric."""
    lat = _lateral_relative_position(y)
    depth = np.clip(
        (x - OPT_ATTACKING_TWO_THIRDS_X) / (FIELD_X - OPT_ATTACKING_TWO_THIRDS_X),
        0.0, 1.0,
    )
    zone_gate = _smoothstep(depth)
    max_discount = XT_V3_LAT_DISC_MAX * zone_gate
    lateral_curve = _smoothstep(lat ** XT_V3_LAT_CURVE_POWER)
    return 1.0 - max_discount * lateral_curve


def _build_heuristic_v3_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    raw = _zone_x_threat_v3_raw(Xc)
    return _map_zonal_threat(raw, Xc) * _location_factor_v3(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v3_fine_grid(nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY) -> np.ndarray:
    """High-resolution lookup grid with zone-based absolute scale (no global min–max)."""
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v3_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v3_xt_grid(NX: int = NX_XT, NY: int = NY_XT, sub: int = XT_V3_DISPLAY_SUB) -> np.ndarray:
    """16×12 display grid via sub-cell averaging (same method as v1)."""
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    threat = _build_heuristic_v3_threat_surface(Xc, Yc)
    grid = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            grid[iy, ix] = threat[iy * sub:(iy + 1) * sub, ix * sub:(ix + 1) * sub].mean()
    return grid


def _adjust_heuristic_v3_pass_delta(row) -> float:
    """v3: capped positive deltas, zone-relative recycle penalty, short-pass discount."""
    if not row.is_won:
        return 0.0
    raw = float(row.xt_end - row.xt_start)
    if raw >= 0:
        adjusted = raw
        if row.pass_distance < XT_V3_SHORT_PASS_DIST:
            adjusted *= XT_V3_SHORT_PASS_FACTOR
        return min(adjusted, XT_V3_MAX_PASS_DELTA)
    lat_start = _lateral_frac(row.y_start)
    lat_end = _lateral_frac(row.y_end)
    if row.x_start < XT_V3_NEG_RECYCLE_X_MAX:
        adjusted = raw * (XT_V3_NEG_PENALTY_FACTOR if lat_end < lat_start else 1.0)
    else:
        adjusted = raw
    if (
        row.x_start < XT_V3_PRESSURE_X_MAX
        and lat_start > XT_V3_WIDE_FRAC
        and lat_end < lat_start - 0.12
    ):
        adjusted += XT_V3_PRESSURE_ESCAPE_BONUS
    return adjusted


def apply_heuristic_v3_xt(df: pd.DataFrame) -> pd.DataFrame:
    fine = compute_heuristic_v3_fine_grid()
    out = df.copy()
    out["xt_start"] = out.apply(lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1)
    out["xt_end"] = out.apply(lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1)
    out["delta_xt"] = out.apply(_adjust_heuristic_v3_pass_delta, axis=1)
    return out


def _v4_v1_lateral_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """v1-style column weight (0.8 wing / 1.0 center), active only from attacking 2/3 (x≥40)."""
    cent = _centrality(y)
    v1_weight = XT_V4_V1_WING_BASE + (XT_V4_V1_CENT_MULT - XT_V4_V1_WING_BASE) * cent
    depth_2_3 = np.clip(
        (x - OPT_ATTACKING_TWO_THIRDS_X) / (FIELD_X - OPT_ATTACKING_TWO_THIRDS_X),
        0.0, 1.0,
    )
    gate = _smoothstep(depth_2_3)
    return 1.0 + gate * (v1_weight - 1.0)


def _v4_box_gate(x: np.ndarray) -> np.ndarray:
    """Smooth ramp into the penalty area (x 90→112), avoids abrupt xT jumps."""
    span = max(XT_V4_BOX_X_FULL - XT_V4_BOX_X_START, 1.0)
    t = np.clip((x - XT_V4_BOX_X_START) / span, 0.0, 1.0)
    return _smoothstep(t)


def _v4_xg_finishing_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gentle xG-style modifier: central premium in the box, mild wide discount — no corner cliff."""
    box_gate = _v4_box_gate(x)
    cent = _centrality(y)
    lat = _lateral_relative_position(y)
    central_bonus = XT_V4_CENTRAL_PREMIUM * box_gate * _smoothstep(cent)
    wide_in_box = box_gate * _smoothstep(np.clip((lat - XT_V4_CORNER_LAT_ON) / 0.42, 0.0, 1.0))
    wide_discount = XT_V4_CORNER_PENALTY * wide_in_box
    return np.clip(1.0 + central_bonus - wide_discount, 0.94, 1.06)


def _enforce_row_monotonic_x(grid: np.ndarray) -> np.ndarray:
    """Ensure xT never decreases toward the opponent goal within each pitch row."""
    out = grid.copy()
    for iy in range(out.shape[0]):
        for ix in range(1, out.shape[1]):
            if out[iy, ix] < out[iy, ix - 1]:
                out[iy, ix] = out[iy, ix - 1]
    return out


def _build_heuristic_v4_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat(_zone_x_threat_v3_raw(Xc), Xc)
    surface = zonal * _v4_v1_lateral_factor(Xc, Yc) * _v4_xg_finishing_factor(Xc, Yc)
    surface = np.clip(surface, 0.0, 1.0)
    return _enforce_row_monotonic_x(surface)


@st.cache_data(show_spinner=False)
def compute_heuristic_v4_fine_grid(nx: int = XT_V4_FINE_NX, ny: int = XT_V4_FINE_NY) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v4_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v4_xt_grid(NX: int = NX_XT, NY: int = NY_XT, sub: int = XT_V4_DISPLAY_SUB) -> np.ndarray:
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    threat = _build_heuristic_v4_threat_surface(Xc, Yc)
    grid = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            grid[iy, ix] = threat[iy * sub:(iy + 1) * sub, ix * sub:(ix + 1) * sub].mean()
    return _enforce_row_monotonic_x(grid)


def _adjust_heuristic_v4_pass_delta(row) -> float:
    """v4: v3 rules with stricter discount on short passes near the box."""
    if not row.is_won:
        return 0.0
    raw = float(row.xt_end - row.xt_start)
    if raw >= 0:
        adjusted = raw
        short_dist = XT_V4_SHORT_PASS_DIST
        short_factor = XT_V4_SHORT_PASS_FACTOR
        if row.pass_distance < short_dist:
            adjusted *= short_factor
        elif row.pass_distance < short_dist + 4.0:
            blend = (row.pass_distance - short_dist) / 4.0
            adjusted *= short_factor + (1.0 - short_factor) * blend
        return min(adjusted, XT_V3_MAX_PASS_DELTA)
    lat_start = _lateral_frac(row.y_start)
    lat_end = _lateral_frac(row.y_end)
    if row.x_start < XT_V3_NEG_RECYCLE_X_MAX:
        adjusted = raw * (XT_V3_NEG_PENALTY_FACTOR if lat_end < lat_start else 1.0)
    else:
        adjusted = raw
    if (
        row.x_start < XT_V3_PRESSURE_X_MAX
        and lat_start > XT_V3_WIDE_FRAC
        and lat_end < lat_start - 0.12
    ):
        adjusted += XT_V3_PRESSURE_ESCAPE_BONUS
    return adjusted


def apply_heuristic_v4_xt(df: pd.DataFrame) -> pd.DataFrame:
    fine = compute_heuristic_v4_fine_grid()
    out = df.copy()
    out["xt_start"] = out.apply(lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1)
    out["xt_end"] = out.apply(lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1)
    out["delta_xt"] = out.apply(_adjust_heuristic_v4_pass_delta, axis=1)
    return out


def _smoothstep_scalar(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    """Perlin smootherstep — gentler than smoothstep for zone transitions."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _map_zonal_threat_v5_smooth(x: np.ndarray) -> np.ndarray:
    """Blend zone scales with soft weights — no abrupt jumps at x=40 / x=80."""
    blend = XT_V5_ZONE_BLEND_WIDTH
    def_raw = np.clip(x / OPT_ATTACKING_TWO_THIRDS_X, 0.0, 1.0)
    mid_raw = np.clip(
        (x - OPT_ATTACKING_TWO_THIRDS_X) / (FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X),
        0.0, 1.0,
    )
    att_raw = np.clip(
        (x - FINAL_THIRD_LINE_X) / (FIELD_X - FINAL_THIRD_LINE_X),
        0.0, 1.0,
    )
    threat_def = XT_V3_DEF_MAX * _smootherstep(def_raw)
    threat_mid = XT_V3_DEF_MAX + (XT_V3_MID_MAX - XT_V3_DEF_MAX) * _smootherstep(mid_raw)
    threat_att = XT_V3_MID_MAX + (1.0 - XT_V3_MID_MAX) * _smootherstep(att_raw)

    w_def = 1.0 - _smootherstep(np.clip((x - (OPT_ATTACKING_TWO_THIRDS_X - blend)) / blend, 0.0, 1.0))
    w_att = _smootherstep(np.clip((x - (FINAL_THIRD_LINE_X - blend)) / blend, 0.0, 1.0))
    w_mid = np.clip(1.0 - w_def - w_att, 0.0, 1.0)
    w_sum = w_def + w_mid + w_att + 1e-12
    return (w_def * threat_def + w_mid * threat_mid + w_att * threat_att) / w_sum


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _smooth_surface_2d(surface: np.ndarray, sigma: float) -> np.ndarray:
    """Light separable Gaussian blur to reduce adjacent-cell xT jumps."""
    if sigma <= 0:
        return surface
    kernel = _gaussian_kernel_1d(sigma)
    pad = len(kernel) // 2
    padded = np.pad(surface, pad, mode="edge")
    along_x = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    padded_y = np.pad(along_x, pad, mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, padded_y)


def _build_heuristic_v5_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat_v5_smooth(Xc)
    surface = zonal * _v4_v1_lateral_factor(Xc, Yc) * _v4_xg_finishing_factor(Xc, Yc)
    surface = np.clip(surface, 0.0, 1.0)
    surface = _smooth_surface_2d(surface, XT_V5_SURFACE_SMOOTH_SIGMA)
    surface = np.clip(surface, 0.0, 1.0)
    return _enforce_row_monotonic_x(surface)


@st.cache_data(show_spinner=False)
def compute_heuristic_v5_fine_grid(nx: int = XT_V5_FINE_NX, ny: int = XT_V5_FINE_NY) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v5_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v5_xt_grid(NX: int = NX_XT, NY: int = NY_XT, sub: int = XT_V5_DISPLAY_SUB) -> np.ndarray:
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    threat = _build_heuristic_v5_threat_surface(Xc, Yc)
    grid = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            grid[iy, ix] = threat[iy * sub:(iy + 1) * sub, ix * sub:(ix + 1) * sub].mean()
    return _enforce_row_monotonic_x(grid)


def _v5_short_pass_multiplier(pass_distance: float) -> float:
    short_dist = XT_V5_SHORT_PASS_DIST
    short_factor = XT_V5_SHORT_PASS_FACTOR
    blend_span = XT_V5_SHORT_PASS_BLEND
    if pass_distance < short_dist:
        return short_factor
    if pass_distance < short_dist + blend_span:
        blend = (pass_distance - short_dist) / blend_span
        return short_factor + (1.0 - short_factor) * blend
    return 1.0


def _v5_zone_max_pass_delta(x_start: float) -> float:
    """Smooth zone-dependent cap: tighter far from goal, looser in the box."""
    x = float(np.clip(x_start, 0.0, FIELD_X))
    control_points = [
        (0.0, XT_V5_MAX_DELTA_DEF),
        (OPT_ATTACKING_TWO_THIRDS_X, XT_V5_MAX_DELTA_MID),
        (FINAL_THIRD_LINE_X, XT_V5_MAX_DELTA_ATT),
        (XT_V4_BOX_X_START, XT_V5_MAX_DELTA_BOX),
        (FIELD_X, XT_V5_MAX_DELTA_BOX),
    ]
    for idx in range(len(control_points) - 1):
        x0, cap0 = control_points[idx]
        x1, cap1 = control_points[idx + 1]
        if x <= x1:
            if x1 <= x0:
                return cap1
            t = _smoothstep_scalar((x - x0) / (x1 - x0))
            return cap0 + (cap1 - cap0) * t
    return control_points[-1][1]


def _adjust_heuristic_v5_pass_delta(row) -> float:
    """v5: short-pass discount + zone-based positive delta cap; same recycle rules as v4."""
    if not row.is_won:
        return 0.0
    raw = float(row.xt_end - row.xt_start)
    if raw >= 0:
        adjusted = raw * _v5_short_pass_multiplier(row.pass_distance)
        return min(adjusted, _v5_zone_max_pass_delta(row.x_start))
    lat_start = _lateral_frac(row.y_start)
    lat_end = _lateral_frac(row.y_end)
    if row.x_start < XT_V3_NEG_RECYCLE_X_MAX:
        adjusted = raw * (XT_V3_NEG_PENALTY_FACTOR if lat_end < lat_start else 1.0)
    else:
        adjusted = raw
    if (
        row.x_start < XT_V3_PRESSURE_X_MAX
        and lat_start > XT_V3_WIDE_FRAC
        and lat_end < lat_start - 0.12
    ):
        adjusted += XT_V3_PRESSURE_ESCAPE_BONUS
    return adjusted


def apply_heuristic_v5_xt(df: pd.DataFrame) -> pd.DataFrame:
    fine = compute_heuristic_v5_fine_grid()
    out = df.copy()
    out["xt_start"] = out.apply(lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1)
    out["xt_end"] = out.apply(lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1)
    out["delta_xt"] = out.apply(_adjust_heuristic_v5_pass_delta, axis=1)
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
    if xt_model == XT_MODEL_HEURISTIC_V3:
        return compute_heuristic_v3_xt_grid()
    if xt_model == XT_MODEL_HEURISTIC_V4:
        return compute_heuristic_v4_xt_grid()
    if xt_model == XT_MODEL_HEURISTIC_V5:
        return compute_heuristic_v5_xt_grid()
    return compute_heuristic_xt_grid()


def apply_xt_model(df: pd.DataFrame, xt_model: str) -> pd.DataFrame:
    if xt_model == XT_MODEL_HEURISTIC_V2:
        out = apply_heuristic_v2_xt(df)
    elif xt_model == XT_MODEL_HEURISTIC_V3:
        out = apply_heuristic_v3_xt(df)
    elif xt_model == XT_MODEL_HEURISTIC_V4:
        out = apply_heuristic_v4_xt(df)
    elif xt_model == XT_MODEL_HEURISTIC_V5:
        out = apply_heuristic_v5_xt(df)
    elif xt_model == XT_MODEL_DATABALLPY:
        out = df.copy()
        out["xt_start"] = out.apply(lambda r: databallpy_xt_value(r["x_start"], r["y_start"]), axis=1)
        out["xt_end"] = out.apply(lambda r: databallpy_xt_value(r["x_end"], r["y_end"]), axis=1)
        out["delta_xt"] = np.where(out["is_won"], out["xt_end"] - out["xt_start"], 0.0)
    else:
        out = apply_xt_grid(df, get_xt_grid(xt_model))
    return _attach_xt_columns(out)


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


def flip_pitch_y(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Mirror coordinates across the pitch center line (swap left/right wings)."""
    return x1, FIELD_Y - y1, x2, FIELD_Y - y2


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
    if player in Y_FLIPPED_WC_PLAYERS:
        x1, y1, x2, y2 = flip_pitch_y(x1, y1, x2, y2)
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
    dfm["progressive_attempt"] = False
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
    base = apply_xt_model(df, xt_model)
    if xt_model == XT_MODEL_HEURISTIC_V2:
        base["delta_xt_v2"] = base["delta_xt"]
    else:
        base["delta_xt_v2"] = apply_heuristic_v2_xt(df)["delta_xt"]
    return apply_progressive_model(base, prog_model, xt_model)


def load_all_pass_data() -> tuple[dict, dict]:
    hudson_path = Path(HUDSON_DOCX)
    if not hudson_path.exists():
        return {}, {}

    hudson_raw = parse_hudson_docx(read_docx_text(hudson_path))
    wc_raw: dict[str, list] = {}

    bentancur_parsed = parse_player_passes_text(BENTANCUR_RAW_DATA)
    if BENTANCUR_KEY in bentancur_parsed:
        wc_raw[BENTANCUR_KEY] = bentancur_parsed[BENTANCUR_KEY]

    vitinha_parsed = parse_player_passes_text(VITINHA_RAW_DATA)
    if VITINHA_KEY in vitinha_parsed:
        wc_raw[VITINHA_KEY] = vitinha_parsed[VITINHA_KEY]

    hudson_dfs = {name: events_to_dataframe(events, name) for name, events in hudson_raw.items()}
    wc_dfs = {name: events_to_dataframe(events, name) for name, events in wc_raw.items()}
    return hudson_dfs, wc_dfs


def compute_stats(
    df: pd.DataFrame,
    match_name: str,
    prog_model: str = PROG_MODEL_WYSCOUT,
    xt_model: str = XT_MODEL_HEURISTIC,
) -> dict:
    if "progressive_attempt" not in df.columns:
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
            "highly_progressive_pct": 0.0,
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
            "sum_dxt_v2": 0.0,
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
    highly_progressive_pct = (highly_progressive / successful * 100.0) if successful else 0.0
    progressive_attempted = int(df["progressive_attempt"].sum())
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
    sum_dxt_v2 = float(df.loc[df["is_won"], "delta_xt_v2"].sum()) if "delta_xt_v2" in df.columns else 0.0
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
        "highly_progressive_pct": round(highly_progressive_pct, 1),
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
        "sum_dxt_v2": round(sum_dxt_v2, 3),
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
    ax.axvline(x=OPT_ATTACKING_TWO_THIRDS_X, color="#ffffff", lw=0.9, alpha=0.22, linestyle="--")
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
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completado", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressivo", alpha=0.90),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=2.0, label="Super Progressivo", alpha=0.95),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incompleto", alpha=0.90),
    ]
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


def draw_xt_grid_map(
    grid: np.ndarray,
    title: str,
    value_fmt: str = ".2f",
    as_percent: bool = False,
    color_percentile: tuple[float, float] | None = None,
):
    fig, ax, pitch = _base_pitch()
    x_bins = np.linspace(0, FIELD_X, NX_XT + 1)
    y_bins = np.linspace(0, FIELD_Y, NY_XT + 1)
    if color_percentile is not None:
        vmin = float(np.percentile(grid, color_percentile[0]))
        vmax = float(np.percentile(grid, color_percentile[1]))
    else:
        vmin = 0.0
        vmax = max(float(grid.max()), 1e-6)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    cmap = LinearSegmentedColormap.from_list("xt", ["#1a1a2e", "#3b82f6", "#fbbf24", "#ef4444"])
    norm = Normalize(vmin=vmin, vmax=vmax)
    threshold = vmin + (vmax - vmin) * 0.45
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


def draw_heuristic_impact_chart(player_rows: list[dict]):
    names = [p["name"] for p in player_rows]
    v1_vals = [p["v1_impact"] for p in player_rows]
    v2_vals = [p["v2_impact"] for p in player_rows]
    v3_vals = [p["v3_impact"] for p in player_rows]
    v4_vals = [p["v4_impact"] for p in player_rows]
    v5_vals = [p["v5_impact"] for p in player_rows]
    fig, ax = plt.subplots(figsize=(10.0, 3.8), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    x = np.arange(len(names))
    w = 0.15
    ax.bar(x - 2 * w, v1_vals, w, label="Heurístico v1", color="#5b9bd5", alpha=0.92)
    ax.bar(x - w, v2_vals, w, label="Heurístico v2", color="#70ad47", alpha=0.92)
    ax.bar(x, v3_vals, w, label="Heurístico v3", color="#d4a843", alpha=0.92)
    ax.bar(x + w, v4_vals, w, label="Heurístico v4", color="#c8102e", alpha=0.92)
    ax.bar(x + 2 * w, v5_vals, w, label="Heurístico v5", color="#9b59b6", alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="#c7cdda", fontsize=9)
    ax.set_ylabel("Pass Impact (Σ ΔxT)", color="#c7cdda", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_title("Pass Impact por jogador — v1 / v2 / v3 / v4 / v5", color="#eef1f7", fontsize=11, pad=10)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444466", labelcolor="#eef1f7", fontsize=7, ncol=5)
    ax.axhline(0, color="#444466", lw=0.8)
    for spine in ax.spines.values():
        spine.set_color("#444466")
    fig.tight_layout()
    return _save_fig(fig), fig


def draw_pass_delta_scatter_pair(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    title: str,
    left_label: str,
    right_label: str,
    color: str = "#d4a843",
):
    left = df_left[df_left["is_won"]][["number", "delta_xt"]].rename(columns={"delta_xt": "left"})
    right = df_right[df_right["is_won"]][["number", "delta_xt"]].rename(columns={"delta_xt": "right"})
    merged = left.merge(right, on="number", how="inner")
    fig, ax = plt.subplots(figsize=(4.2, 4.0), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    if not merged.empty:
        lim = max(merged[["left", "right"]].abs().max().max(), 0.02)
        ax.scatter(merged["left"], merged["right"], c=color, alpha=0.75, s=24, edgecolors="white", linewidths=0.4)
        ax.plot([-lim, lim], [-lim, lim], "--", color="#64748b", lw=1.0, alpha=0.8)
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_ylim(-lim * 1.05, lim * 1.05)
        corr = merged["left"].corr(merged["right"])
        ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, color="#eef1f7", fontsize=8, va="top")
    ax.set_xlabel(f"ΔxT {left_label}", color="#c7cdda", fontsize=8)
    ax.set_ylabel(f"ΔxT {right_label}", color="#c7cdda", fontsize=8)
    ax.set_title(title, color="#eef1f7", fontsize=9, pad=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#444466")
    fig.tight_layout()
    return _save_fig(fig), fig


def render_heuristic_comparison(
    hudson_base: pd.DataFrame,
    bentancur_base: pd.DataFrame,
    vitinha_base: pd.DataFrame,
    prog_model: str,
    hudson_label: str,
):
    grid_v1 = compute_heuristic_xt_grid()
    grid_v2 = compute_heuristic_v2_xt_grid()
    grid_v3 = compute_heuristic_v3_xt_grid()
    grid_v4 = compute_heuristic_v4_xt_grid()
    grid_v5 = compute_heuristic_v5_xt_grid()

    st.markdown("---")
    st.markdown("### Comparativo gráfico — xT Heurístico v1 / v2 / v3 / v4 / v5")
    st.caption(
        "v5: limiares P75/P92 por zona (ref. StatsBomb) + ΔxT ajustado na classificação "
        "+ teto de ganho por zona (maior na área, menor no campo defensivo)."
    )

    gcols_top = st.columns(4)
    grid_panels_top = [
        (grid_v1, "Heurístico v1", {"value_fmt": ".2f"}),
        (grid_v2, "Heurístico v2", {"value_fmt": ".2f"}),
        (grid_v3, "Heurístico v3", {"value_fmt": ".2f", "as_percent": True, "color_percentile": (5, 95)}),
        (grid_v4, "Heurístico v4", {"value_fmt": ".2f", "as_percent": True, "color_percentile": (5, 95)}),
    ]
    for col, (grid, label, kwargs) in zip(gcols_top, grid_panels_top):
        with col:
            img, fig = draw_xt_grid_map(grid, label, **kwargs)
            plt.close(fig)
            st.image(img, use_container_width=True)

    gcols_v5 = st.columns(3)
    grid_panels_v5 = [
        (grid_v5, "Heurístico v5", {"value_fmt": ".2f", "as_percent": True, "color_percentile": (5, 95)}),
    ]
    diff_panels_v5 = [
        (grid_v5, grid_v4, "Δ v5 − v4 (normalizados)"),
        (grid_v5, grid_v3, "Δ v5 − v3 (normalizados)"),
    ]
    with gcols_v5[0]:
        grid, label, kwargs = grid_panels_v5[0]
        img, fig = draw_xt_grid_map(grid, label, **kwargs)
        plt.close(fig)
        st.image(img, use_container_width=True)
    for col, (grid_a, grid_b, label) in zip(gcols_v5[1:], diff_panels_v5):
        with col:
            img, fig = draw_xt_diff_map(grid_a, grid_b, label)
            plt.close(fig)
            st.image(img, use_container_width=True)

    diff_cols = st.columns(3)
    diff_panels = [
        (grid_v4, grid_v3, "Δ v4 − v3 (normalizados)"),
        (grid_v4, grid_v1, "Δ v4 − v1 (normalizados)"),
        (grid_v3, grid_v2, "Δ v3 − v2 (normalizados)"),
    ]
    for col, (grid_a, grid_b, label) in zip(diff_cols, diff_panels):
        with col:
            img, fig = draw_xt_diff_map(grid_a, grid_b, label)
            plt.close(fig)
            st.image(img, use_container_width=True)

    bases = [
        ("Hudson Cicala", hudson_base, hudson_label),
        ("Bentancur", bentancur_base, "Copa — vs Arábia Saudita"),
        ("Vitinha", vitinha_base, "Vitinha"),
    ]
    impact_rows = []
    for name, base_df, _ in bases:
        df_v1 = prepare_player_df(base_df, XT_MODEL_HEURISTIC, prog_model)
        df_v2 = prepare_player_df(base_df, XT_MODEL_HEURISTIC_V2, prog_model)
        df_v3 = prepare_player_df(base_df, XT_MODEL_HEURISTIC_V3, prog_model)
        df_v4 = prepare_player_df(base_df, XT_MODEL_HEURISTIC_V4, prog_model)
        df_v5 = prepare_player_df(base_df, XT_MODEL_HEURISTIC_V5, prog_model)
        impact_rows.append({
            "name": name,
            "v1_impact": float(df_v1.loc[df_v1["is_won"], "delta_xt"].sum()),
            "v2_impact": float(df_v2.loc[df_v2["is_won"], "delta_xt"].sum()),
            "v3_impact": float(df_v3.loc[df_v3["is_won"], "delta_xt"].sum()),
            "v4_impact": float(df_v4.loc[df_v4["is_won"], "delta_xt"].sum()),
            "v5_impact": float(df_v5.loc[df_v5["is_won"], "delta_xt"].sum()),
            "df_v1": df_v1,
            "df_v2": df_v2,
            "df_v3": df_v3,
            "df_v4": df_v4,
            "df_v5": df_v5,
        })

    st.markdown("#### Pass Impact e correlação de ΔxT")
    chart_col, scatter_col = st.columns([1.5, 1.0])
    with chart_col:
        img, fig = draw_heuristic_impact_chart(impact_rows)
        plt.close(fig)
        st.image(img, use_container_width=True)
    with scatter_col:
        scatter_tabs = st.tabs(["v4 vs v5", "v3 vs v4", "v1 vs v4"])
        with scatter_tabs[0]:
            bent = next(r for r in impact_rows if r["name"] == "Bentancur")
            img, fig = draw_pass_delta_scatter_pair(
                bent["df_v4"], bent["df_v5"],
                "Bentancur — ΔxT v4 vs v5",
                "v4", "v5", color="#9b59b6",
            )
            plt.close(fig)
            st.image(img, use_container_width=True)
        with scatter_tabs[1]:
            bent = next(r for r in impact_rows if r["name"] == "Bentancur")
            img, fig = draw_pass_delta_scatter_pair(
                bent["df_v3"], bent["df_v4"],
                "Bentancur — ΔxT v3 vs v4",
                "v3", "v4", color="#c8102e",
            )
            plt.close(fig)
            st.image(img, use_container_width=True)
        with scatter_tabs[2]:
            bent = next(r for r in impact_rows if r["name"] == "Bentancur")
            img, fig = draw_pass_delta_scatter_pair(
                bent["df_v1"], bent["df_v4"],
                "Bentancur — ΔxT v1 vs v4",
                "v1", "v4", color="#5b9bd5",
            )
            plt.close(fig)
            st.image(img, use_container_width=True)

    scatter_player_cols = st.columns(3)
    scatter_colors = {
        "Hudson Cicala": "#5b9bd5",
        "Bentancur": "#70ad47",
        "Vitinha": "#d4a843",
    }
    for col, row in zip(scatter_player_cols, impact_rows):
        with col:
            img, fig = draw_pass_delta_scatter_pair(
                row["df_v4"], row["df_v5"],
                f"{row['name']} — v4 vs v5",
                "v4", "v5",
                color=scatter_colors.get(row["name"], "#9b59b6"),
            )
            plt.close(fig)
            st.image(img, use_container_width=True)

    st.markdown("#### Resumo numérico (jogo atual)")
    summary_rows = []
    for row in impact_rows:
        df_v1, df_v2, df_v3, df_v4, df_v5 = (
            row["df_v1"], row["df_v2"], row["df_v3"], row["df_v4"], row["df_v5"],
        )
        won_v1 = df_v1[df_v1["is_won"]]
        won_v2 = df_v2[df_v2["is_won"]]
        won_v3 = df_v3[df_v3["is_won"]]
        won_v4 = df_v4[df_v4["is_won"]]
        won_v5 = df_v5[df_v5["is_won"]]
        summary_rows.append({
            "Jogador": row["name"],
            "ΣΔxT v1": round(float(won_v1["delta_xt"].sum()), 3),
            "ΣΔxT v2": round(float(won_v2["delta_xt"].sum()), 3),
            "ΣΔxT v3": round(float(won_v3["delta_xt"].sum()), 3),
            "ΣΔxT v4": round(float(won_v4["delta_xt"].sum()), 3),
            "ΣΔxT v5": round(float(won_v5["delta_xt"].sum()), 3),
            "% Δ>0 v1": round((won_v1["delta_xt"] > 0).mean() * 100, 1),
            "% Δ>0 v2": round((won_v2["delta_xt"] > 0).mean() * 100, 1),
            "% Δ>0 v3": round((won_v3["delta_xt"] > 0).mean() * 100, 1),
            "% Δ>0 v4": round((won_v4["delta_xt"] > 0).mean() * 100, 1),
            "% Δ>0 v5": round((won_v5["delta_xt"] > 0).mean() * 100, 1),
            "Prog. v1": int(df_v1["progressive"].sum()),
            "Super v1": int(df_v1["highly_progressive"].sum()),
            "Prog. v2": int(df_v2["progressive"].sum()),
            "Super v2": int(df_v2["highly_progressive"].sum()),
            "Prog. v3": int(df_v3["progressive"].sum()),
            "Super v3": int(df_v3["highly_progressive"].sum()),
            "Prog. v4": int(df_v4["progressive"].sum()),
            "Super v4": int(df_v4["highly_progressive"].sum()),
            "Prog. v5": int(df_v5["progressive"].sum()),
            "Super v5": int(df_v5["highly_progressive"].sum()),
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
    h2_grid = compute_heuristic_v2_xt_grid()
    h3_grid = compute_heuristic_v3_xt_grid()
    h4_grid = compute_heuristic_v4_xt_grid()
    h5_grid = compute_heuristic_v5_xt_grid()
    sb_grid = compute_statsbomb_xt_grid()
    db_grid = compute_databallpy_xt_grid()

    panels = [
        (draw_xt_grid_map(h_grid, "xT Heurístico v1", value_fmt=".2f")[0], "Heurístico v1"),
        (draw_xt_grid_map(sb_grid, "xT StatsBomb", value_fmt=".3f")[0], "StatsBomb"),
        (draw_xt_grid_map(db_grid, "xT DataBallPy", value_fmt=".3f")[0], "DataBallPy"),
        (
            draw_xt_diff_map(
                h_grid, sb_grid,
                "Δ Heurístico v1 − StatsBomb (normalizados)",
            )[0],
            "Δ H1 − SB",
        ),
        (
            draw_xt_diff_map(
                h_grid, db_grid,
                "Δ Heurístico v1 − DataBallPy (normalizados)",
            )[0],
            "Δ H1 − DB",
        ),
        (
            draw_xt_diff_map(
                sb_grid, db_grid,
                "Δ StatsBomb − DataBallPy (normalizados)",
            )[0],
            "Δ SB − DB",
        ),
    ]
    heuristic_panels = [
        (draw_xt_grid_map(h_grid, "Heurístico v1", value_fmt=".2f")[0], "Heurístico v1"),
        (draw_xt_grid_map(h2_grid, "Heurístico v2", value_fmt=".2f")[0], "Heurístico v2"),
        (
            draw_xt_grid_map(
                h3_grid, "Heurístico v3", value_fmt=".2f",
                as_percent=True, color_percentile=(5, 95),
            )[0],
            "Heurístico v3",
        ),
        (
            draw_xt_diff_map(h2_grid, h_grid, "Δ v2 − v1 (normalizados)")[0],
            "Δ v2 − v1",
        ),
        (
            draw_xt_diff_map(h3_grid, h2_grid, "Δ v3 − v2 (normalizados)")[0],
            "Δ v3 − v2",
        ),
        (
            draw_xt_diff_map(h3_grid, h_grid, "Δ v3 − v1 (normalizados)")[0],
            "Δ v3 − v1",
        ),
    ]
    heuristic_v4_panels = [
        (
            draw_xt_grid_map(
                h4_grid, "Heurístico v4", value_fmt=".2f",
                as_percent=True, color_percentile=(5, 95),
            )[0],
            "Heurístico v4",
        ),
        (
            draw_xt_diff_map(h4_grid, h3_grid, "Δ v4 − v3 (normalizados)")[0],
            "Δ v4 − v3",
        ),
        (
            draw_xt_diff_map(h4_grid, h_grid, "Δ v4 − v1 (normalizados)")[0],
            "Δ v4 − v1",
        ),
    ]
    heuristic_v5_panels = [
        (
            draw_xt_grid_map(
                h5_grid, "Heurístico v5", value_fmt=".2f",
                as_percent=True, color_percentile=(5, 95),
            )[0],
            "Heurístico v5",
        ),
        (
            draw_xt_diff_map(h5_grid, h4_grid, "Δ v5 − v4 (normalizados)")[0],
            "Δ v5 − v4",
        ),
        (
            draw_xt_diff_map(h5_grid, h3_grid, "Δ v5 − v3 (normalizados)")[0],
            "Δ v5 − v3",
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

    _draw_page_header(
        "Evolução heurística v1 → v2 → v3 | v3: escala fixa por terço, monotônico no ataque",
    )
    _pdf_draw_image_row(
        c, heuristic_panels[:3], page_w, page_h,
        y_top=page_h - 68, row_h=page_h - 100,
    )
    c.showPage()

    _draw_page_header(
        "Diferenças entre versões heurísticas v1–v3 (normalizadas)",
    )
    _pdf_draw_image_row(
        c, heuristic_panels[3:], page_w, page_h,
        y_top=page_h - 68, row_h=page_h - 100,
    )
    c.showPage()

    _draw_page_header(
        "Heurístico v4 | centralidade v1 nos 2/3 + finalização suave (monotônico no ataque)",
    )
    _pdf_draw_image_row(
        c, heuristic_v4_panels, page_w, page_h,
        y_top=page_h - 68, row_h=page_h - 100,
    )
    c.showPage()

    _draw_page_header(
        "Heurístico v5 | zonas suaves + classificação com ΔxT ajustado + teto por zona",
    )
    _pdf_draw_image_row(
        c, heuristic_v5_panels, page_w, page_h,
        y_top=page_h - 68, row_h=page_h - 100,
    )
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner="Gerando PDF comparativo xT…")
def get_xt_comparison_pdf_bytes() -> bytes:
    return build_xt_comparison_pdf()


def render_player_maps(df: pd.DataFrame, prog_model: str, pass_filter: str = PASS_MAP_FILTER_ALL):
    if pass_filter == PASS_MAP_FILTER_SUPER:
        df_map = df[df["highly_progressive"]].copy()
        if df_map.empty:
            st.info("Nenhum passe super progressivo neste recorte.")
            return
    else:
        df_map = df

    img_pm, fig_pm = draw_pass_map(df_map, prog_model)
    plt.close(fig_pm)
    st.markdown('<div class="map-label">Pass Map</div>', unsafe_allow_html=True)
    st.image(img_pm, use_container_width=True)

    img_ht, fig_ht = draw_corridor_heatmap(df_map)
    plt.close(fig_ht)
    st.markdown('<div class="map-label">Zone Heatmap (Destination)</div>', unsafe_allow_html=True)
    st.image(img_ht, use_container_width=True)

    img_xt, fig_xt = draw_top_xt_map(df_map, top_n=5)
    plt.close(fig_xt)
    st.markdown('<div class="map-label">Top 5 Pass Impact</div>', unsafe_allow_html=True)
    st.image(img_xt, use_container_width=True)


def render_player_cards(stats: dict, tone: str, prog_model: str):
    progressive_items = [
        ("Tentativas Progressivas", f"{stats['progressive_attempted']:.0f}"),
        ("Passes Progressivos", f"{stats['progressive_successful']:.0f}"),
        ("Super Progressivos", f"{stats['highly_progressive']:.0f}"),
        ("% Acurácia Progressiva", f"{stats['progressive_accuracy_pct']:.1f}%"),
    ]
    if prog_model == PROG_MODEL_XT:
        progressive_items.append(
            ("% Altamente Prog. (xT)", f"{stats['highly_progressive_pct']:.1f}%"),
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
            ("Pass Impact (modelo)", f"{stats['sum_dxt']:.2f}"),
            ("Pass Impact v2", f"{stats['sum_dxt_v2']:.2f}"),
            ("% Positive Impact", f"{stats['pos_pct']:.1f}%"),
        ],
    )


# ── DATA LOAD ──────────────────────────────────────────────────
hudson_dfs, wc_dfs = load_all_pass_data()

if not hudson_dfs or BENTANCUR_KEY not in wc_dfs or VITINHA_KEY not in wc_dfs:
    st.error(
        "Não foi possível carregar os dados. Verifique se o arquivo "
        f"'{HUDSON_DOCX}' está no diretório do app."
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
      Bentancur e Vitinha: jogos fixos da Copa do Mundo.<br>
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
if xt_model == XT_MODEL_HEURISTIC_V5:
    _v5_th = compute_v5_reference_thresholds()
    _v5_lines = [
        f"{zone}: prog≥{vals['prog']:.3f}, super>{vals['high']:.3f}"
        for zone, vals in _v5_th.items()
    ]
    st.sidebar.caption("Limiares v5 (ref. StatsBomb): " + " · ".join(_v5_lines))

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
st.sidebar.markdown("**Filtro do mapa de passes**")
pass_map_filter = st.sidebar.radio(
    "Filtro do mapa",
    options=list(PASS_MAP_FILTER_LABELS.keys()),
    format_func=lambda k: PASS_MAP_FILTER_LABELS[k],
    key="pass_map_filter_selector",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
pdf_bytes = get_xt_comparison_pdf_bytes()
st.sidebar.download_button(
    label="Exportar PDF — Mapas xT",
    data=pdf_bytes,
    file_name="comparativo_xt_mapas.pdf",
    mime="application/pdf",
    use_container_width=True,
)
st.sidebar.caption("PDF: modelos externos + evolução heurística v1/v2/v3/v4/v5.")

# ── MAIN LAYOUT ────────────────────────────────────────────────
st.markdown("## Passes — Comparação de Jogadores")
st.caption("Hudson Cicala vs Bentancur (vs Arábia Saudita) vs Vitinha")

selected_hudson_match = st.selectbox(
    "Selecione o jogo de Hudson Cicala para comparar",
    options=hudson_match_names,
    index=0,
    key="hudson_match_selector",
)

hudson_df = prepare_player_df(hudson_dfs[selected_hudson_match], xt_model, prog_model)
bentancur_df = prepare_player_df(wc_dfs[BENTANCUR_KEY], xt_model, prog_model)
vitinha_df = prepare_player_df(wc_dfs[VITINHA_KEY], xt_model, prog_model)

hudson_stats = compute_stats(hudson_df, selected_hudson_match, prog_model, xt_model)
bentancur_stats = compute_stats(bentancur_df, BENTANCUR_KEY, prog_model, xt_model)
vitinha_stats = compute_stats(vitinha_df, VITINHA_KEY, prog_model, xt_model)

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
        "name": "Vitinha",
        "subtitle": "Vitinha",
        "df": vitinha_df,
        "stats": vitinha_stats,
        "tone": PLAYER_TONES["Vitinha"],
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
        render_player_maps(player["df"], prog_model, pass_map_filter)

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

render_heuristic_comparison(
    hudson_dfs[selected_hudson_match],
    wc_dfs[BENTANCUR_KEY],
    wc_dfs[VITINHA_KEY],
    prog_model,
    selected_hudson_match,
)
