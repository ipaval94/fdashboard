# ============================================================
# GOALKEEPER DASHBOARD + MULTI-FILE GK COMPARISON (Streamlit)
# Futuristic + modern styling, proportionate charts
# ============================================================

import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# -------------------- Page config --------------------
st.set_page_config(page_title="GK Dashboard", page_icon="🧤", layout="wide")

# -------------------- Global visual theme (modern / futuristic) --------------------
THEME_BG = "#0b0f1a"          # deep navy
THEME_GRID = "rgba(255,255,255,0.08)"
THEME_TEXT = "rgba(255,255,255,0.92)"

CHART_HEIGHT = 420           # consistent, not too big
CHART_MARGIN = dict(l=40, r=25, t=60, b=45)

pio.templates.default = "plotly_dark"

def style_future(fig, title=None, height=CHART_HEIGHT):
    fig.update_layout(
        title=title if title else (fig.layout.title.text if fig.layout.title else None),
        title_x=0.02,
        height=height,
        margin=CHART_MARGIN,
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font=dict(family="Inter, Arial", size=13, color=THEME_TEXT),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            title_text=""
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=THEME_GRID, zeroline=False)
    return fig

def show_st(fig, title=None, height=CHART_HEIGHT, use_container_width=True):
    fig = style_future(fig, title=title, height=height)
    st.plotly_chart(fig, use_container_width=use_container_width)

def safe_time_axis(d: pd.DataFrame):
    return "date" if "date" in d.columns and d["date"].notna().any() else None

# -------------------- Data helpers --------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def make_plot_safe(d: pd.DataFrame) -> pd.DataFrame:
    return d.copy().replace({pd.NA: np.nan})

def to_num(d: pd.DataFrame, cols) -> pd.DataFrame:
    d = d.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def per90(val, minutes):
    minutes = minutes.replace(0, np.nan)
    return val / (minutes / 90)

def read_any_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_err = None
        raw = uploaded_file.getvalue()
        for enc in encodings:
            try:
                return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc, on_bad_lines="skip")
            except Exception as e:
                last_err = e
        raise ValueError(f"Could not read {uploaded_file.name}. Last error: {last_err}")
    else:
        return pd.read_excel(uploaded_file)

def detect_gk_id_column(d: pd.DataFrame) -> str:
    for c in ["goalkeeper", "player", "player_name", "name", "athlete"]:
        if c in d.columns:
            return c
    return "source_file"

def ensure_date(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.sort_values("date")
    return d

# ============================================================
# 1) SINGLE DATASET DASHBOARD
# ============================================================
def goalkeeper_dashboards(d: pd.DataFrame):
    d = ensure_date(make_plot_safe(d))

    numeric_cols = [
        "minutes_played","shots_against","conceded_goals","xcg",
        "saves_with_reflexes",
        "exits","sweeper_actions","claims","high_claims","punches",
        "short_goal_kicks","long_goal_kicks",
        "short_passes_accurate","long_passes_accurate",
        "short_passes","long_passes"
    ]
    d = to_num(d, numeric_cols)
    x_time = safe_time_axis(d)

    left, right = st.columns(2)

    # 1) Saves vs Shots Against -> BAR
    if {"saves_with_reflexes","shots_against"}.issubset(d.columns):
        with left:
            if x_time is None:
                tmp = pd.DataFrame({
                    "metric":["saves_with_reflexes","shots_against"],
                    "value":[np.nansum(d["saves_with_reflexes"]), np.nansum(d["shots_against"])]
                })
                fig = px.bar(tmp, x="metric", y="value")
                fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
                show_st(fig, "Saves vs Shots Against (Totals)")
            else:
                tmp = d[[x_time,"saves_with_reflexes","shots_against"]].dropna(subset=[x_time])
                fig = px.bar(tmp, x=x_time, y=["saves_with_reflexes","shots_against"], barmode="group")
                fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
                show_st(fig, "Saves vs Shots Against (Per Date)")

    # 2) Goal kick distribution short vs long -> BAR
    if {"short_goal_kicks","long_goal_kicks"}.issubset(d.columns):
        with right:
            if x_time is None:
                tmp = pd.DataFrame({
                    "type":["short_goal_kicks","long_goal_kicks"],
                    "count":[np.nansum(d["short_goal_kicks"]), np.nansum(d["long_goal_kicks"])]
                })
                fig = px.bar(tmp, x="type", y="count")
                fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
                show_st(fig, "Goal Kick Distribution (Totals): Short vs Long")
            else:
                tmp = d[[x_time,"short_goal_kicks","long_goal_kicks"]].dropna(subset=[x_time])
                fig = px.bar(tmp, x=x_time, y=["short_goal_kicks","long_goal_kicks"], barmode="group")
                fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
                show_st(fig, "Goal Kick Distribution: Short vs Long (Over Time)")

    # 3) Distribution accuracy -> BAR (prefer rates if attempts exist)
    left2, right2 = st.columns(2)
    if {"short_passes_accurate","long_passes_accurate","short_passes","long_passes"}.issubset(d.columns):
        d["short_acc_rate"] = d["short_passes_accurate"] / d["short_passes"].replace(0, np.nan)
        d["long_acc_rate"]  = d["long_passes_accurate"]  / d["long_passes"].replace(0, np.nan)

        with left2:
            if x_time is None:
                tmp = pd.DataFrame({
                    "type":["short_acc_rate","long_acc_rate"],
                    "rate":[np.nanmean(d["short_acc_rate"]), np.nanmean(d["long_acc_rate"])]
                })
                fig = px.bar(tmp, x="type", y="rate")
                fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
                show_st(fig, "Distribution Accuracy (Avg Rate): Short vs Long")
            else:
                tmp = d[[x_time,"short_acc_rate","long_acc_rate"]].dropna(subset=[x_time])
                fig = px.bar(tmp, x=x_time, y=["short_acc_rate","long_acc_rate"], barmode="group")
                fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
                show_st(fig, "Distribution Accuracy (Rate): Short vs Long (Over Time)")

    elif {"short_passes_accurate","long_passes_accurate"}.issubset(d.columns):
        with left2:
            if x_time is None:
                tmp = pd.DataFrame({
                    "type":["short_passes_accurate","long_passes_accurate"],
                    "count":[np.nansum(d["short_passes_accurate"]), np.nansum(d["long_passes_accurate"])]
                })
                fig = px.bar(tmp, x="type", y="count")
                fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
                show_st(fig, "Distribution Accuracy (Totals): Short Accurate vs Long Accurate")
            else:
                tmp = d[[x_time,"short_passes_accurate","long_passes_accurate"]].dropna(subset=[x_time])
                fig = px.bar(tmp, x=x_time, y=["short_passes_accurate","long_passes_accurate"], barmode="group")
                fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
                show_st(fig, "Distribution Accuracy: Short Accurate vs Long Accurate (Over Time)")

    # 4) Conceded goals -> BAR
    if "conceded_goals" in d.columns:
        with right2:
            if x_time is None:
                tmp = pd.DataFrame({"metric":["conceded_goals_total"], "value":[np.nansum(d["conceded_goals"])]})
                fig = px.bar(tmp, x="metric", y="value")
                fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
                show_st(fig, "Conceded Goals (Total)")
            else:
                tmp = d[[x_time,"conceded_goals"]].dropna(subset=[x_time])
                fig = px.bar(tmp, x=x_time, y="conceded_goals")
                fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
                show_st(fig, "Conceded Goals (Over Time)")

    # 5) Exits / Sweeper -> RADAR
    st.markdown("### Exits / Sweeper Profile")
    minutes = d["minutes_played"] if "minutes_played" in d.columns else None
    radar_axes, radar_vals = [], []

    def add_axis(label, series, mode="mean"):
        val = np.nanmean(series) if mode == "mean" else np.nansum(series)
        if np.isfinite(val):
            radar_axes.append(label)
            radar_vals.append(val)

    if "exits" in d.columns and minutes is not None:
        add_axis("Exits p90", per90(d["exits"], minutes), "mean")
    elif "exits" in d.columns:
        add_axis("Exits (avg)", d["exits"], "mean")

    for col, label in [
        ("sweeper_actions", "Sweeper Actions (avg)"),
        ("claims", "Claims (avg)"),
        ("high_claims", "High Claims (avg)"),
        ("punches", "Punches (avg)")
    ]:
        if col in d.columns:
            add_axis(label, d[col], "mean")

    if {"saves_with_reflexes","shots_against"}.issubset(d.columns):
        add_axis("Save Rate", d["saves_with_reflexes"] / d["shots_against"].replace(0, np.nan), "mean")

    if len(radar_axes) >= 3:
        r = radar_vals + [radar_vals[0]]
        theta = radar_axes + [radar_axes[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            line=dict(width=2),
            name="Profile"
        ))
        fig.update_layout(
            height=CHART_HEIGHT,
            margin=CHART_MARGIN,
            paper_bgcolor=THEME_BG,
            font=dict(family="Inter, Arial", size=13, color=THEME_TEXT),
            title="Exits / Sweeper Actions — Radar Profile",
            title_x=0.02,
            showlegend=False,
            polar=dict(
                bgcolor=THEME_BG,
                radialaxis=dict(showgrid=True, gridcolor=THEME_GRID),
                angularaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough sweeper-related metrics found to build a radar (need at least 3).")

# ============================================================
# 2) GK COMPARISON
# ============================================================
def compare_goalkeepers(d: pd.DataFrame, max_gks: int = 20):
    d = ensure_date(make_plot_safe(d))
    gk_id = detect_gk_id_column(d)

    if "date" not in d.columns or not d["date"].notna().any():
        st.warning("No usable 'date' column detected for time-series comparison.")
        return

    d = to_num(d, [
        "minutes_played","shots_against","conceded_goals","xcg","saves_with_reflexes",
        "exits","sweeper_actions","claims","high_claims","punches",
        "short_goal_kicks","long_goal_kicks",
        "short_passes_accurate","long_passes_accurate","short_passes","long_passes"
    ])

    d = d.dropna(subset=[gk_id, "date"])
    gk_list = list(d[gk_id].dropna().unique())[:max_gks]
    d = d[d[gk_id].isin(gk_list)]

    agg_cols = {
        "minutes_played":"sum",
        "shots_against":"sum",
        "conceded_goals":"sum",
        "xcg":"sum",
        "saves_with_reflexes":"sum",
        "exits":"sum",
        "sweeper_actions":"sum",
        "claims":"sum",
        "high_claims":"sum",
        "punches":"sum",
        "short_goal_kicks":"sum",
        "long_goal_kicks":"sum",
        "short_passes_accurate":"sum",
        "long_passes_accurate":"sum",
        "short_passes":"sum",
        "long_passes":"sum"
    }
    agg_cols = {k:v for k,v in agg_cols.items() if k in d.columns}
    agg = d.groupby([gk_id, "date"], as_index=False).agg(agg_cols)

    if {"saves_with_reflexes","shots_against"}.issubset(agg.columns):
        agg["save_rate"] = agg["saves_with_reflexes"] / agg["shots_against"].replace(0, np.nan)

    if {"xcg","conceded_goals"}.issubset(agg.columns):
        agg["xg_prevented_proxy"] = agg["xcg"] - agg["conceded_goals"]

    if "minutes_played" in agg.columns:
        if "shots_against" in agg.columns:
            agg["shots_against_p90"] = per90(agg["shots_against"], agg["minutes_played"])
        if "conceded_goals" in agg.columns:
            agg["conceded_p90"] = per90(agg["conceded_goals"], agg["minutes_played"])
        if "exits" in agg.columns:
            agg["exits_p90"] = per90(agg["exits"], agg["minutes_played"])

    if {"short_passes_accurate","short_passes"}.issubset(agg.columns):
        agg["short_acc_rate"] = agg["short_passes_accurate"] / agg["short_passes"].replace(0, np.nan)
    if {"long_passes_accurate","long_passes"}.issubset(agg.columns):
        agg["long_acc_rate"] = agg["long_passes_accurate"] / agg["long_passes"].replace(0, np.nan)

    st.caption(f"Comparing GKs using ID column: `{gk_id}` • GKs detected: {agg[gk_id].nunique()}")

    # Time-series multi-line
    def line_future(y, title):
        fig = px.line(agg, x="date", y=y, color=gk_id)
        fig.update_traces(line=dict(width=2.6))
        show_st(fig, title, height=440)

    c1, c2 = st.columns(2)
    with c1:
        if "save_rate" in agg.columns:
            line_future("save_rate", "GK Comparison (Over Time): Save Rate")
    with c2:
        if "conceded_p90" in agg.columns:
            line_future("conceded_p90", "GK Comparison (Over Time): Conceded Goals p90")

    c3, c4 = st.columns(2)
    with c3:
        if "shots_against_p90" in agg.columns:
            line_future("shots_against_p90", "GK Comparison (Over Time): Shots Against p90")
    with c4:
        if "exits_p90" in agg.columns:
            line_future("exits_p90", "GK Comparison (Over Time): Exits p90")

    st.markdown("### Summary comparisons")

    # Summary BAR charts per GK
    cols = st.columns(2)

    if "conceded_goals" in agg.columns:
        totals = agg.groupby(gk_id, as_index=False)["conceded_goals"].sum()
        fig = px.bar(totals, x=gk_id, y="conceded_goals")
        fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
        with cols[0]:
            show_st(fig, "Conceded Goals (Total)")

    if {"saves_with_reflexes","shots_against"}.issubset(agg.columns):
        totals = agg.groupby(gk_id, as_index=False).agg({"saves_with_reflexes":"sum","shots_against":"sum"})
        melted = totals.melt(id_vars=[gk_id], var_name="metric", value_name="value")
        fig = px.bar(melted, x=gk_id, y="value", color="metric", barmode="group")
        fig.update_traces(marker_line_width=0.7, marker_line_color="rgba(255,255,255,0.22)")
        with cols[1]:
            show_st(fig, "Saves vs Shots Against (Totals)")

    cols2 = st.columns(2)

    if {"short_goal_kicks","long_goal_kicks"}.issubset(agg.columns):
        totals = agg.groupby(gk_id, as_index=False).agg({"short_goal_kicks":"sum","long_goal_kicks":"sum"})
        melted = totals.melt(id_vars=[gk_id], var_name="type", value_name="count")
        fig = px.bar(melted, x=gk_id, y="count", color="type", barmode="group")
        fig.update_traces(marker_line_width=0.7, marker_line_color="rgba(255,255,255,0.22)")
        with cols2[0]:
            show_st(fig, "Goal Kick Distribution (Short vs Long)")

    if {"short_acc_rate","long_acc_rate"}.issubset(agg.columns):
        means = agg.groupby(gk_id, as_index=False).agg({"short_acc_rate":"mean","long_acc_rate":"mean"})
        melted = means.melt(id_vars=[gk_id], var_name="type", value_name="rate")
        fig = px.bar(melted, x=gk_id, y="rate", color="type", barmode="group")
        fig.update_traces(marker_line_width=0.7, marker_line_color="rgba(255,255,255,0.22)")
        with cols2[1]:
            show_st(fig, "Distribution Accuracy (Avg Rate) — Short vs Long")

    # Radar: sweeper profile per GK
    st.markdown("### GK Comparison: Exits / Sweeper Radar Profile")
    radar_candidates = []
    if "exits_p90" in agg.columns: radar_candidates.append(("exits_p90","Exits p90","mean"))
    if "sweeper_actions" in agg.columns: radar_candidates.append(("sweeper_actions","Sweeper Actions","sum"))
    if "claims" in agg.columns: radar_candidates.append(("claims","Claims","sum"))
    if "high_claims" in agg.columns: radar_candidates.append(("high_claims","High Claims","sum"))
    if "punches" in agg.columns: radar_candidates.append(("punches","Punches","sum"))
    if "save_rate" in agg.columns: radar_candidates.append(("save_rate","Save Rate","mean"))

    if len(radar_candidates) >= 3:
        agg_map = {col: (mode if mode in ["sum","mean"] else "mean") for col,_,mode in radar_candidates}
        per_gk = agg.groupby(gk_id, as_index=False).agg(agg_map)

        axes = [label for _,label,_ in radar_candidates]

        fig = go.Figure()
        for _, row in per_gk.iterrows():
            vals = [row[col] for col,_,_ in radar_candidates]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=axes + [axes[0]],
                fill="toself",
                line=dict(width=2),
                name=str(row[gk_id])
            ))

        fig.update_layout(
            height=440,
            margin=CHART_MARGIN,
            paper_bgcolor=THEME_BG,
            font=dict(family="Inter, Arial", size=13, color=THEME_TEXT),
            title="GK Comparison: Exits / Sweeper Radar Profile",
            title_x=0.02,
            polar=dict(
                bgcolor=THEME_BG,
                radialaxis=dict(showgrid=True, gridcolor=THEME_GRID),
                angularaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            ),
            legend=dict(orientation="h", y=1.05, x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough sweeper-related metrics found to build a multi-GK radar (need at least 3).")

# -------------------- UI --------------------
st.title("🧤 Goalkeeper Dashboard + Multi-file GK Comparison")
st.caption("Upload one or more CSV/XLSX files. Streamlit will build the dashboard and compare goalkeepers over time.")

with st.sidebar:
    st.header("Upload")
    uploaded_files = st.file_uploader(
        "Upload GK performance file(s) (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    st.divider()
    st.header("Options")
    max_gks = st.slider("Max goalkeepers in comparison", min_value=2, max_value=30, value=20)
    view_mode = st.radio("View", ["Dataset dashboard", "GK comparison", "Both"], index=2)

if not uploaded_files:
    st.info("Upload at least one file to begin.")
    st.stop()

# -------------------- Load data --------------------
dfs = []
for uf in uploaded_files:
    df = read_any_file(uf)
    df = clean_cols(df)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["source_file"] = uf.name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True, sort=False)
data = make_plot_safe(data)
data = ensure_date(data)

st.success(f"✅ Files loaded: {len(dfs)} • Rows: {len(data)} • Columns: {len(data.columns)}")

with st.expander("Preview data"):
    st.write("Detected columns:")
    st.code(", ".join(list(data.columns)))
    st.dataframe(data.head(25), use_container_width=True)

# -------------------- Render dashboards --------------------
if view_mode in ("Dataset dashboard", "Both"):
    st.subheader("Dataset dashboard")
    goalkeeper_dashboards(data)

if view_mode in ("GK comparison", "Both"):
    st.subheader("GK comparison")
    compare_goalkeepers(data, max_gks=max_gks)
