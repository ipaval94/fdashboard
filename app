# ============================================================
# GOALKEEPER DASHBOARD + MULTI-FILE GK COMPARISON (Google Colab)
# Futuristic + modern styling, proportionate charts
# ============================================================

!pip -q install plotly openpyxl

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from google.colab import files

# -------------------- Global visual theme (modern / futuristic) --------------------
THEME_BG = "#0b0f1a"          # deep navy
THEME_GRID = "rgba(255,255,255,0.08)"
THEME_TEXT = "rgba(255,255,255,0.92)"

CHART_HEIGHT = 420           # consistent, not too big
CHART_MARGIN = dict(l=40, r=25, t=60, b=45)

pio.templates.default = "plotly_dark"

def style_future(fig, title=None, height=CHART_HEIGHT):
    fig.update_layout(
        title=title if title else fig.layout.title.text,
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

def show(fig, title=None, height=CHART_HEIGHT):
    fig = style_future(fig, title=title, height=height)
    fig.show()

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

def read_any_file(fname: str) -> pd.DataFrame:
    if fname.lower().endswith(".csv"):
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_err = None
        for enc in encodings:
            try:
                return pd.read_csv(fname, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
            except Exception as e:
                last_err = e
        raise ValueError(f"Could not read {fname}. Last error: {last_err}")
    else:
        return pd.read_excel(fname)

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

# -------------------- Upload & load --------------------
print("Upload your GK performance file(s) (CSV or Excel). You can select multiple.")
uploaded = files.upload()

dfs = []
for fname in uploaded.keys():
    df = read_any_file(fname)
    df = clean_cols(df)

    # strip text cols
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["source_file"] = fname
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True, sort=False)
data = make_plot_safe(data)
data = ensure_date(data)

print("\n✅ Files loaded:", len(dfs))
print("✅ Rows:", len(data), "| Columns:", len(data.columns))
print("\nColumns detected:")
print(list(data.columns))
print("\nPreview:")
display(data.head(10))

# ============================================================
# 1) SINGLE DATASET DASHBOARD (requested chart types)
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

    # 1) Saves vs Shots Against -> BAR
    if {"saves_with_reflexes","shots_against"}.issubset(d.columns):
        if x_time is None:
            tmp = pd.DataFrame({
                "metric":["saves_with_reflexes","shots_against"],
                "value":[np.nansum(d["saves_with_reflexes"]), np.nansum(d["shots_against"])]
            })
            fig = px.bar(tmp, x="metric", y="value")
            fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
            show(fig, "Saves vs Shots Against (Totals)")
        else:
            tmp = d[[x_time,"saves_with_reflexes","shots_against"]].dropna(subset=[x_time])
            fig = px.bar(tmp, x=x_time, y=["saves_with_reflexes","shots_against"], barmode="group")
            fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
            show(fig, "Saves vs Shots Against (Per Date)")

    # 2) Goal kick distribution short vs long -> BAR
    if {"short_goal_kicks","long_goal_kicks"}.issubset(d.columns):
        if x_time is None:
            tmp = pd.DataFrame({
                "type":["short_goal_kicks","long_goal_kicks"],
                "count":[np.nansum(d["short_goal_kicks"]), np.nansum(d["long_goal_kicks"])]
            })
            fig = px.bar(tmp, x="type", y="count")
            fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
            show(fig, "Goal Kick Distribution (Totals): Short vs Long")
        else:
            tmp = d[[x_time,"short_goal_kicks","long_goal_kicks"]].dropna(subset=[x_time])
            fig = px.bar(tmp, x=x_time, y=["short_goal_kicks","long_goal_kicks"], barmode="group")
            fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
            show(fig, "Goal Kick Distribution: Short vs Long (Over Time)")

    # 3) Distribution accuracy -> BAR (prefer rates if attempts exist)
    if {"short_passes_accurate","long_passes_accurate","short_passes","long_passes"}.issubset(d.columns):
        d["short_acc_rate"] = d["short_passes_accurate"] / d["short_passes"].replace(0, np.nan)
        d["long_acc_rate"]  = d["long_passes_accurate"]  / d["long_passes"].replace(0, np.nan)

        if x_time is None:
            tmp = pd.DataFrame({
                "type":["short_acc_rate","long_acc_rate"],
                "rate":[np.nanmean(d["short_acc_rate"]), np.nanmean(d["long_acc_rate"])]
            })
            fig = px.bar(tmp, x="type", y="rate")
            fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
            show(fig, "Distribution Accuracy (Avg Rate): Short vs Long")
        else:
            tmp = d[[x_time,"short_acc_rate","long_acc_rate"]].dropna(subset=[x_time])
            fig = px.bar(tmp, x=x_time, y=["short_acc_rate","long_acc_rate"], barmode="group")
            fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
            show(fig, "Distribution Accuracy (Rate): Short vs Long (Over Time)")

    elif {"short_passes_accurate","long_passes_accurate"}.issubset(d.columns):
        # fallback: accurate counts
        if x_time is None:
            tmp = pd.DataFrame({
                "type":["short_passes_accurate","long_passes_accurate"],
                "count":[np.nansum(d["short_passes_accurate"]), np.nansum(d["long_passes_accurate"])]
            })
            fig = px.bar(tmp, x="type", y="count")
            fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
            show(fig, "Distribution Accuracy (Totals): Short Accurate vs Long Accurate")
        else:
            tmp = d[[x_time,"short_passes_accurate","long_passes_accurate"]].dropna(subset=[x_time])
            fig = px.bar(tmp, x=x_time, y=["short_passes_accurate","long_passes_accurate"], barmode="group")
            fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
            show(fig, "Distribution Accuracy: Short Accurate vs Long Accurate (Over Time)")

    # 4) Conceded goals -> BAR
    if "conceded_goals" in d.columns:
        if x_time is None:
            tmp = pd.DataFrame({"metric":["conceded_goals_total"], "value":[np.nansum(d["conceded_goals"])]})
            fig = px.bar(tmp, x="metric", y="value")
            fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
            show(fig, "Conceded Goals (Total)")
        else:
            tmp = d[[x_time,"conceded_goals"]].dropna(subset=[x_time])
            fig = px.bar(tmp, x=x_time, y="conceded_goals")
            fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.22)")
            show(fig, "Conceded Goals (Over Time)")

    # 5) Exits / Sweeper -> RADAR (uses what exists; stays readable)
    minutes = d["minutes_played"] if "minutes_played" in d.columns else None

    radar_axes, radar_vals = [], []

    def add_axis(label, series, mode="mean"):
        val = np.nanmean(series) if mode == "mean" else np.nansum(series)
        if np.isfinite(val):
            radar_axes.append(label)
            radar_vals.append(val)

    # Primary
    if "exits" in d.columns and minutes is not None:
        add_axis("Exits p90", per90(d["exits"], minutes), "mean")
    elif "exits" in d.columns:
        add_axis("Exits (avg)", d["exits"], "mean")

    # Add other “sweeper-ish” dims if present
    for col, label in [
        ("sweeper_actions", "Sweeper Actions (avg)"),
        ("claims", "Claims (avg)"),
        ("high_claims", "High Claims (avg)"),
        ("punches", "Punches (avg)")
    ]:
        if col in d.columns:
            add_axis(label, d[col], "mean")

    # Add context dims if available
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
        fig.show()
    else:
        # fallback: if only exits exists, show bar (still modern)
        if "exits" in d.columns:
            if x_time is None:
                tmp = pd.DataFrame({"metric":["exits_total"], "value":[np.nansum(d["exits"])]})
                fig = px.bar(tmp, x="metric", y="value")
                show(fig, "Exits / Sweeper Actions (Total)")
            else:
                tmp = d[[x_time,"exits"]].dropna(subset=[x_time])
                fig = px.bar(tmp, x=x_time, y="exits")
                show(fig, "Exits / Sweeper Actions (Over Time)")

print("\n--- DASHBOARD (dataset view) ---")
goalkeeper_dashboards(data)

# ============================================================
# 2) GK COMPARISON (time-series + modern summary bars + radar)
# ============================================================
def compare_goalkeepers(d: pd.DataFrame, max_gks: int = 20):
    d = ensure_date(make_plot_safe(d))
    gk_id = detect_gk_id_column(d)

    if "date" not in d.columns or not d["date"].notna().any():
        print("\n⚠️ No usable 'date' column detected for time-series comparison.")
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

    # Aggregate per GK per date
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

    # Derived
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

    print(f"\n✅ Comparing goalkeepers using ID column: {gk_id}")
    print("Goalkeepers detected:", agg[gk_id].nunique())

    # --- Time-series multi-line (kept compact + readable)
    def line_future(y, title):
        fig = px.line(agg, x="date", y=y, color=gk_id)
        fig.update_traces(line=dict(width=2.6))
        show(fig, title, height=440)

    if "save_rate" in agg.columns:
        line_future("save_rate", "GK Comparison (Over Time): Save Rate")

    if "conceded_p90" in agg.columns:
        line_future("conceded_p90", "GK Comparison (Over Time): Conceded Goals p90")

    if "shots_against_p90" in agg.columns:
        line_future("shots_against_p90", "GK Comparison (Over Time): Shots Against p90")

    if "exits_p90" in agg.columns:
        line_future("exits_p90", "GK Comparison (Over Time): Exits p90")

    # --- Summary BAR charts per GK (requested)
    # Conceded goals (total)
    if "conceded_goals" in agg.columns:
        totals = agg.groupby(gk_id, as_index=False)["conceded_goals"].sum()
        fig = px.bar(totals, x=gk_id, y="conceded_goals")
        fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
        show(fig, "GK Comparison: Conceded Goals (Total)")

    # Saves vs shots against (totals)
    if {"saves_with_reflexes","shots_against"}.issubset(agg.columns):
        totals = agg.groupby(gk_id, as_index=False).agg({"saves_with_reflexes":"sum","shots_against":"sum"})
        melted = totals.melt(id_vars=[gk_id], var_name="metric", value_name="value")
        fig = px.bar(melted, x=gk_id, y="value", color="metric", barmode="group")
        fig.update_traces(marker_line_width=0.7, marker_line_color="rgba(255,255,255,0.22)")
        show(fig, "GK Comparison: Saves vs Shots Against (Totals)")

    # Goal kick distribution (totals)
    if {"short_goal_kicks","long_goal_kicks"}.issubset(agg.columns):
        totals = agg.groupby(gk_id, as_index=False).agg({"short_goal_kicks":"sum","long_goal_kicks":"sum"})
        melted = totals.melt(id_vars=[gk_id], var_name="type", value_name="count")
        fig = px.bar(melted, x=gk_id, y="count", color="type", barmode="group")
        fig.update_traces(marker_line_width=0.7, marker_line_color="rgba(255,255,255,0.22)")
        show(fig, "GK Comparison: Goal Kick Distribution (Short vs Long)")

    # Distribution accuracy (prefer rates)
    if {"short_acc_rate","long_acc_rate"}.issubset(agg.columns):
        means = agg.groupby(gk_id, as_index=False).agg({"short_acc_rate":"mean","long_acc_rate":"mean"})
        melted = means.melt(id_vars=[gk_id], var_name="type", value_name="rate")
        fig = px.bar(melted, x=gk_id, y="rate", color="type", barmode="group")
        fig.update_traces(marker_line_width=0.7, marker_line_color="rgba(255,255,255,0.22)")
        show(fig, "GK Comparison: Distribution Accuracy (Avg Rate) — Short vs Long")

    elif {"short_passes_accurate","long_passes_accurate"}.issubset(agg.columns):
        totals = agg.groupby(gk_id, as_index=False).agg({"short_passes_accurate":"sum","long_passes_accurate":"sum"})
        melted = totals.melt(id_vars=[gk_id], var_name="type", value_name="accurate")
        fig = px.bar(melted, x=gk_id, y="accurate", color="type", barmode="group")
        fig.update_traces(marker_line_width=0.7, marker_line_color="rgba(255,255,255,0.22)")
        show(fig, "GK Comparison: Distribution Accuracy (Totals) — Short vs Long Accurate")

    # --- Radar: sweeper profile per GK (compact + readable)
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
        fig.show()

print("\n--- GK COMPARISON ---")
compare_goalkeepers(data)

print("\n✅ Done.")
