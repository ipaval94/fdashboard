import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# -------------------- Page config --------------------
st.set_page_config(page_title="Sport Auto Dashboard", page_icon="📊", layout="wide")

# -------------------- Theme --------------------
THEME_BG = "#0b0f1a"
THEME_GRID = "rgba(255,255,255,0.08)"
THEME_TEXT = "rgba(255,255,255,0.92)"
CHART_HEIGHT = 420
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

def show(fig, title=None, height=CHART_HEIGHT):
    fig = style_future(fig, title=title, height=height)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Helpers --------------------
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

def read_any_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv"):
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_err = None
        for enc in encodings:
            try:
                return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc, on_bad_lines="skip")
            except Exception as e:
                last_err = e
        raise ValueError(f"Could not read {uploaded_file.name}. Last error: {last_err}")
    return pd.read_excel(uploaded_file)

def coerce_dates(df: pd.DataFrame, max_cols_to_try: int = 6) -> tuple[pd.DataFrame, str | None]:
    """
    Try to detect a date-like column:
    - Prefer common names (date, match_date, game_date, timestamp, etc.)
    - Else test a few object columns by attempting to parse
    """
    df = df.copy()
    preferred = [
        "date", "match_date", "game_date", "timestamp", "time", "datetime",
        "kickoff", "kick_off", "start_time"
    ]
    for c in preferred:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
            if parsed.notna().mean() >= 0.6:
                df[c] = parsed
                return df.sort_values(c), c

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    # Try the most promising few (by uniqueness and length)
    candidates = sorted(
        obj_cols,
        key=lambda c: (df[c].nunique(dropna=True), df[c].astype(str).str.len().median() if len(df) else 0),
        reverse=True
    )[:max_cols_to_try]

    for c in candidates:
        parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
        if parsed.notna().mean() >= 0.6:
            df[c] = parsed
            return df.sort_values(c), c

    return df, None

def detect_id_candidates(df: pd.DataFrame) -> list[str]:
    """
    Suggest grouping columns: team/player/match/opponent/competition etc.
    Heuristics: object columns with reasonable cardinality (not too high, not too low).
    """
    id_like_names = [
        "team", "opponent", "player", "player_name", "name", "athlete",
        "match", "fixture", "competition", "season", "phase",
        "position", "unit", "squad", "venue"
    ]
    candidates = []

    # name-based first
    for c in id_like_names:
        if c in df.columns and df[c].dtype == "object":
            candidates.append(c)

    # heuristic-based additions
    for c in df.columns:
        if df[c].dtype == "object":
            nunq = df[c].nunique(dropna=True)
            if 2 <= nunq <= 50 and c not in candidates:
                candidates.append(c)

    return candidates[:12]

def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def categorical_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == "object"]

def make_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            # try to parse numeric strings like "12.3", "45%", "1,234"
            s = df[c].astype(str).str.replace(",", "", regex=False)
            s = s.str.replace("%", "", regex=False)
            # only convert if it seems numeric-ish
            if s.str.match(r"^\s*-?\d+(\.\d+)?\s*$").mean() >= 0.7:
                df[c] = pd.to_numeric(s, errors="coerce")
    return df

def add_source_file(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = filename
    return df

# -------------------- UI --------------------
st.title("📊 Sport Auto Dashboard (Any Dataset)")
st.caption("Upload any sport dataset (team, player, match, training). The app auto-detects variables and builds charts.")

with st.sidebar:
    st.header("Upload")
    uploaded_files = st.file_uploader(
        "Upload file(s) (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    st.divider()
    st.header("Controls")
    max_vars = st.slider("Max numeric variables to chart", 3, 20, 10)
    max_categories = st.slider("Max category levels to display", 5, 30, 12)
    sample_rows = st.slider("Preview rows", 10, 200, 25)

if not uploaded_files:
    st.info("Upload at least one file to begin.")
    st.stop()

# -------------------- Load & combine --------------------
dfs = []
for uf in uploaded_files:
    d = read_any_file(uf)
    d = clean_cols(d)

    # trim strings
    for col in d.select_dtypes(include=["object"]).columns:
        d[col] = d[col].astype(str).str.strip()

    d = add_source_file(d, uf.name)
    dfs.append(d)

data = pd.concat(dfs, ignore_index=True, sort=False)
data = data.replace({pd.NA: np.nan})
data = make_numeric(data)
data, date_col = coerce_dates(data)

st.success(f"✅ Files loaded: {len(dfs)} • Rows: {len(data)} • Columns: {len(data.columns)}")

with st.expander("Preview / detected columns"):
    st.write("Detected columns:")
    st.code(", ".join(list(data.columns)))
    st.dataframe(data.head(sample_rows), use_container_width=True)

# -------------------- Detect variable types --------------------
num_cols = numeric_cols(data)
cat_cols = categorical_cols(data)

# Suggest grouping column
id_candidates = detect_id_candidates(data)
default_group = None
for c in ["team", "player", "player_name", "name", "match", "competition", "source_file"]:
    if c in data.columns:
        default_group = c
        break
if default_group is None:
    default_group = id_candidates[0] if id_candidates else "source_file"

# Sidebar selectors
with st.sidebar:
    group_col = st.selectbox(
        "Group by (optional)",
        options=["(none)"] + id_candidates + (["source_file"] if "source_file" in data.columns else []),
        index=(["(none)"] + id_candidates + (["source_file"] if "source_file" in data.columns else [])).index(default_group)
        if default_group in (["(none)"] + id_candidates + (["source_file"] if "source_file" in data.columns else []))
        else 0
    )

    if date_col:
        st.caption(f"Date detected: `{date_col}`")
    else:
        st.caption("No date column detected.")

# Apply optional group filter
filtered = data.copy()
if group_col != "(none)" and group_col in filtered.columns:
    values = sorted([v for v in filtered[group_col].dropna().unique().tolist()])[:500]
    chosen = st.multiselect(f"Filter {group_col}", options=values, default=values[: min(3, len(values))])
    if chosen:
        filtered = filtered[filtered[group_col].isin(chosen)]

# -------------------- Overview --------------------
st.subheader("Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(filtered):,}")
c2.metric("Numeric vars", f"{len(num_cols)}")
c3.metric("Categorical vars", f"{len(cat_cols)}")
c4.metric("Files", f"{filtered['source_file'].nunique() if 'source_file' in filtered.columns else 1}")

# Missingness
with st.expander("Data quality (missing values)"):
    miss = (filtered.isna().mean().sort_values(ascending=False) * 100).round(1)
    miss_df = miss.reset_index()
    miss_df.columns = ["column", "missing_%"]
    st.dataframe(miss_df, use_container_width=True)

# -------------------- Auto charts --------------------
st.subheader("Auto Charts")

if not num_cols:
    st.warning("No numeric columns detected. Add numeric metrics to generate charts.")
    st.stop()

# Pick top numeric variables by non-null coverage + variance
scores = []
for c in num_cols:
    coverage = filtered[c].notna().mean()
    var = np.nanvar(filtered[c].astype(float).values) if coverage > 0 else 0
    scores.append((c, coverage, var))
scores = sorted(scores, key=lambda x: (x[1], x[2]), reverse=True)
top_nums = [c for c, _, _ in scores[:max_vars]]

# 1) Time series / trend (if date exists)
if date_col:
    st.markdown("### Trends over time")
    # aggregate: mean for numeric
    if group_col != "(none)" and group_col in filtered.columns:
        for c in top_nums[: min(6, len(top_nums))]:
            agg = (
                filtered[[date_col, group_col, c]]
                .dropna(subset=[date_col])
                .groupby([group_col, date_col], as_index=False)[c]
                .mean()
            )
            fig = px.line(agg, x=date_col, y=c, color=group_col)
            fig.update_traces(line=dict(width=2.6))
            show(fig, f"{c} (mean) over time")
    else:
        for c in top_nums[: min(6, len(top_nums))]:
            agg = (
                filtered[[date_col, c]]
                .dropna(subset=[date_col])
                .groupby(date_col, as_index=False)[c]
                .mean()
            )
            fig = px.line(agg, x=date_col, y=c)
            fig.update_traces(line=dict(width=2.6))
            show(fig, f"{c} (mean) over time")

# 2) Distributions (histograms)
st.markdown("### Distributions")
dist_cols = st.columns(2)
for i, c in enumerate(top_nums[: min(6, len(top_nums))]):
    with dist_cols[i % 2]:
        fig = px.histogram(filtered, x=c, nbins=30)
        show(fig, f"{c} distribution")

# 3) Compare groups (box plots) if group chosen
if group_col != "(none)" and group_col in filtered.columns:
    st.markdown("### Group comparisons")
    box_cols = st.columns(2)
    for i, c in enumerate(top_nums[: min(6, len(top_nums))]):
        with box_cols[i % 2]:
            # limit categories for readability
            top_levels = (
                filtered[group_col].value_counts(dropna=True).head(max_categories).index.tolist()
            )
            d2 = filtered[filtered[group_col].isin(top_levels)]
            fig = px.box(d2, x=group_col, y=c, points="outliers")
            show(fig, f"{c} by {group_col}")

# 4) Scatter explorer (choose any 2 numeric)
st.markdown("### Scatter explorer")
x = st.selectbox("X axis", options=top_nums, index=0)
y = st.selectbox("Y axis", options=top_nums, index=min(1, len(top_nums) - 1))
color_opt = None
if group_col != "(none)" and group_col in filtered.columns:
    color_opt = group_col
fig = px.scatter(filtered, x=x, y=y, color=color_opt, trendline="ols" if len(filtered) >= 20 else None)
show(fig, f"{y} vs {x}")

# 5) Correlation heatmap (numeric only)
st.markdown("### Correlations")
corr_vars = top_nums[: min(12, len(top_nums))]
if len(corr_vars) >= 3:
    corr = filtered[corr_vars].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    show(fig, "Correlation heatmap", height=520)
else:
    st.info("Need at least 3 numeric variables for a correlation heatmap.")

# 6) Categorical frequency (if present)
if cat_cols:
    st.markdown("### Categorical frequencies")
    cat_choice = st.selectbox("Category column", options=cat_cols, index=0)
    vc = filtered[cat_choice].value_counts(dropna=True).head(max_categories).reset_index()
    vc.columns = [cat_choice, "count"]
    fig = px.bar(vc, x=cat_choice, y="count")
    fig.update_traces(marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.25)")
    show(fig, f"Top {len(vc)} values of {cat_choice}")
