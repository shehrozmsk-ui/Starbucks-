from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "starbucks.csv"
MODEL_PATH = ROOT / "decision_tree_model.joblib"
META_PATH = ROOT / "model_metadata.json"
TREE_PATH = ROOT / "outputs" / "decision_tree.png"

st.set_page_config(
    page_title="Starbucks Nutrition & ML Studio",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0e1117; color: #f5f7fb; }
    [data-testid="stSidebar"] { background: #151923; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(120deg, #173c31 0%, #16242b 55%, #25202b 100%);
        border: 1px solid rgba(255,255,255,.10);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: #ffffff; }
    .hero p { margin: .45rem 0 0 0; color: #d7e4df; }
    div.stButton > button[kind="primary"] {
        background: #e11d48; color: white; border: 0; border-radius: 10px;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: #1b202a; border: 1px solid #303744;
        padding: 14px 16px; border-radius: 12px;
    }
    .small-note { color: #aeb7c4; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def normalize_columns(frame):
    frame = frame.copy()
    frame.columns = (
        frame.columns.astype(str).str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    return frame

@st.cache_data(show_spinner=False)
def load_csv(path, modified_time):
    data = pd.read_csv(path)
    data = normalize_columns(data)
    data = data.replace(r"^\s*-\s*$", np.nan, regex=True)
    data = data.replace(r"^\s*$", np.nan, regex=True)
    return data

@st.cache_resource(show_spinner=False)
def load_model(path, modified_time):
    return joblib.load(path)

def get_name_column(frame):
    candidates = ["beverage", "item", "name", "drink", "unnamed_0"]
    return next((c for c in candidates if c in frame.columns), None)

def get_calorie_column(frame, metadata):
    preferred = metadata.get("calorie_column")
    if preferred in frame.columns:
        return preferred
    return next((c for c in frame.columns if "calorie" in c.lower()), None)

def category_from_calories(value):
    if pd.isna(value):
        return "Unknown"
    if value < 100:
        return "Low"
    if value < 200:
        return "Medium"
    return "High"

def pretty_name(value):
    return str(value).replace("_", " ").strip().title()

# Load data and supporting files
st.sidebar.markdown("## ☕ Starbucks ML Studio")
st.sidebar.caption("Nutrition exploration • Decision Tree prediction")

if not CSV_PATH.exists():
    st.error("The file `starbucks.csv` was not found in this repository.")
    st.info("Upload `starbucks.csv` to the same GitHub repository folder as `app.py`.")
    st.stop()

try:
    df = load_csv(str(CSV_PATH), CSV_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read starbucks.csv: {exc}")
    st.stop()

metadata = {}
if META_PATH.exists():
    try:
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.warning("model_metadata.json could not be read. Some model details may be unavailable.")

model = None
model_error = None
if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as exc:
        model_error = str(exc)

name_col = get_name_column(df)
calorie_col = get_calorie_column(df, metadata)

if calorie_col:
    df[calorie_col] = pd.to_numeric(df[calorie_col], errors="coerce")

feature_cols = metadata.get("feature_columns", [])
numeric_features = metadata.get("numeric_features", [])
categorical_features = metadata.get("categorical_features", [])

# Use the saved training metadata as the source of truth for model inputs.
if model is not None and not feature_cols:
    st.warning("Model metadata is missing feature_columns. Re-run the training notebook and upload model_metadata.json.")
    model = None

if model is not None:
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        st.warning("The CSV does not contain every feature expected by the saved model: " + ", ".join(missing_features))
        model = None

# Sidebar dataset controls
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.write(f"**Rows:** {len(df):,}")
st.sidebar.write(f"**Columns:** {df.shape[1]:,}")
if calorie_col:
    st.sidebar.write(f"**Calories field:** `{calorie_col}`")
st.sidebar.caption("This app uses the CSV stored in GitHub. Visitors do not need to upload a file.")

st.markdown(
    """
    <div class="hero">
      <h1>Starbucks Nutrition & ML Studio</h1>
      <p>Explore drink nutrition, compare menu items, and estimate calorie categories with a trained Decision Tree model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        "The menu explorer is available, but ML predictions are disabled until a compatible "
        "`decision_tree_model.joblib` and `model_metadata.json` are uploaded beside `app.py`."
    )
    if model_error:
        st.caption(f"Model loading detail: {model_error}")

overview_tab, menu_tab, predict_tab, model_tab = st.tabs(
    ["📊 Overview", "☕ Menu Explorer", "🌳 ML Prediction", "🧠 Model Details"]
)

with overview_tab:
    valid_calories = df[calorie_col].dropna() if calorie_col else pd.Series(dtype=float)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Menu records", f"{len(df):,}")
    c2.metric("Nutrition fields", f"{len(df.columns):,}")
    c3.metric("Average calories", f"{valid_calories.mean():.0f}" if len(valid_calories) else "N/A")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Dataset preview")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Missing values by field")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if len(missing):
            st.bar_chart(missing)
        else:
            st.success("No missing values detected.")
    with right:
        st.markdown("#### Calorie distribution")
        if len(valid_calories):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(valid_calories, bins=15, edgecolor="white", alpha=0.85)
            ax.set_xlabel("Calories")
            ax.set_ylabel("Number of menu records")
            ax.set_title("Calories across dataset")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("A usable calories column was not found.")

with menu_tab:
    st.subheader("Find a drink")
    if name_col:
        names = df[name_col].fillna("Unnamed menu item").astype(str)
    else:
        names = pd.Series([f"Menu item {i+1}" for i in range(len(df))], index=df.index)

    search = st.text_input("Search menu items", placeholder="Type a drink name or keyword")
    view = df.copy()
    view["_display_name"] = names
    if search:
        view = view[view["_display_name"].str.contains(search, case=False, na=False)]

    if calorie_col:
        available_calories = df[calorie_col].dropna()
        if len(available_calories):
            min_cal = int(max(0, np.floor(available_calories.min())))
            max_cal = int(np.ceil(available_calories.max()))
            if min_cal < max_cal:
                selected_range = st.slider(
                    "Filter by calories",
                    min_value=min_cal,
                    max_value=max_cal,
                    value=(min_cal, max_cal),
                )
                view = view[
                    view[calorie_col].isna()
                    | view[calorie_col].between(selected_range[0], selected_range[1])
                ]

    display_cols = [c for c in view.columns if c != "_display_name"]
    st.caption(f"Showing {len(view):,} matching records")
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True)

    if len(view):
        st.markdown("#### Selected item nutrition")
        labels = view["_display_name"].tolist()
        selected_label = st.selectbox("Choose a menu item", labels, key="menu_selected")
        selected_index = view.index[view["_display_name"] == selected_label][0]
        selected_row = df.loc[selected_index]

        metric_cols = st.columns(4)
        metric_specs = []
        if calorie_col:
            metric_specs.append(("Calories", selected_row.get(calorie_col, np.nan)))
        for c in df.select_dtypes(include=np.number).columns:
            if c != calorie_col and len(metric_specs) < 4:
                metric_specs.append((pretty_name(c), selected_row.get(c, np.nan)))
        for col, (label, value) in zip(metric_cols, metric_specs):
            col.metric(label, "N/A" if pd.isna(value) else f"{value:g}")

        if calorie_col and pd.notna(selected_row.get(calorie_col)):
            actual_level = category_from_calories(selected_row[calorie_col])
            st.write(f"**Category from recorded calories:** {actual_level}")
            st.caption("Project thresholds: Low = 0–99, Medium = 100–199, High = 200+ calories.")

with predict_tab:
    st.subheader("Decision Tree calorie-category prediction")
    st.write(
        "Choose an existing menu item or enter nutrition values to estimate its calorie category. "
        "Calories themselves are intentionally not used as an input feature."
    )

    if model is None:
        st.error("Prediction is not ready. Upload the matching model and metadata files to GitHub.")
    else:
        mode = st.radio(
            "Prediction method",
            ["Use an item from the dataset", "Enter nutrition values"],
            horizontal=True,
        )

        if mode == "Use an item from the dataset":
            choices = df.index.tolist()
            if name_col:
                labels = {
                    i: str(df.loc[i, name_col]) if pd.notna(df.loc[i, name_col]) else f"Menu item {i+1}"
                    for i in choices
                }
            else:
                labels = {i: f"Menu item {i+1}" for i in choices}

            chosen_index = st.selectbox(
                "Select drink",
                choices,
                format_func=lambda i: labels[i],
            )
            row = df.loc[chosen_index]
            input_values = {c: row.get(c, np.nan) for c in feature_cols}
            input_df = pd.DataFrame([input_values], columns=feature_cols)

            with st.expander("View nutrition values supplied to the model"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

            run_prediction = st.button("Predict calorie category", type="primary", use_container_width=True)
        else:
            st.markdown("##### Enter the available nutrition information")
            input_values = {}
            with st.form("custom_prediction_form"):
                cols = st.columns(2)
                for i, feature in enumerate(feature_cols):
                    container = cols[i % 2]
                    if feature in numeric_features:
                        series = pd.to_numeric(df[feature], errors="coerce").dropna()
                        default = float(series.median()) if len(series) else 0.0
                        minimum = float(series.min()) if len(series) else 0.0
                        maximum = float(series.max()) if len(series) else max(default, 1.0)
                        step = 1.0 if abs(default) >= 10 else 0.1
                        input_values[feature] = container.number_input(
                            pretty_name(feature),
                            min_value=float(minimum),
                            max_value=float(maximum) if maximum > minimum else float(minimum + 1),
                            value=float(np.clip(default, minimum, maximum if maximum > minimum else minimum + 1)),
                            step=step,
                            help="Range is based on the available dataset.",
                        )
                    else:
                        options = sorted(df[feature].dropna().astype(str).unique().tolist())
                        if not options:
                            options = ["Unknown"]
                        input_values[feature] = container.selectbox(pretty_name(feature), options)

                submitted = st.form_submit_button(
                    "Predict calorie category",
                    type="primary",
                    use_container_width=True,
                )
            input_df = pd.DataFrame([input_values], columns=feature_cols)
            run_prediction = submitted

        if run_prediction:
            try:
                prediction = str(model.predict(input_df)[0])
                st.markdown("### Prediction result")
                result_col, detail_col = st.columns([1, 2])
                with result_col:
                    st.metric("Estimated calorie category", prediction)
                with detail_col:
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_df)[0]
                        classes = model.named_steps["classifier"].classes_
                        prob_df = pd.DataFrame({
                            "Category": classes,
                            "Model probability": probabilities,
                        }).sort_values("Model probability", ascending=False)
                        st.dataframe(
                            prob_df.style.format({"Model probability": "{:.1%}"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.bar_chart(prob_df.set_index("Category"))
                st.caption(
                    "This is a model estimate, not an official Starbucks nutrition claim. "
                    "The calorie bands are project-defined, and results depend on the small dataset."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Make sure model_metadata.json and the trained model were created by the same training notebook.")

with model_tab:
    st.subheader("Model information")
    if metadata:
        a, b = st.columns(2)
        with a:
            st.write("**Target column**")
            st.code(str(metadata.get("target_column", "Not specified")))
            st.write("**Target definition**")
            st.write(metadata.get("calorie_thresholds", {
                "Low": "0–99 calories",
                "Medium": "100–199 calories",
                "High": "200+ calories",
            }))
        with b:
            accuracy = metadata.get("accuracy")
            st.write("**Held-out test accuracy**")
            st.metric("Accuracy", f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else "Not recorded")
            st.write("**Model type**")
            st.write("Decision Tree Classifier")

        st.markdown("#### Features used by the model")
        st.write(", ".join(feature_cols) if feature_cols else "Feature metadata unavailable.")
    else:
        st.info("Upload model_metadata.json to show the training configuration.")

    if TREE_PATH.exists():
        st.markdown("#### Decision Tree visualization")
        st.image(str(TREE_PATH), use_container_width=True)
    else:
        st.caption("Optional: upload `outputs/decision_tree.png` to display the tree diagram.")

    st.markdown("---")
    st.caption(
        "Project note: calorie categories are generated from the dataset's recorded calories "
        "(Low: under 100; Medium: 100–199; High: 200 or more). Model accuracy is only an estimate "
        "on a held-out sample and may be unstable with a small dataset."
    )

st.markdown("---")
st.caption("Starbucks Nutrition & ML Studio • Educational project • Verify nutrition details against current official sources.")
