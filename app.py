import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple, Optional

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# =========================
# Page & Title
# =========================
st.set_page_config(page_title="Autism Prediction (Adult)", layout="centered")
st.title("🧠 Autism Spectrum Disorder (ASD) Prediction — Adult Screener")
st.markdown(
    "This tool estimates the probability of ASD **for adults** using clinical/demographic inputs. "
    "It is a screening aid, **not** a diagnosis. Always consult qualified clinicians."
)

# =========================
# Load Pipeline
# =========================
@st.cache_resource(show_spinner=False)
def load_pipeline(path: str = "autism_pipeline.pkl"):
    return joblib.load(path)

pipeline = load_pipeline()

# =========================
# Category hints from OneHotEncoder (if present)
# =========================
CATEGORY_HINTS: dict[str, List[str]] = {}
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if isinstance(step, ColumnTransformer):
                for _, trans, cols in step.transformers_:
                    if isinstance(trans, OneHotEncoder) and hasattr(trans, "categories_"):
                        for col, cats in zip(cols, trans.categories_):
                            CATEGORY_HINTS[str(col)] = [str(c) for c in cats]
except Exception:
    pass

# Fallback option lists
ETHNICITIES_FALLBACK = [
    'Asian', 'Black', 'Hispanic', 'Latino', 'Middle Eastern', 'Mixed',
    'Native Indian', 'Others', 'PaciFica', 'South Asian', 'White European'
]
RELATIONS_FALLBACK = ["Self", "Parent", "Health Care Professional", "Relative", "Others"]
GENDERS_FALLBACK = ["male", "female"]
AGEBANDS_FALLBACK = ["18 and more"]

# =========================
# Raw required column names referenced inside the pipeline (by name)
# =========================

def discover_raw_required_cols(pipe) -> List[str]:
    required: List[str] = []
    try:
        from sklearn.compose import ColumnTransformer
        if hasattr(pipe, "named_steps"):
            for step in pipe.named_steps.values():
                if isinstance(step, ColumnTransformer):
                    for _, trans, cols in step.transformers_:
                        if isinstance(cols, str):
                            required.append(cols)
                        elif hasattr(cols, "__iter__") and not isinstance(cols, str):
                            for c in cols:
                                if isinstance(c, str):
                                    required.append(c)
    except Exception:
        pass
    # unique, keep order
    seen = set()
    out: List[str] = []
    for c in required:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

RAW_REQUIRED = discover_raw_required_cols(pipeline)

# =========================
# Input schema — start with adult schema, then union with pipeline's raw requirements
# =========================
EXPECTED_BASE = [
    "result",   # screening outcome/score
    "ethnicity",
    "austim",
    "jundice",
    "relation",
    "age",
    "gender",
]
ALL_INPUT_COLS = list(dict.fromkeys([*EXPECTED_BASE, *RAW_REQUIRED]))

# =========================
# Sidebar Settings
# =========================
st.sidebar.header("Settings")
thr = st.sidebar.slider("Decision threshold (ASD if probability ≥ threshold)", 0.0, 1.0, 0.50, 0.01)
uncert_band = st.sidebar.slider("Inconclusive band (± around threshold)", 0.0, 0.20, 0.05, 0.01)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Model notes")
st.sidebar.caption("Trained for **adults**. Inputs outside the training distribution may reduce reliability.")

# =========================
# Helpers
# =========================

def predict_proba_safe(pipe, X: pd.DataFrame) -> float:
    """Return the positive-class probability robustly. Falls back to label if needed."""
    # Try standard predict_proba first
    try:
        proba = pipe.predict_proba(X)
    except Exception:
        try:
            pred = pipe.predict(X)
            return float(np.ravel(pred)[0])
        except Exception:
            return 0.0

    # Decide which column corresponds to the positive class
    pos_idx = 1  # default for [neg, pos]
    try:
        est = None
        if hasattr(pipe, "named_steps") and pipe.named_steps:
            est = list(pipe.named_steps.values())[-1]
        elif hasattr(pipe, "steps") and pipe.steps:
            est = pipe.steps[-1][1]
        classes = getattr(est, "classes_", None)
        if classes is not None:
            classes_list = [str(c) for c in list(classes)]
            if "1" in classes_list:
                pos_idx = classes_list.index("1")
            else:
                for i, c in enumerate(classes_list):
                    if c.lower() in {"asd", "yes", "true", "positive", "pos"}:
                        pos_idx = i
                        break
    except Exception:
        pass

    try:
        return float(proba[0, pos_idx])
    except Exception:
        return float(np.ravel(proba)[pos_idx])


def warn_on_unseen_categories(pipe, X: pd.DataFrame):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        if hasattr(pipe, "named_steps"):
            for step in pipe.named_steps.values():
                if isinstance(step, ColumnTransformer):
                    for _, trans, cols in step.transformers_:
                        if isinstance(trans, OneHotEncoder) and hasattr(trans, "categories_"):
                            for col, cats in zip(cols, trans.categories_):
                                c = str(col)
                                if c in X.columns:
                                    val = X[c].iloc[0]
                                    if (val not in cats) and (not pd.isna(val)):
                                        st.warning(f"Value '{val}' for **{c}** was not seen during training; predictions may be less reliable.")
    except Exception:
        return

# Simple session log for drift badge
if "_input_logs" not in st.session_state:
    st.session_state._input_logs = []  # list of dict rows


def log_input(row: pd.Series):
    st.session_state._input_logs.append(row.to_dict())
    # keep it light
    if len(st.session_state._input_logs) > 500:
        st.session_state._input_logs = st.session_state._input_logs[-500:]


def drift_badge(row: pd.Series):
    issues = []
    # unseen categories
    for c, cats in CATEGORY_HINTS.items():
        if c in row.index and str(row[c]) not in set(cats):
            issues.append(f"{c}: '{row[c]}' unseen")
    # simple numeric sanity
    if "age" in row.index:
        try:
            a = float(row["age"])
            if (a < 18) or (a > 110):
                issues.append(f"age={a} out-of-range for adult")
        except Exception:
            pass
    if issues:
        st.warning("⚠️ Potential input drift: " + "; ".join(issues))
    else:
        st.success("✅ Input within training distribution (best-effort check)")

# =========================
# Dynamic Input Form
# =========================

def get_user_input() -> Tuple[pd.DataFrame, bool]:
    with st.form("user_form"):
        payload: dict[str, object] = {}
        for col in ALL_INPUT_COLS:
            key = str(col)
            lkey = key.lower()

            if lkey == "result":
                payload[key] = st.slider("Screening Result (AQ-10 score)", min_value=0, max_value=10, value=1, step=1)
            elif lkey in {"austim", "jundice"}:
                yn = st.selectbox(key, ["no", "yes"])
                payload[key] = 1 if yn == "yes" else 0
            elif lkey == "ethnicity":
                opts = CATEGORY_HINTS.get(key, sorted(ETHNICITIES_FALLBACK))
                payload[key] = st.selectbox("Ethnicity", opts)
            elif lkey == "relation":
                opts = CATEGORY_HINTS.get(key, RELATIONS_FALLBACK)
                payload[key] = st.selectbox("Relation to the person being tested", opts)
            elif lkey == "age":
                payload[key] = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
            elif lkey in {"gender", "sex"}:
                opts = CATEGORY_HINTS.get(key, GENDERS_FALLBACK)
                label = "Gender" if lkey == "gender" else "Sex"
                payload[key] = st.selectbox(label, opts)
            elif lkey == "age_desc":
                opts = CATEGORY_HINTS.get(key, AGEBANDS_FALLBACK)
                payload[key] = st.selectbox("Age Group", opts)
            else:
                opts = CATEGORY_HINTS.get(key)
                payload[key] = st.selectbox(key, opts) if opts else st.text_input(key)

        submitted = st.form_submit_button("Predict")

    # --- compatibility shims ---
    if ("age_desc" in ALL_INPUT_COLS) and ("age_desc" not in payload):
        payload["age_desc"] = "18 and more"
    if ("sex" in ALL_INPUT_COLS) and ("sex" not in payload) and ("gender" in payload):
        payload["sex"] = str(payload["gender"])
    if ("gender" in ALL_INPUT_COLS) and ("gender" not in payload) and ("sex" in payload):
        payload["gender"] = str(payload["sex"])

    # Coerce types
    for k in list(payload.keys()):
        lk = k.lower()
        if lk in {"result", "austim", "jundice"}:
            payload[k] = int(payload[k]) if str(payload[k]).strip() != "" else 0
        elif lk == "age":
            payload[k] = float(payload[k]) if str(payload[k]).strip() != "" else np.nan
        else:
            payload[k] = str(payload[k])

    df = pd.DataFrame([payload], columns=ALL_INPUT_COLS)
    return df, submitted

# =========================
# TABS
# =========================

tab_predict, tab_eval, tab_about = st.tabs(["Predict", "Evaluation", "About"]) 

# ----------------- PREDICT TAB -----------------
with tab_predict:
    input_df, submitted = get_user_input()

    if submitted:
        try:
            input_df = input_df.reindex(columns=ALL_INPUT_COLS)
            warn_on_unseen_categories(pipeline, input_df)
            proba_pos = predict_proba_safe(pipeline, input_df)
            label = int(proba_pos >= thr)

            st.subheader("🔍 Prediction")
            st.metric("Predicted probability (ASD)", f"{proba_pos:.2f}")

            if abs(proba_pos - thr) < uncert_band:
                st.warning("Result **inconclusive** near threshold — consider clinical follow-up.")

            if label == 1:
                st.error("🔴 Likely Autism Spectrum Disorder")
            else:
                st.success("🟢 Not Likely Autism Spectrum Disorder")

            st.caption(f"Decision threshold: {thr:.2f} | Inconclusive band: ±{uncert_band:.2f}")

            # Drift badge + log
            drift_badge(input_df.iloc[0])
            log_input(input_df.iloc[0])

            if show_debug:
                with st.expander("See input & dtypes sent to the model"):
                    st.write(input_df)
                    st.write(input_df.dtypes)
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            if show_debug:
                with st.expander("Debug payload"):
                    st.write(input_df)
                    st.write(input_df.dtypes)

    st.markdown("---")
    st.subheader("📦 Batch Predictions (CSV)")
    st.caption(f"Upload a CSV with the columns the **model expects**: {', '.join(map(str, ALL_INPUT_COLS))}")

    # Provide a template
    _template = pd.DataFrame(columns=ALL_INPUT_COLS).to_csv(index=False)
    st.download_button(label="Download CSV template", data=_template, file_name="asd_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            missing = [c for c in ALL_INPUT_COLS if c not in df_in.columns]
            if missing:
                st.error(f"Your file is missing required columns: {missing}")
            else:
                df_in = df_in[ALL_INPUT_COLS].copy()

                # Alias and defaults
                if ("sex" in df_in.columns) and ("gender" not in df_in.columns) and ("gender" in ALL_INPUT_COLS):
                    df_in["gender"] = df_in["sex"].astype(str)
                if ("gender" in df_in.columns) and ("sex" not in df_in.columns) and ("sex" in ALL_INPUT_COLS):
                    df_in["sex"] = df_in["gender"].astype(str)
                if ("age_desc" in ALL_INPUT_COLS) and ("age_desc" not in df_in.columns):
                    df_in["age_desc"] = "18 and more"

                # Coerce
                for c in ALL_INPUT_COLS:
                    lc = str(c).lower()
                    if lc in {"result", "austim", "jundice"}:
                        df_in[c] = pd.to_numeric(df_in[c], errors="coerce").fillna(0).astype(int)
                    elif lc == "age":
                        df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
                    else:
                        df_in[c] = df_in[c].astype(str)

                # Predict
                try:
                    proba = pipeline.predict_proba(df_in)
                    probas = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else np.ravel(proba)
                except Exception:
                    probas = pipeline.predict(df_in).astype(float)

                labels = (probas >= thr).astype(int)
                out = df_in.copy()
                out["prob_asd"] = probas
                out["pred_label"] = labels
                out["inconclusive"] = (np.abs(out["prob_asd"] - thr) < uncert_band)

                st.success("Batch prediction complete.")
                st.dataframe(out.head(100))

                csv_out = out.to_csv(index=False)
                st.download_button(label="Download results.csv", data=csv_out, file_name="asd_batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction error: {e}")

# ----------------- EVALUATION TAB -----------------
with tab_eval:
    st.subheader("Model evaluation on a labelled dataset")
    st.caption("Upload a CSV that includes the required input columns and a ground-truth label column.")

    LABEL_CANDIDATES = ["class", "Class", "ASD", "asd", "label", "target", "Class/ASD", "class/ASD", "class/Asd"]

    val_file = st.file_uploader("Upload labelled CSV", type=["csv"], accept_multiple_files=False, key="eval_upl")
    if val_file is not None:
        try:
            df = pd.read_csv(val_file)
            # Find label column
            label_col: Optional[str] = None
            lowermap = {c.lower(): c for c in df.columns}
            for cand in LABEL_CANDIDATES:
                if cand.lower() in lowermap:
                    label_col = lowermap[cand.lower()]
                    break
            if label_col is None:
                st.error("No label column found. Expected one of: " + ", ".join(LABEL_CANDIDATES))
            else:
                missing = [c for c in ALL_INPUT_COLS if c not in df.columns]
                if missing:
                    st.error(f"Your file is missing required columns: {missing}")
                else:
                    X = df[ALL_INPUT_COLS].copy()
                    # Map y to {0,1}
                    y_raw = df[label_col].astype(str).str.strip()
                    y = y_raw.str.lower().isin(["1", "yes", "asd", "true", "y"]).astype(int)

                    # Type coercion similar to predict path
                    for c in ALL_INPUT_COLS:
                        lc = str(c).lower()
                        if lc in {"result", "austim", "jundice"}:
                            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)
                        elif lc == "age":
                            X[c] = pd.to_numeric(X[c], errors="coerce")
                        else:
                            X[c] = X[c].astype(str)

                    # Predict probabilities
                    try:
                        proba = pipeline.predict_proba(X)
                        p = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else np.ravel(proba)
                    except Exception:
                        p = pipeline.predict(X).astype(float)

                    yhat = (p >= thr).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
                    sens = recall_score(y, yhat)  # TPR
                    spec = tn / (tn + fp) if (tn + fp) else 0.0
                    prec = precision_score(y, yhat) if (tp + fp) else 0.0
                    f1 = f1_score(y, yhat) if (tp + fp) and (tp + fn) else 0.0
                    roc_auc = roc_auc_score(y, p)
                    pr_auc = average_precision_score(y, p)

                    st.markdown("**Key metrics at current threshold**")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Sensitivity", f"{sens:.2f}")
                    c2.metric("Specificity", f"{spec:.2f}")
                    c3.metric("Precision", f"{prec:.2f}")
                    c4.metric("F1-score", f"{f1:.2f}")
                    c5.metric("ROC-AUC", f"{roc_auc:.2f}")
                    st.caption(f"PR-AUC: {pr_auc:.2f}")

                    # Confusion matrix table
                    st.markdown("**Confusion matrix**")
                    cm_df = pd.DataFrame([[tn, fp], [fn, tp]], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
                    st.table(cm_df)

                    # ROC curve
                    fpr, tpr, _ = roc_curve(y, p)
                    fig1 = plt.figure()
                    plt.plot(fpr, tpr)
                    plt.plot([0, 1], [0, 1], linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC curve')
                    st.pyplot(fig1)

                    # Precision-Recall curve
                    precs, recalls, _ = precision_recall_curve(y, p)
                    fig2 = plt.figure()
                    plt.plot(recalls, precs)
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall curve')
                    st.pyplot(fig2)

                    # Calibration curve
                    prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
                    fig3 = plt.figure()
                    plt.plot(prob_pred, prob_true)
                    plt.plot([0, 1], [0, 1], linestyle='--')
                    plt.xlabel('Mean predicted probability')
                    plt.ylabel('Fraction of positives')
                    plt.title('Calibration curve (10 bins)')
                    st.pyplot(fig3)

        except Exception as e:
            st.error(f"Evaluation error: {e}")

# ----------------- ABOUT TAB -----------------
# ----------------- ABOUT TAB -----------------
with tab_about:
    st.subheader("About this app")
    st.markdown("""
**Intended use.** Educational screening support for adults. Not a diagnostic device.

**Inputs.** `result` (AQ-10 score or binary), `ethnicity`, previous autism diagnosis (`austim`), `jundice` at birth, `relation`, `age`, and `gender`/`sex`.

**Output.** Probability of ASD with a configurable decision threshold and an inconclusive zone.

**Limitations.** Predictions depend on training data coverage. Unseen categories or out-of-range values may reduce reliability.

**Ethics & safety.** Always seek professional clinical assessment when concerned about ASD symptoms.
""")
    st.markdown("**Ethics & safety** — Always seek professional clinical assessment when concerned about ASD symptoms.")

st.markdown("---")
st.caption("© 2025 ASD Screening Aid — Educational use only.")