import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Biomarker Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Design system
# Palette: ink #10242E · lab-paper #F2F6F7 · teal #0B6E75 · coral #C2483D
#          mist #DCE6E9 · amber #B7791F
# Type:    Bricolage Grotesque (headings) + Public Sans (body)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');

:root{
  --ink:#10242E; --paper:#F2F6F7; --teal:#0B6E75; --teal-soft:#DDEFF0;
  --coral:#C2483D; --coral-soft:#F8E4E1; --mist:#DCE6E9; --muted:#5B7079;
  --amber:#B7791F;
}

html, body, [class*="css"], .stApp { font-family:'Public Sans', system-ui, sans-serif; color:var(--ink); }
.stApp { background:var(--paper); }
h1,h2,h3,h4 { font-family:'Bricolage Grotesque', 'Public Sans', sans-serif !important; letter-spacing:-0.01em; color:var(--ink); }

/* tighten default Streamlit chrome */
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:1100px; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid var(--mist); }
section[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
section[data-testid="stSidebar"] details {
  border:1px solid var(--mist); border-radius:10px; background:var(--paper); margin-bottom:.6rem;
}
section[data-testid="stSidebar"] summary { font-weight:600; }

/* Hero */
.hero { display:flex; gap:1.4rem; align-items:flex-end; justify-content:space-between; flex-wrap:wrap;
        border-bottom:2px solid var(--ink); padding-bottom:1.1rem; margin-bottom:1.4rem; }
.hero h1 { font-size:2.5rem; line-height:1.05; margin:0; font-weight:700; }
.hero p  { margin:.5rem 0 0; color:var(--muted); max-width:46ch; font-size:1.02rem; line-height:1.5; }
.hero .stamp { border:1.5px solid var(--teal); color:var(--teal); border-radius:8px; padding:.45rem .8rem;
               font-weight:600; font-size:.88rem; background:var(--teal-soft); }

/* Value tiles */
.tiles { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:.7rem; }
.tile { background:#fff; border:1px solid var(--mist); border-radius:10px; padding:.75rem .9rem; }
.tile .k { font-size:.82rem; color:var(--muted); font-weight:500; }
.tile .v { font-family:'Bricolage Grotesque',sans-serif; font-size:1.55rem; font-weight:700; line-height:1.2; }
.tile .u { font-size:.78rem; color:var(--muted); margin-left:.25rem; font-weight:400; font-family:'Public Sans',sans-serif; }

/* Result card */
.result { border-radius:14px; padding:1.5rem 1.6rem; border:1.5px solid; margin:.4rem 0 1.2rem; }
.result.control { background:var(--teal-soft); border-color:var(--teal); }
.result.patient { background:var(--coral-soft); border-color:var(--coral); }
.result .tag { font-weight:600; font-size:.9rem; }
.result.control .tag { color:var(--teal); } .result.patient .tag { color:var(--coral); }
.result .big { font-family:'Bricolage Grotesque',sans-serif; font-size:2.3rem; font-weight:700; line-height:1.1; margin:.2rem 0 .3rem; }
.result .sub { color:var(--ink); opacity:.8; font-size:.98rem; }

/* Probability bars */
.pbar { margin:.9rem 0; }
.pbar .row { display:flex; justify-content:space-between; font-weight:600; margin-bottom:.3rem; }
.pbar .track { height:14px; background:var(--mist); border-radius:99px; overflow:hidden; }
.pbar .fill { height:100%; border-radius:99px; transition:width .6s ease; }
.fill.c { background:var(--teal); } .fill.p { background:var(--coral); }

/* Empty state */
.empty { background:#fff; border:1.5px dashed var(--mist); border-radius:14px; padding:2rem; text-align:center; color:var(--muted); }
.empty b { color:var(--ink); font-family:'Bricolage Grotesque',sans-serif; font-size:1.2rem; display:block; margin-bottom:.3rem; }

/* Note */
.note { font-size:.86rem; color:var(--muted); border-left:3px solid var(--amber); padding:.2rem .8rem; margin-top:1rem; line-height:1.5; }

/* Primary button */
div.stButton > button[kind="primary"], div.stButton > button {
  border-radius:10px; font-weight:600; padding:.65rem 1.2rem;
}
div.stButton > button[kind="primary"] { background:var(--teal); border:1px solid var(--teal); color:#fff; }
div.stButton > button[kind="primary"]:hover { background:#095a60; border-color:#095a60; }
button:focus-visible, input:focus-visible { outline:3px solid var(--amber) !important; outline-offset:2px; }

/* Tabs */
button[data-baseweb="tab"] { font-weight:600; }

@media (prefers-reduced-motion: reduce){ .pbar .fill{ transition:none; } }
@media (max-width:640px){ .hero h1{ font-size:1.9rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    with open("model.pkl", "rb") as f:
        mdl = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scl = pickle.load(f)
    return mdl, scl


try:
    model, scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Can't find `{e.filename}`. Put `model.pkl` and `scaler.pkl` in the same "
        "folder as this app, then reload the page."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Defaults + reset handling
# ----------------------------------------------------------------------------
DEFAULTS = {
    "age": 50, "bmi": 25.0, "glucose": 90.0, "insulin": 10.0, "homa": 2.0,
    "leptin": 20.0, "adiponectin": 10.0, "resistin": 10.0, "mcp_1": 400.0,
    "auto_homa": True,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_inputs():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.pop("result", None)


# ----------------------------------------------------------------------------
# Sidebar: grouped inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Patient values")
    st.caption("Enter the measurements, then press **Run classification**.")

    with st.expander("Body", expanded=True):
        st.slider("Age (years)", 10, 100, key="age", step=1)
        st.slider("BMI (kg/m²)", 10.0, 50.0, key="bmi", step=0.1)

    with st.expander("Blood sugar", expanded=True):
        st.number_input("Glucose (mg/dL)", 10.0, 300.0, key="glucose", step=1.0)
        st.number_input("Insulin (µU/mL)", 1.0, 100.0, key="insulin", step=0.5)
        st.checkbox(
            "Calculate HOMA from glucose and insulin",
            key="auto_homa",
            help="HOMA-IR = glucose × insulin ÷ 405. Turn off to enter your own value.",
        )
        if st.session_state["auto_homa"]:
            calc = st.session_state["glucose"] * st.session_state["insulin"] / 405
            st.session_state["homa"] = float(np.clip(calc, 0.1, 30.0))
        st.number_input(
            "HOMA", 0.1, 30.0, key="homa", step=0.1, format="%.2f",
            disabled=st.session_state["auto_homa"],
        )

    with st.expander("Hormones and inflammation", expanded=False):
        st.number_input("Leptin (ng/mL)", 1.0, 120.0, key="leptin", step=0.5)
        st.number_input("Adiponectin (µg/mL)", 1.0, 40.0, key="adiponectin", step=0.5)
        st.number_input("Resistin (ng/mL)", 1.0, 100.0, key="resistin", step=0.5)
        st.number_input("MCP-1 (pg/mL)", 10.0, 2000.0, key="mcp_1", step=10.0)

    st.button("Reset to defaults", on_click=reset_inputs, use_container_width=True)

# ----------------------------------------------------------------------------
# Collect inputs
# ----------------------------------------------------------------------------
FEATURES = [
    ("age", "Age", "yrs", "{:.0f}"),
    ("bmi", "BMI", "kg/m²", "{:.1f}"),
    ("glucose", "Glucose", "mg/dL", "{:.0f}"),
    ("insulin", "Insulin", "µU/mL", "{:.1f}"),
    ("homa", "HOMA", "", "{:.2f}"),
    ("leptin", "Leptin", "ng/mL", "{:.1f}"),
    ("adiponectin", "Adiponectin", "µg/mL", "{:.1f}"),
    ("resistin", "Resistin", "ng/mL", "{:.1f}"),
    ("mcp_1", "MCP-1", "pg/mL", "{:.0f}"),
]
input_data = pd.DataFrame([{k: st.session_state[k] for k, *_ in FEATURES}])

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div>
    <h1>Biomarker screening</h1>
    <p>Compares nine blood and body measurements against a trained
       K-Nearest Neighbors model and estimates whether the profile looks
       more like the Control or the Patient group.</p>
  </div>
  <div class="stamp">KNN classifier · 9 biomarkers</div>
</div>
""",
    unsafe_allow_html=True,
)

tab_result, tab_inputs, tab_about = st.tabs(["Result", "Your values", "About"])

# ----------------------------------------------------------------------------
# Inputs tab
# ----------------------------------------------------------------------------
with tab_inputs:
    tiles = "".join(
        f'<div class="tile"><div class="k">{label}</div>'
        f'<div class="v">{fmt.format(st.session_state[key])}'
        f'<span class="u">{unit}</span></div></div>'
        for key, label, unit, fmt in FEATURES
    )
    st.markdown(f'<div class="tiles">{tiles}</div>', unsafe_allow_html=True)
    with st.expander("View as table"):
        st.dataframe(input_data, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# Result tab
# ----------------------------------------------------------------------------
LABELS = {1: ("Control group", "control"), 2: ("Patient group", "patient")}

with tab_result:
    if st.button("Run classification", type="primary"):
        try:
            with st.spinner("Comparing with the training data…"):
                scaled = scaler.transform(input_data)
                pred = int(model.predict(scaled)[0])
                proba = model.predict_proba(scaled)[0]
                classes = [int(c) for c in model.classes_]
            st.session_state["result"] = {
                "pred": pred,
                "proba": dict(zip(classes, map(float, proba))),
                "snapshot": input_data.iloc[0].to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("result", None)
            st.error(
                "The model couldn't process these values. Check that the inputs "
                f"match the features it was trained on.\n\nDetails: {exc}"
            )

    res = st.session_state.get("result")

    if not res:
        st.markdown(
            """
<div class="empty">
  <b>No result yet</b>
  Set the values in the sidebar, then press <b style="display:inline;font-size:1rem">Run classification</b>.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        # Warn if inputs changed since the last run
        if res["snapshot"] != input_data.iloc[0].to_dict():
            st.warning("Inputs changed since this result. Run the classification again to update it.")

        name, css = LABELS.get(res["pred"], (f"Class {res['pred']}", "control"))
        top = res["proba"].get(res["pred"], max(res["proba"].values()))
        st.markdown(
            f"""
<div class="result {css}">
  <div class="tag">Predicted group</div>
  <div class="big">{name}</div>
  <div class="sub">The model is {top:.0%} confident in this result.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("#### Confidence by group")
            for cls in sorted(res["proba"]):
                lbl, kind = LABELS.get(cls, (f"Class {cls}", "control"))
                p = res["proba"][cls]
                fill = "c" if kind == "control" else "p"
                st.markdown(
                    f"""
<div class="pbar">
  <div class="row"><span>{lbl}</span><span>{p:.1%}</span></div>
  <div class="track"><div class="fill {fill}" style="width:{p*100:.1f}%"></div></div>
</div>
""",
                    unsafe_allow_html=True,
                )
            if top < 0.7:
                st.info(
                    "This is a close call. KNN votes among nearby cases, so a "
                    "low-confidence result means the profile sits between the groups."
                )
        with right:
            st.markdown("#### Values used")
            snap = pd.DataFrame(
                {
                    "Biomarker": [lab for _, lab, _, _ in FEATURES],
                    "Value": [
                        fmt.format(res["snapshot"][k]) + (f" {u}" if u else "")
                        for k, _, u, fmt in FEATURES
                    ],
                }
            )
            st.dataframe(snap, hide_index=True, use_container_width=True)

        st.markdown(
            '<div class="note">This tool supports screening and learning. It is not a '
            "diagnosis. A clinician should interpret any result alongside the full "
            "clinical picture.</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# About tab
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
#### How it works
1. Your nine values are rescaled with the same scaler used during training.
2. The KNN model finds the most similar cases in the training data.
3. The share of those neighbors in each group becomes the confidence shown.

#### Groups
- **Control group** (class 1): healthy comparison profiles.
- **Patient group** (class 2): profiles from patients.

#### Tips
- HOMA is calculated for you (glucose × insulin ÷ 405). Untick the box in the sidebar to type your own.
- Use **Reset to defaults** to start over.
"""
    )
# https://glucoinsulin-predict-bbprqrsz4ytnmrrcivfuow.streamlit.app/
