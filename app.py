# app.py
import os
import base64
import numpy as np
import streamlit as st
import joblib

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Smart Irrigation System", page_icon="🌱", layout="wide")

# Change this to your real repo URL
GITHUB_URL = "https://github.com/Jayita2004/Smart_Irrigation"

# Your local background image path
LOCAL_BG_PATH = r"C:\Users\USER\OneDrive\Desktop\green-abstract-blur-background_629712-568.avif"

# ----------------------------
# HELPERS
# ----------------------------
def file_to_base64(path: str) -> str | None:
    """Return base64 string for a local file, or None if not available."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

def inject_css(bg_base64: str | None):
    """Inject CSS for background & styling (no black bars)."""
    if bg_base64:
        bg_css = f"""
        .stApp {{
            background: url("data:image/jpeg;base64,{bg_base64}") no-repeat center center fixed;
            background-size: cover;
            color: #fff;
        }}
        """
    else:
        bg_css = """
        .stApp {
            background: radial-gradient(1200px 600px at 20% 10%, #486a58 0%, #2e4438 40%, #1f2e27 100%);
            color: #fff;
        }
        """

    st.markdown(
        f"""
        <style>
        {bg_css}

        /* Remove black glass overlay */
        .glass {{
            background: transparent;
            border: none;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
            border-radius: 0;
            padding: 0;
        }}

        /* Section titles */
        .section-title {{
            font-weight: 800;
            letter-spacing: 0.2px;
        }}

        /* Prediction chips */
        .chip {{
            display: inline-block;
            padding: 10px 14px;
            border-radius: 999px;
            font-weight: 700;
            margin: 6px 8px 0 0;
            border: 1px solid rgba(255,255,255,0.12);
        }}
        .chip-on  {{ background: rgba(16,185,129,0.18); }}
        .chip-off {{ background: rgba(239,68,68,0.18);  }}

        /* Nice full-width button */
        .stButton > button {{
            width: 100%;
            font-weight: 700;
            border-radius: 12px;
            padding: 10px 14px;
        }}

        /* Slider labels visible on dark bg */
        .stSlider label {{
            color: #fff !important;
            font-weight: 600;
        }}

        /* Footer link */
        .footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 0.9rem;
        }}
        .footer a {{
            color: #90ee90;
            text-decoration: none;
            font-weight: bold;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# BACKGROUND
bg_b64 = file_to_base64(LOCAL_BG_PATH)
inject_css(bg_b64)
# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.write("Use sliders on the main page to set **scaled** sensor values (0 → 1).")
    st.divider()
    st.markdown("### ℹ️ About")
    st.write(
        "This app predicts which **sprinklers (parcels)** should turn **ON/OFF** "
        "using a trained **Random Forest (MultiOutputClassifier)**."
    )
    st.caption("Model file: `Farm_Irrigation_System.pkl`")

# LOAD MODEL
@st.cache_resource
def load_model():
    try:
        return joblib.load("Farm_Irrigation_System.pkl")
    except Exception as e:
        st.error(
            "❌ Could not load `Farm_Irrigation_System.pkl`. "
            "Make sure it's in the same folder as `app.py`."
        )
        st.exception(e)
        return None

model = load_model()
# HEADER
st.title("🌱 Smart Sprinkler System")
st.write(
    "Welcome to the **Smart Irrigation Predictor**! Adjust the sliders below to set "
    "**scaled sensor values** (0 → 1) and check which sprinklers should be **ON** or **OFF**."
)

st.write("")  # spacing
# SENSOR INPUTS
st.markdown("<h3 class='section-title'>🔧 Sensor Inputs</h3>", unsafe_allow_html=True)

sensor_values = []
cols = st.columns(4)
for i in range(20):
    with cols[i % 4]:
        val = st.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"s{i}")
        sensor_values.append(val)

st.write("")  # spacing
# PREDICTION
if st.button("🔍 Predict Sprinklers"):
    if model is None:
        st.stop()

    x = np.array(sensor_values, dtype=float).reshape(1, -1)
    y_pred = model.predict(x)[0]

    st.markdown("<h3 class='section-title'>💡 Prediction Results</h3>", unsafe_allow_html=True)

    # Summary bar
    on_count = int(np.sum(y_pred == 1))
    off_count = len(y_pred) - on_count
    st.progress(on_count / len(y_pred))
    st.caption(f"✅ ON: {on_count} / {len(y_pred)} | ❌ OFF: {off_count} / {len(y_pred)}")

    # Chips grid
    c1, c2, c3, c4 = st.columns(4)
    cols_pred = [c1, c2, c3, c4]
    for i, status in enumerate(y_pred):
        html = (
            f"<span class='chip {'chip-on' if status == 1 else 'chip-off'}'>"
            f"{'✅' if status == 1 else '❌'} Sprinkler {i} — <b>{'ON' if status == 1 else 'OFF'}</b>"
            f"</span>"
        )
        with cols_pred[i % 4]:
            st.markdown(html, unsafe_allow_html=True)

    # (Optional) raw array
    with st.expander("See raw prediction array"):
        st.write(y_pred.tolist())
else:
    st.info("Set sensor values and click **Predict Sprinklers** to see results.")

# FOOTER
st.markdown("""
    <div class='footer'>
        Created by <a href="https://github.com/Jayita2004/Smart_Irrigation" target="_blank">Jayita</a>
    </div>
""", unsafe_allow_html=True)
