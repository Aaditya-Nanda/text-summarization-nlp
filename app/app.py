import os
import sys
import random
import base64
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st

PRIMARY_MODEL_ID = "vinnyawda/pegasus-samsum"
LIGHTWEIGHT_FALLBACK_MODEL_ID = "philschmid/bart-large-cnn-samsum"
FALLBACK_MODEL_ID = "google/pegasus-xsum"

st.set_page_config(
    page_title="Dialogue Summarizer",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
# Quotes
# ------------------------------------------------------------------ #
QUOTES = [
    "Focus is the new superpower.",
    "Clarity beats chaos.",
    "Build in silence, ship in clarity.",
    "Compress the noise. Amplify the signal.",
    "Less words, more meaning.",
    "Brevity is the soul of understanding.",
    "Distill conversations into insight.",
    "Every great summary starts with listening.",
    "Turn dialogue into direction.",
    "The best ideas fit in one sentence.",
    "Think deep, summarize sharp.",
    "Intelligence is compression.",
]

if "quote" not in st.session_state:
    st.session_state.quote = random.choice(QUOTES)

# ------------------------------------------------------------------ #
# Background image → base64
# ------------------------------------------------------------------ #
BG_PATH = APP_DIR / "hero_bg.png"

def get_bg_base64():
    if BG_PATH.exists():
        return base64.b64encode(BG_PATH.read_bytes()).decode()
    return None

bg_b64 = get_bg_base64()

# ------------------------------------------------------------------ #
# CSS
# ------------------------------------------------------------------ #
bg_css = ""
if bg_b64:
    bg_css = f"""
    .stApp {{
        background: url("data:image/png;base64,{bg_b64}") no-repeat center center fixed;
        background-size: cover;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: linear-gradient(
            180deg,
            rgba(5, 5, 15, 0.75) 0%,
            rgba(5, 5, 15, 0.88) 50%,
            rgba(5, 5, 15, 0.95) 100%
        );
        z-index: 0;
        pointer-events: none;
    }}
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    color: #e4e4e7 !important;
}}

{bg_css}

header, footer, #MainMenu {{ visibility: hidden !important; }}

.block-container {{
    padding-top: 8vh !important;
    max-width: 860px !important;
    position: relative;
    z-index: 1;
}}

/* ---- Fade-in animation ---- */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(25px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in {{ animation: fadeUp 0.8s ease-out forwards; }}
.fade-in-delayed {{ animation: fadeUp 0.8s ease-out 0.3s forwards; opacity: 0; }}
.fade-in-delayed-2 {{ animation: fadeUp 0.8s ease-out 0.55s forwards; opacity: 0; }}

/* ---- Hero heading ---- */
.hero-quote {{
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    line-height: 1.15;
    text-align: center;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #f0f0f0 0%, #a5b4fc 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.hero-sub {{
    text-align: center;
    font-size: 1.05rem;
    color: #a1a1aa;
    font-weight: 400;
    margin-bottom: 45px;
}}

/* ---- Glassmorphism Input ---- */
.stTextArea textarea {{
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    color: #e4e4e7 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 18px 20px !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.25) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}}
.stTextArea textarea::placeholder {{
    color: #52525b !important;
}}
.stTextArea textarea:focus {{
    border-color: rgba(129,140,248,0.5) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.12), 0 4px 30px rgba(0,0,0,0.25) !important;
}}
.stTextArea label {{ display: none !important; }}
[data-testid="stTextArea"] {{ margin-bottom: 0 !important; }}

/* ---- Buttons ---- */
.stButton > button {{
    border-radius: 40px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 6px 28px rgba(99,102,241,0.55) !important;
    transform: translateY(-1px);
}}
.stButton > button[kind="secondary"] {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #a1a1aa !important;
}}
.stButton > button[kind="secondary"]:hover {{
    background: rgba(255,255,255,0.08) !important;
    color: #f4f4f5 !important;
}}

/* ---- Result panel ---- */
@keyframes resultFade {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.result-panel {{
    animation: resultFade 0.5s ease-out forwards;
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 28px 30px;
    margin-top: 30px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}}
.result-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #818cf8;
    margin-bottom: 12px;
    font-weight: 600;
}}
.result-text {{
    font-size: 1.05rem;
    line-height: 1.7;
    color: #e4e4e7;
    font-weight: 400;
}}
.result-model {{
    margin-top: 16px;
    font-size: 0.75rem;
    color: #52525b;
}}

/* ---- Metrics ---- */
[data-testid="stMetricValue"] {{
    font-size: 1.5rem !important;
    color: #f4f4f5 !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricLabel"] {{
    color: #71717a !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}

/* ---- Fallback banner ---- */
.fallback-banner {{
    background: rgba(234,179,8,0.08);
    border: 1px solid rgba(234,179,8,0.2);
    border-radius: 10px;
    padding: 10px 16px;
    margin-top: 14px;
    font-size: 0.82rem;
    color: #fbbf24;
}}

/* ---- Footer ---- */
.app-footer {{
    text-align: center;
    color: #3f3f46;
    font-size: 0.75rem;
    margin-top: 60px;
    padding-bottom: 30px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Model pipeline (unchanged logic)
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner=False)
def load_pipeline(preferred_model_id=PRIMARY_MODEL_ID):
    import os as _os
    from pipeline.stage_05_prediction import PredictionPipeline

    token = _os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    model_candidates = [preferred_model_id]
    for mid in [PRIMARY_MODEL_ID, LIGHTWEIGHT_FALLBACK_MODEL_ID, FALLBACK_MODEL_ID]:
        if mid not in model_candidates:
            model_candidates.append(mid)

    last_error = None
    for mid in model_candidates:
        try:
            pipe = PredictionPipeline(hub_model_id=mid)
            return pipe, mid
        except Exception as exc:
            last_error = exc

    raise RuntimeError("Unable to load any summarization model.") from last_error

# ------------------------------------------------------------------ #
# Hero
# ------------------------------------------------------------------ #
st.markdown(
    f'<div class="fade-in"><div class="hero-quote">{st.session_state.quote}</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="fade-in-delayed"><div class="hero-sub">'
    "Paste any conversation and get an abstractive summary — powered by fine-tuned Pegasus."
    "</div></div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Sample data
# ------------------------------------------------------------------ #
SAMPLE = """Hannah: Hey, are you free this weekend?
Amanda: Yes! What's up?
Hannah: A few of us are going hiking on Saturday. Want to join?
Amanda: That sounds amazing, I'd love to!
Hannah: Great, we're meeting at the trailhead at 9am.
Amanda: Perfect, I'll be there. Should I bring anything?
Hannah: Just water, snacks, and good shoes. We'll be back by noon.
Amanda: Awesome, can't wait!"""

# ------------------------------------------------------------------ #
# Input section
# ------------------------------------------------------------------ #
st.markdown('<div class="fade-in-delayed-2">', unsafe_allow_html=True)

col_ex, _ = st.columns([2, 5])
with col_ex:
    if st.button("Load example", use_container_width=True):
        st.session_state["dialogue"] = SAMPLE

dialogue = st.text_area(
    label="Input",
    value=st.session_state.get("dialogue", ""),
    height=200,
    placeholder="Paste your dialogue here...",
    label_visibility="collapsed",
)

col_btn1, col_btn2, _ = st.columns([2, 1.2, 4])
with col_btn1:
    summarize = st.button("✦  Summarize", type="primary")
with col_btn2:
    clear = st.button("Clear")

st.markdown("</div>", unsafe_allow_html=True)

if clear:
    st.session_state["dialogue"] = ""
    st.session_state.pop("summary_result", None)
    st.session_state.pop("summary_model", None)
    st.rerun()

# ------------------------------------------------------------------ #
# Execution
# ------------------------------------------------------------------ #
if summarize:
    if not dialogue.strip():
        st.warning("Please paste a dialogue first.")
    else:
        try:
            with st.spinner("Generating summary…"):
                pipeline, loaded_model_id = load_pipeline(PRIMARY_MODEL_ID)
                summary = pipeline.predict(dialogue)
                if not summary.strip():
                    pipeline, loaded_model_id = load_pipeline(LIGHTWEIGHT_FALLBACK_MODEL_ID)
                    summary = pipeline.predict(dialogue)
            st.session_state["summary_result"] = summary
            st.session_state["summary_model"] = loaded_model_id
        except Exception as exc:
            st.error(f"Model could not be loaded: {exc}")

# ------------------------------------------------------------------ #
# Output
# ------------------------------------------------------------------ #
if "summary_result" in st.session_state and st.session_state["summary_result"]:
    summary = st.session_state["summary_result"]
    model_used = st.session_state.get("summary_model", "unknown")

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-label">Generated Summary</div>
            <div class="result-text">{summary}</div>
            <div class="result-model">Model: {model_used}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_used != PRIMARY_MODEL_ID:
        st.markdown(
            f'<div class="fallback-banner">⚠ Primary model was unavailable — '
            f'used <b>{model_used}</b> as fallback.</div>',
            unsafe_allow_html=True,
        )

    # Stats
    dialogue_words = len(dialogue.split())
    summary_words = len(summary.split())
    ratio = round(dialogue_words / max(summary_words, 1), 1)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Input words", dialogue_words)
    c2.metric("Output words", summary_words)
    c3.metric("Compression", f"{ratio}×")

# ------------------------------------------------------------------ #
# Footer
# ------------------------------------------------------------------ #
st.markdown(
    '<div class="app-footer">'
    "Built with Hugging Face Transformers & Streamlit ·  Aaditya Nanda"
    "</div>",
    unsafe_allow_html=True,
)
