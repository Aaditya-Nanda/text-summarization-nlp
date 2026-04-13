import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add project root to path so pipeline/ and src/ are importable.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# Avoid Streamlit's file watcher scanning large dependency trees on Cloud.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st

PRIMARY_MODEL_ID = "vinnyawda/pegasus-samsum"
LIGHTWEIGHT_FALLBACK_MODEL_ID = "philschmid/bart-large-cnn-samsum"
FALLBACK_MODEL_ID = "google/pegasus-xsum"

st.set_page_config(
    page_title="Summarization CLI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================
# 🎨 CUSTOM CSS (OPENCODE STYLE)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0a0a !important;
    color: #e5e7eb !important;
}

/* Base Streamlit App Background */
.stApp {
    background-color: #0a0a0a !important;
}

/* Title */
.title {
    font-size: 2.2rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.02em;
    margin-bottom: 5px;
    color: #fff;
}

/* Subtitle */
.subtitle {
    color: #9ca3af;
    font-size: 1.1rem;
    margin-bottom: 25px;
}

/* Input box */
.stTextArea textarea {
    background-color: #111827 !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
    border: 1px solid #1f2937 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    padding: 16px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    transition: all 0.2s ease-in-out !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2), inset 0 2px 4px rgba(0,0,0,0.2) !important;
}

/* Buttons */
.stButton>button {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    border: none !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important;
    color: white !important;
}
.stButton>button[kind="primary"]:hover {
    opacity: 0.9 !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

.stButton>button[kind="secondary"] {
    background-color: #1f2937 !important;
    color: #d1d5db !important;
    border: 1px solid #374151 !important;
}
.stButton>button[kind="secondary"]:hover {
    background-color: #374151 !important;
    color: #f3f4f6 !important;
}

/* Terminal Output */
.terminal {
    background: #020617;
    border-radius: 12px;
    padding: 15px 20px;
    border: 1px solid #1f2937;
    margin-top: 20px;
    margin-bottom: 25px;
    font-family: 'JetBrains Mono', monospace;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

/* Terminal header */
.terminal-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 15px;
    padding-bottom: 12px;
    border-bottom: 1px dashed #334155;
}

.dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.red { background: #ef4444; }
.yellow { background: #f59e0b; }
.green { background: #10b981; }

.terminal-title {
    color: #64748b;
    font-size: 0.85rem;
    margin-left: auto;
}

/* Output text */
.output {
    color: #10b981;
    white-space: pre-wrap;
    line-height: 1.6;
    font-size: 0.95rem;
}

/* Error */
.error-box {
    background: #7f1d1d;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
    color: #fecaca;
    font-weight: 500;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    color: #f3f4f6 !important;
}
[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
    font-size: 0.9rem !important;
}

/* Hide Streamlit Branding elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =========================
# 🧠 MODEL PIPELINE
# =========================
@st.cache_resource(show_spinner=True)
def load_pipeline(preferred_model_id=PRIMARY_MODEL_ID):
    import os
    from pipeline.stage_05_prediction import PredictionPipeline

    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    model_candidates = [preferred_model_id]
    for model_id in [PRIMARY_MODEL_ID, LIGHTWEIGHT_FALLBACK_MODEL_ID, FALLBACK_MODEL_ID]:
        if model_id not in model_candidates:
            model_candidates.append(model_id)
    last_error = None
    for model_id in model_candidates:
        try:
            pipeline = PredictionPipeline(hub_model_id=model_id)
            return pipeline, model_id
        except Exception as exc:
            last_error = exc

    raise RuntimeError("Unable to load any summarization model.") from last_error


# =========================
# 🎨 HEADER
# =========================
st.markdown('<div class="title">⚡ Summarization CLI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Abstractive compression using NLP models</div>', unsafe_allow_html=True)

# Sample dialogue button
SAMPLE = """Hannah: Hey, are you free this weekend?
Amanda: Yes! What's up?
Hannah: A few of us are going hiking on Saturday. Want to join?
Amanda: That sounds amazing, I'd love to!
Hannah: Great, we're meeting at the trailhead at 9am.
Amanda: Perfect, I'll be there. Should I bring anything?
Hannah: Just water, snacks, and good shoes. We'll be back by noon.
Amanda: Awesome, can't wait!"""

col_sample, _ = st.columns([2, 5])
with col_sample:
    if st.button("Load Example", use_container_width=True):
        st.session_state["dialogue"] = SAMPLE

# =========================
# ✏ INPUT
# =========================
dialogue = st.text_area("",
    value=st.session_state.get("dialogue", ""),
    placeholder="> Paste your dialogue here...",
    height=220,
    label_visibility="collapsed"
)

st.caption("Press Ctrl + Enter to apply line breaks")

col1, col2, col3 = st.columns([2, 1, 4])

with col1:
    summarize = st.button("Summarize", type="primary")

with col2:
    clear = st.button("Clear")

if clear:
    st.session_state["dialogue"] = ""
    st.rerun()

# =========================
# 🚀 EXECUTION
# =========================
if summarize:
    if not dialogue.strip():
        st.markdown('<div class="error-box">⚠ Input cannot be empty. Please paste a dialogue.</div>', unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Processing generation pipeline..."):
                pipeline, loaded_model_id = load_pipeline(PRIMARY_MODEL_ID)
                summary = pipeline.predict(dialogue)
                if not summary.strip():
                    # Fallback logic
                    pipeline, loaded_model_id = load_pipeline(LIGHTWEIGHT_FALLBACK_MODEL_ID)
                    summary = pipeline.predict(dialogue)
            
            # Render Terminal Output
            st.markdown(f"""
            <div class="terminal">
                <div class="terminal-header">
                    <div class="dot red"></div>
                    <div class="dot yellow"></div>
                    <div class="dot green"></div>
                    <div class="terminal-title">Output // {loaded_model_id}</div>
                </div>
                <div class="output">
● Summarization complete

→ {summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if loaded_model_id != PRIMARY_MODEL_ID:
                st.warning(
                    f"Warning: Primary model unavailable or unreliable. Fell back to `{loaded_model_id}`.", 
                    icon="⚠️"
                )

            # Word count stats
            dialogue_words = len(dialogue.split())
            summary_words = len(summary.split())
            ratio = round(dialogue_words / max(summary_words, 1), 1)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Input Tokens", dialogue_words)
            c2.metric("Output Tokens", summary_words)
            c3.metric("Compression", f"{ratio}x")
            
        except Exception as exc:
            st.markdown(f'<div class="error-box">⚠ Execution Failed: Model loading timeout or OOM on deployment.<br>{exc}</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Powered by HuggingFace Transformers & Streamlit | Developer Toolkit V1")
