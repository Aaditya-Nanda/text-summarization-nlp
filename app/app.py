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

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Dialogue Summarizer | OpenCode",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------ #
# Custom CSS
# ------------------------------------------------------------------ #
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0a0a0a !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    
    /* Typography */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        color: #ffffff !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.2rem !important;
        padding-bottom: 0 !important;
    }
    h3 {
        font-weight: 500 !important;
        color: #a1a1aa !important;
        font-size: 1.1rem !important;
        margin-top: 0 !important;
    }
    
    /* Text Area Command Bar Styling */
    .stTextArea textarea {
        background-color: #121214 !important;
        border: 1px solid #27272a !important;
        border-radius: 12px !important;
        color: #e4e4e7 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.95rem !important;
        padding: 1.2rem !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2), inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9 !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #18181b !important;
        color: #a1a1aa !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #27272a !important;
        color: #f4f4f5 !important;
    }

    /* Terminal Output Panel */
    .terminal-output {
        background-color: #0f0f11;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        position: relative;
    }
    .terminal-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid #27272a;
        padding-bottom: 0.8rem;
    }
    .terminal-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .terminal-title {
        color: #71717a;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
        margin-left: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .terminal-content {
        color: #10b981; /* Emerald green for output */
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        color: #f4f4f5 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a1a1aa !important;
    }
    
    /* Utility */
    .helper-text {
        font-size: 0.8rem;
        color: #52525b;
        margin-top: -12px;
        margin-bottom: 16px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Hide Streamlit Branding elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Load model once and cache it
# ------------------------------------------------------------------ #
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

# ------------------------------------------------------------------ #
# UI Layout
# ------------------------------------------------------------------ #

# Header
st.markdown("<h1>⚡ Summarization CLI</h1>", unsafe_allow_html=True)
st.markdown("### Abstractive compression using fine-tuned NLP models", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

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

dialogue = st.text_area(
    label="Input Dialog",
    label_visibility="collapsed",
    value=st.session_state.get("dialogue", ""),
    height=220,
    placeholder="> Paste your dialogue here...",
)

st.markdown("<div class='helper-text'>Press Ctrl+Enter to apply line breaks. Click Summarize to execute.</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 4])
with col1:
    summarize = st.button("Summarize", type="primary")
with col2:
    clear = st.button("Clear")

if clear:
    st.session_state["dialogue"] = ""
    st.rerun()

if summarize:
    if not dialogue.strip():
        st.error("Input cannot be empty. Please paste a dialogue.")
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
                <div class="terminal-output">
                    <div class="terminal-header">
                        <span class="terminal-dot" style="background-color: #ef4444;"></span>
                        <span class="terminal-dot" style="background-color: #eab308;"></span>
                        <span class="terminal-dot" style="background-color: #22c55e;"></span>
                        <span class="terminal-title">Output // {loaded_model_id}</span>
                    </div>
                    <div class="terminal-content">{summary}</div>
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
            st.error("Execution Failed: Model loading timeout or OOM on deployment.")
            st.code(str(exc), language="text")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption(
    "Powered by HuggingFace Transformers & Streamlit | Developer Toolkit V1"
)
