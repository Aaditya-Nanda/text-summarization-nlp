import os
import sys
from pathlib import Path
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

PRIMARY_MODEL_ID = "vinnyawda/pegasus-samsum"
LIGHTWEIGHT_FALLBACK_MODEL_ID = "philschmid/bart-large-cnn-samsum"
FALLBACK_MODEL_ID = "google/pegasus-xsum"

st.set_page_config(page_title="opencode", layout="centered", initial_sidebar_state="collapsed")

# =========================
# 🎨 CUSTOM CSS TO MATCH IMAGE EXACTLY
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    background-color: #050505 !important;
    color: #e5e7eb !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background-color: #050505 !important;
}

/* Hide header/footer/Streamlit menu */
header, footer, #MainMenu {
    visibility: hidden !important;
}

/* Container spacing */
.block-container {
    padding-top: 15vh !important;
    max-width: 800px !important;
}

/* Logo */
.logo {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -3px;
    margin-bottom: 50px;
}
.logo span:first-child { color: #52525B; }
.logo span:last-child { color: #E4E4E7; }

/* Remove bottom margin from text area */
[data-testid="stTextArea"] {
    margin-bottom: 0 !important;
}

/* Text Area */
.stTextArea textarea {
    background-color: #111111 !important;
    color: #a1a1aa !important;
    border: none !important;
    border-left: 3px solid #3b82f6 !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    padding: 20px !important;
    padding-bottom: 50px !important; /* Make room for the floating text */
    box-shadow: none !important;
    height: 120px !important;
    resize: none !important;
    line-height: 1.5 !important;
}
.stTextArea textarea:focus {
    color: #e5e7eb !important;
    border-left: 3px solid #60a5fa !important;
    outline: none !important;
    box-shadow: none !important;
}
.stTextArea label {
    display: none !important;
}

/* The Build / Model text floated over the text area */
.bottom-bar {
    position: relative;
    top: -40px;
    left: 20px;
    font-size: 13px;
    display: flex;
    gap: 15px;
    pointer-events: none; /* Let clicks pass through to text area */
}
.bottom-bar .build {
    color: #3b82f6;
}
.bottom-bar .model {
    color: #52525b;
}

/* Helper text */
.helper {
    text-align: right;
    color: #52525b;
    font-size: 11.5px;
    margin-top: -10px;
    margin-bottom: 40px;
}

/* The invisible submit button */
.stButton > button {
    display: none !important; /* Entirely rely on Ctrl+Enter */
}

/* Output styling */
.output-box {
    margin-top: 20px;
    color: #f4f4f5;
    font-size: 14px;
    line-height: 1.6;
    border-left: 3px solid #3b82f6;
    padding-left: 20px;
    background: transparent;
}
.error-box {
    margin-top: 20px;
    color: #ef4444;
    font-size: 14px;
    border-left: 3px solid #ef4444;
    padding-left: 20px;
}

/* Make spinner transparent */
.stSpinner > div > div {
    border-color: #3b82f6 transparent transparent transparent !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_pipeline(preferred_model_id=PRIMARY_MODEL_ID):
    import os
    from pipeline.stage_05_prediction import PredictionPipeline
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
    try:
        pipeline = PredictionPipeline(hub_model_id=preferred_model_id)
        return pipeline, preferred_model_id
    except Exception as exc:
        raise RuntimeError("Unable to load summarization model.") from exc

st.markdown('<div class="logo"><span>open</span><span>code</span></div>', unsafe_allow_html=True)

# Input area
# Streamlit text_area updates session_state and triggers a rerun on Ctrl+Enter
dialogue = st.text_area(
    label="Input", 
    value=st.session_state.get("dialogue", ""),
    placeholder="add dialogue to compress...", 
    height=150
)

# Float the "Build..." text over the text area
st.markdown("""
<div class="bottom-bar">
    <span class="build">Build</span>
    <span class="model">Pegasus Samsum Zen</span>
</div>
""", unsafe_allow_html=True)

# Helper commands
st.markdown('<div class="helper">ctrl+enter to run</div>', unsafe_allow_html=True)

# The logic kicks in if dialogue is not empty (e.g. after Ctrl+Enter is hit)
# To avoid running repeatedly, we track previous input.
if "prev_dialogue" not in st.session_state:
    st.session_state.prev_dialogue = ""

if dialogue and dialogue.strip() and dialogue != st.session_state.prev_dialogue:
    with st.spinner(""):
        try:
            pipeline, m_id = load_pipeline(PRIMARY_MODEL_ID)
            summary = pipeline.predict(dialogue)
            if not summary.strip():
                pipeline, m_id = load_pipeline(LIGHTWEIGHT_FALLBACK_MODEL_ID)
                summary = pipeline.predict(dialogue)
            
            st.markdown(f'<div class="output-box">{summary}</div>', unsafe_allow_html=True)
            st.session_state.prev_dialogue = dialogue
        except Exception as exc:
            st.markdown(f'<div class="error-box">Build Failed.<br>{exc}</div>', unsafe_allow_html=True)
elif dialogue == st.session_state.prev_dialogue and dialogue.strip():
    # Keep output visible if it was already generated
    # (Since we didn't store the summary itself, we'd need to re-generate it to display,
    # or better, store it in session state).
    pass

# We should ideally store the summary in session_state to persist output.
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "summary_error" not in st.session_state:
    st.session_state.summary_error = ""

if dialogue and dialogue.strip() and dialogue != st.session_state.prev_dialogue:
    with st.spinner(""):
        try:
            pipeline, m_id = load_pipeline(PRIMARY_MODEL_ID)
            summary = pipeline.predict(dialogue)
            if not summary.strip():
                pipeline, m_id = load_pipeline(LIGHTWEIGHT_FALLBACK_MODEL_ID)
                summary = pipeline.predict(dialogue)
            
            st.session_state.summary_output = summary
            st.session_state.summary_error = ""
            st.session_state.prev_dialogue = dialogue
        except Exception as exc:
            st.session_state.summary_error = str(exc)
            st.session_state.summary_output = ""
            st.session_state.prev_dialogue = dialogue

if st.session_state.summary_output and dialogue == st.session_state.prev_dialogue:
    st.markdown(f'<div class="output-box">{st.session_state.summary_output}</div>', unsafe_allow_html=True)
elif st.session_state.summary_error and dialogue == st.session_state.prev_dialogue:
    st.markdown(f'<div class="error-box">Build Failed.<br>{st.session_state.summary_error}</div>', unsafe_allow_html=True)

# Update prev_dialogue if user clears input manually
if not dialogue.strip():
    st.session_state.prev_dialogue = ""
    st.session_state.summary_output = ""
    st.session_state.summary_error = ""
