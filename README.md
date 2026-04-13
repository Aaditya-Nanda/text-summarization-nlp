# 📝 Dialogue Summarizer — End-to-End NLP Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dialogue-summarizer.streamlit.app)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/vinnyawda/pegasus-samsum)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)

> Fine-tuned **Google Pegasus** on the **DialogSum** dataset to produce 
> high-quality abstractive text summaries from conversations.

---

## 🚀 Live Demo

**Try it here:** [dialogue-summarizer.streamlit.app](https://dialogue-summarizer.streamlit.app)

Paste any conversation and get an abstractive summary instantly.

## 🚀 App Preview

![App Screenshot](https://github.com/Aaditya-Nanda/text-summarization-nlp/raw/main/Screenshot%202026-04-13%20220226.png)

---

## 🎯 Project Highlights

- Fine-tuned `google/pegasus-xsum` on 12,460 dialogue-summary pairs
- Achieved validation loss of **1.08** after 3 epochs
- **5.4x average compression ratio** — 65 words → 12 word summary
- Full production-grade modular Python pipeline
- Deployed on Streamlit Cloud with HuggingFace Hub integration

---

## 🏗️ Architecture
```
Raw Dialogue Text
        │
        ▼
┌─────────────────┐
│  Data Ingestion │  ← knkarthick/dialogsum from HF Hub
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Data Transformation │  ← Pegasus tokenizer, max_len=512
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │  ← Pegasus fine-tuning on Colab T4 GPU
│                 │    Adafactor + fp16 + gradient checkpointing
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Model Evaluation │  ← ROUGE-1, ROUGE-2, ROUGE-L
└────────┬─────────┘
         │
         ▼
┌────────────────────┐
│ Prediction Pipeline│  ← Inference with beam search
└────────┬───────────┘
         │
         ▼
┌──────────────┐
│ Streamlit UI │  ← Live at dialogue-summarizer.streamlit.app
└──────────────┘
```

---

## 📁 Project Structure
```
text-summarization-nlp/
├── config/
│   └── config.yaml              # All path/dataset/model configs
├── params.yaml                  # Training hyperparameters
├── src/textSummarizer/
│   ├── components/
│   │   ├── data_ingestion.py    # Download & validate dataset
│   │   ├── data_transformation.py  # Tokenization pipeline
│   │   ├── model_trainer.py     # Fine-tune Pegasus
│   │   └── model_evaluation.py  # ROUGE evaluation
│   ├── config/                  # ConfigurationManager
│   ├── entity/                  # Typed config dataclasses
│   ├── logging/                 # Centralized logger
│   └── utils/                   # Common utilities
├── pipeline/
│   ├── stage_01_data_ingestion.py
│   ├── stage_02_data_transformation.py
│   ├── stage_03_model_trainer.py
│   ├── stage_04_model_evaluation.py
│   └── stage_05_prediction.py
├── app/
│   ├── app.py                   # Streamlit web UI
│   └── requirements.txt
├── notebooks/
│   └── 01_EDA.ipynb             # Exploratory Data Analysis
├── main.py                      # Orchestrates pipeline
├── requirements.txt
└── setup.py
```

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Base Model | `google/pegasus-xsum` |
| Fine-tuned On | `knkarthick/dialogsum` |
| Training Platform | Google Colab (T4 GPU) |
| HuggingFace Hub | [vinnyawda/pegasus-samsum](https://huggingface.co/vinnyawda/pegasus-samsum) |
| Training Samples | 12,460 |
| Epochs | 3 |
| Final Val Loss | 1.0803 |
| Max Input Length | 512 tokens |
| Max Output Length | 128 tokens |

---

## ⚙️ Training Configuration

```yaml
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8    # effective batch size = 16
learning_rate: 5e-5
warmup_steps: 500
fp16: true
optimizer: Adafactor
gradient_checkpointing: true
```

---

## 🛠️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Aaditya-Nanda/text-summarization-nlp.git
cd text-summarization-nlp
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -e .
pip install -r requirements.txt
```

### 4. Run data pipeline
```bash
python main.py
```

### 5. Launch Streamlit app
```bash
streamlit run app/app.py
```

---

## 📊 EDA Insights

| Metric | Value |
|--------|-------|
| Train samples | 12,460 |
| Avg dialogue words | 131 |
| Avg summary words | 22.9 |
| Avg compression ratio | 5.8x |
| Dialogues > 512 tokens | 2% |

---

## 🔑 Key Dependencies
```
transformers    — Pegasus model + Trainer API
datasets        — DialogSum dataset loading
evaluate        — ROUGE metric computation
torch           — PyTorch backend
accelerate      — Distributed training utilities
streamlit       — Web UI
huggingface_hub — Model hosting + push to hub
sentencepiece   — Pegasus tokenizer
```

---

## 👤 Author

**Aaditya Nanda**
- LinkedIn: [linkedin.com/in/aaditya-nanda](https://www.linkedin.com/in/aaditya-nanda-1668b3257/)
- GitHub: [github.com/Aaditya-Nanda](https://github.com/Aaditya-Nanda)

---

## 📄 License

MIT License — feel free to use and build on this project.
