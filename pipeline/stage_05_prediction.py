# pipeline/stage_05_prediction.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textSummarizer.logging.logger import logger


class PredictionPipeline:
    """
    Loads the fine-tuned Pegasus model and generates
    abstractive summaries for raw dialogue input.
    Can load from local path or HuggingFace Hub.
    """

    def __init__(self, model_path: str = None, hub_model_id: str = None):
        """
        Args:
            model_path  : Local path to saved model directory.
            hub_model_id: HuggingFace Hub model ID (fallback if local not found).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Prediction device: {self.device}")

        # Resolve model source
        if model_path:
            source = model_path
        elif hub_model_id:
            source = hub_model_id
        else:
            # NOTE: The local model was trained on only 100 samples (smoke test)
            # and produces empty outputs. Using base google/pegasus-xsum instead.
            # Switch to local_path after a full training run (2000+ samples).
            source = "google/pegasus-xsum"

        logger.info(f"Loading tokenizer from: {source}")
        self.tokenizer = AutoTokenizer.from_pretrained(source)

        logger.info(f"Loading model from: {source}")
        model_kwargs = {
            "dtype": torch.float16 if self.device == "cuda" else "auto",
        }
        if self.device == "cpu":
            # Reduce peak RAM usage on Streamlit Community Cloud.
            model_kwargs["low_cpu_mem_usage"] = True

        self.model = AutoModelForSeq2SeqLM.from_pretrained(source, **model_kwargs)
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Prediction pipeline ready.")

    def predict(self, dialogue: str) -> str:
        """
        Generates an abstractive summary for the given dialogue.

        Args:
            dialogue: Raw conversation text.

        Returns:
            Generated summary string.
        """
        if not dialogue or not dialogue.strip():
            return "Please provide a valid dialogue."

        inputs = self.tokenizer(
            dialogue,
            max_length=384,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summaries = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                length_penalty=1.0,
                num_beams=2,
                max_new_tokens=64,
                min_new_tokens=8,
                no_repeat_ngram_size=2,
                use_cache=False if self.device == "cpu" else True,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(
            summaries[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        logger.info(f"Summary generated: {summary[:80]}...")
        return summary
