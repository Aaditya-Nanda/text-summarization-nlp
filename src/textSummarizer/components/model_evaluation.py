# src/textSummarizer/components/model_evaluation.py

import torch
import pandas as pd
from tqdm import tqdm
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from pathlib import Path
from textSummarizer.entity.config_entity import ModelEvaluationConfig
from textSummarizer.logging.logger import logger


class ModelEvaluation:
    """
    Loads the fine-tuned Pegasus model and evaluates it on the
    test split using ROUGE-1, ROUGE-2, and ROUGE-L metrics.
    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Evaluation device: {self.device}")

    def generate_batch_sized_chunks(self, list_of_elements: list, batch_size: int):
        """Splits a list into batches for memory-efficient inference."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size: int = 8,
        column_text: str = "dialogue",
        column_summary: str = "summary",
    ):
        """
        Runs batched inference on the dataset and computes ROUGE scores.
        """
        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches),
            desc="Evaluating batches",
        ):
            inputs = tokenizer(
                article_batch,
                max_length=self.config.max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            summaries = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                length_penalty=0.8,
                num_beams=8,
                max_length=self.config.max_target_length,
            )

            decoded_summaries = tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            metric.add_batch(
                predictions=decoded_summaries,
                references=target_batch,
            )

        score = metric.compute()
        return score

    def evaluate(self):
        """
        Loads model, tokenizer, and test dataset.
        Computes and saves ROUGE metrics to CSV.
        """
        # Load tokenizer and model
        logger.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))

        logger.info(f"Loading model from: {self.config.model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.config.model_path),
            torch_dtype=torch.float16,
        ).to(self.device)
        model.eval()
        logger.info("Model loaded in eval mode.")

        # Load test dataset (raw — not tokenized)
        logger.info(f"Loading test dataset from: {self.config.data_path}")

        # Load raw dataset for evaluation (needs original text columns)
        from datasets import load_from_disk
        dataset = load_from_disk(
            str(Path("artifacts") / "data_ingestion" / "samsum_dataset")
        )
        test_dataset = dataset["test"]
        logger.info(f"Test samples: {len(test_dataset):,}")

        # Load ROUGE metric
        rouge_metric = load("rouge")

        # Run evaluation on a subset for speed
        eval_subset = test_dataset.select(range(min(100, len(test_dataset))))
        logger.info(f"Evaluating on {len(eval_subset)} test samples...")

        score = self.calculate_metric_on_test_ds(
            dataset=eval_subset,
            metric=rouge_metric,
            model=model,
            tokenizer=tokenizer,
            batch_size=4,
            column_text=self.config.source_column,
            column_summary=self.config.target_column,
        )

        # Format and save results
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = {rn: round(score[rn] * 100, 2) for rn in rouge_names}

        logger.info("=" * 45)
        logger.info("         ROUGE EVALUATION RESULTS")
        logger.info("=" * 45)
        for k, v in rouge_dict.items():
            logger.info(f"  {k:<12}: {v}")
        logger.info("=" * 45)

        # Save to CSV
        df = pd.DataFrame([rouge_dict])
        df.to_csv(str(self.config.metric_file_name), index=False)
        logger.info(f"Metrics saved to: {self.config.metric_file_name}")

        return rouge_dict

    def run(self):
        """Full evaluation pipeline."""
        results = self.evaluate()
        logger.info("Model evaluation complete.")
        return results