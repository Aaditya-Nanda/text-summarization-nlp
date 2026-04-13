from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "vinnyawda/pegasus-samsum"
MODEL_DIR = Path("artifacts/model_trainer/pegasus-samsum-model")


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(MODEL_DIR),
        commit_message="Upload fine-tuned Pegasus dialogue summarization model",
        ignore_patterns=[
            "checkpoint-*",
            "*.pt",
        ],
    )
    print(f"Upload complete: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
