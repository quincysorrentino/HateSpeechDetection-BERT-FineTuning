#!/usr/bin/env python3
"""
Script to upload ToxiBERT model to Hugging Face Hub.
Ensure required packages are installed:
    pip install huggingface_hub transformers torch
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuration
MODEL_NAME = "toXibert"
HF_USERNAME = "QuincySorrentino"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
LOCAL_MODEL_PATH = "hate_speech_model"
HF_TOKEN = os.getenv("HF_TOKEN")

def ensure_token():
    global HF_TOKEN
    if not HF_TOKEN:
        HF_TOKEN = input("🔐 Enter your Hugging Face token: ").strip()
    if not HF_TOKEN:
        print("❌ Hugging Face token is required.")
        exit(1)

def setup_model_directory():
    """Check if model directory exists"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ Model directory '{LOCAL_MODEL_PATH}' not found.")
        exit(1)
    print(f"✅ Found model directory: {LOCAL_MODEL_PATH}")

def load_model_and_tokenizer():
    """Load trained model and tokenizer from local directory"""
    print("📦 Loading model and tokenizer from local path...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        print("✅ Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed to load model or tokenizer: {e}")
        exit(1)

def maybe_update_config():
    """Update config.json with metadata if not already present"""
    config_path = os.path.join(LOCAL_MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        print("❌ config.json not found. Ensure model was saved correctly.")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    updated = False
    if "label2id" not in config:
        config["label2id"] = {
            "hate_speech": 0,
            "offensive": 1,
            "normal": 2
        }
        updated = True

    if "id2label" not in config:
        config["id2label"] = {str(v): k for k, v in config["label2id"].items()}
        updated = True

    if "num_labels" not in config:
        config["num_labels"] = 3
        updated = True

    if updated:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("🔧 config.json updated with missing metadata.")
    else:
        print("✅ config.json already contains required metadata.")

def upload_to_huggingface():
    """Upload the model to Hugging Face Hub"""
    print("🚀 Uploading to Hugging Face Hub...")
    api = HfApi(token=HF_TOKEN)

    try:
        create_repo(repo_id=REPO_ID, token=HF_TOKEN, private=False, exist_ok=True)
        print(f"✅ Repository ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        exit(1)

    try:
        upload_folder(
            folder_path=LOCAL_MODEL_PATH,
            repo_id=REPO_ID,
            token=HF_TOKEN,
            commit_message="Initial model upload"
        )
        print(f"✅ Upload complete: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Failed to upload model: {e}")
        exit(1)

def main():
    print("📤 Starting ToxiBERT upload process...")
    ensure_token()
    setup_model_directory()

    model, tokenizer = load_model_and_tokenizer()
    maybe_update_config()

    # Skipping README.md creation if already exists
    readme_path = os.path.join(LOCAL_MODEL_PATH, "README.md")
    if not os.path.exists(readme_path):
        print("⚠️ README.md not found. You can add a model card manually.")
    else:
        print("✅ Found existing README.md (model card)")

    upload_to_huggingface()

    print("\n🎉 Done!")
    print(f"💻 Load your model using:\nfrom transformers import AutoModelForSequenceClassification\nmodel = AutoModelForSequenceClassification.from_pretrained('{REPO_ID}')")

if __name__ == "__main__":
    main()
