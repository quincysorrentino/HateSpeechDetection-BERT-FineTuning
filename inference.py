import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import emoji
import re

# === Optional text cleaning ===
def clean_text(text):
    """Clean text by converting emojis to text and normalizing whitespace"""
    # Convert emojis to text descriptions
    text = emoji.demojize(text)
    # Remove special characters but keep emoji descriptions
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === Inference function ===
def predict_text_with_threshold(text, model, tokenizer, thresholds, fallback="argmax"):
    if not isinstance(text, str):
        text = str(text)
    text = clean_text(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    confident = [(cls, p) for cls, p in enumerate(probs) if p >= thresholds[cls]]
    if confident:
        return label_map[max(confident, key=lambda x: x[1])[0]]
    else:
        return label_map[int(probs.argmax())] if fallback == "argmax" else fallback

# === CLI Application ===
def run_cli():
    print("\n=== Hate Speech Detection CLI ===")
    print("Type your text to analyze")
    print("Type '/exit' to quit")
    print("-" * 30)

    while True:
        # Get input with proper prompt
        text = input("\nEnter text: ").strip()
        
        # Check for exit command
        if text.lower() == '/exit':
            print("\nGoodbye!")
            break
        
        # Skip empty inputs
        if not text:
            print("Please enter some text to analyze.")
            continue
            
        try:
            # Run inference
            result = predict_text_with_threshold(text, model, tokenizer, thresholds)
            print(f"\nPrediction: {result}")
            
            # Add confidence scores
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                print("\nConfidence scores:")
                for cls_id, prob in enumerate(probs):
                    print(f"{label_map[cls_id]}: {prob:.2%}")
            
        except Exception as e:
            print(f"Error processing text: {str(e)}")


if __name__ == "__main__":
        
    # === Load model and tokenizer ===
    model_path = "hate_speech_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()

    # === Label map ===
    label_map = {0: "hatespeech", 1: "offensive", 2: "normal"}

    # === Hardcoded best thresholds ===
    thresholds = {
        0: 0.560,   # hatespeech
        1: 0.420,  # offensive
        2: 0.320   # normal
    }
    run_cli()
