import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import emoji
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# === Text Cleaning Function ===
def clean_text(text):
    """Clean text by converting emojis to text and normalizing whitespace."""
    # Convert emojis to text descriptions (e.g., 😊 -> :smiling_face_with_smiling_eyes:)
    text = emoji.demojize(text)
    # Remove special characters but keep the text of emoji descriptions
    text = re.sub(r'[^\w\s_:]+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === Label Normalization Function ===
def normalize_label(label, label_map):
    """
    Normalize labels to ensure consistency.
    Converts numeric labels to string labels and handles variations.
    """
    if pd.isna(label):
        return None
    
    # Convert to string and strip whitespace
    label_str = str(label).strip().lower()
    
    # Handle numeric labels (0, 1, 2)
    if label_str in ['0', '0.0']:
        return label_map[0]
    elif label_str in ['1', '1.0']:
        return label_map[1]  
    elif label_str in ['2', '2.0']:
        return label_map[2]
    
    # Handle string labels with variations
    label_variations = {
        'hatespeech': label_map[0],
        'hate_speech': label_map[0],
        'hate speech': label_map[0],
        'offensive': label_map[1],
        'normal': label_map[2],
        'neither': label_map[2],
        'not offensive': label_map[2]
    }
    
    return label_variations.get(label_str, label_str)

# === Inference Function ===
def predict_text_with_threshold(text, model, tokenizer, thresholds, label_map, fallback="argmax"):
    """
    Predicts the class for a given text using confidence thresholds.

    Args:
        text (str): The input text to classify.
        model: The loaded transformer model.
        tokenizer: The loaded tokenizer.
        thresholds (dict): A dictionary mapping class IDs to their confidence thresholds.
        label_map (dict): A dictionary mapping class IDs to their string labels.
        fallback (str): The strategy if no threshold is met. 
                        "argmax" for the highest probability, or a default label.

    Returns:
        str: The predicted label.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Handle empty text after cleaning
    if not cleaned_text:
        return fallback if fallback != "argmax" else label_map[2]  # default to 'normal'

    try:
        # Tokenize and prepare input for the model
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Perform inference
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Ensure probs has the expected number of classes
        expected_classes = len(label_map)
        if len(probs) != expected_classes:
            print(f"Warning: Model output has {len(probs)} classes, expected {expected_classes}")
            # Pad with zeros if too few classes, or truncate if too many
            if len(probs) < expected_classes:
                padded_probs = [0.0] * expected_classes
                for i in range(len(probs)):
                    padded_probs[i] = probs[i]
                probs = padded_probs
            else:
                probs = probs[:expected_classes]

        # Apply confidence thresholds
        confident_predictions = []
        for cls_id, probability in enumerate(probs):
            if cls_id in thresholds and probability >= thresholds[cls_id]:
                confident_predictions.append((cls_id, probability))
        
        # If there are predictions that meet the threshold, choose the one with the highest probability
        if confident_predictions:
            # Get the class ID from the tuple (cls_id, probability) with the max probability
            best_class_id = max(confident_predictions, key=lambda item: item[1])[0]
            return label_map[best_class_id]
        else:
            # If no prediction meets its threshold, use the fallback strategy
            if fallback == "argmax":
                best_class_id = int(probs.index(max(probs))) if isinstance(probs, list) else int(probs.argmax())
                return label_map.get(best_class_id, label_map[2])  # fallback to 'normal' if key doesn't exist
            else:
                return fallback
                
    except Exception as e:
        print(f"Error during prediction: {e}")
        return fallback if fallback != "argmax" else label_map[2]

# === Evaluation Utilities ===
def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using Seaborn's heatmap.
    
    Args:
        cm (numpy.ndarray): The confusion matrix.
        class_names (list): A list of class names for labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, tokenizer, thresholds, label_map, eval_filepath):
    """
    Evaluates the model on a given dataset and prints performance metrics.
    Skips rows that cause errors during prediction.

    Args:
        model: The loaded transformer model.
        tokenizer: The loaded tokenizer.
        thresholds (dict): Confidence thresholds for each class.
        label_map (dict): A dictionary mapping class IDs to their string labels.
        eval_filepath (str): The path to the evaluation CSV file.
    """
    print("\n" + "="*30)
    print("      STARTING MODEL EVALUATION")
    print("="*30 + "\n")

    # Load evaluation data
    print(f"Loading evaluation data from: {eval_filepath}")
    df_eval = pd.read_csv(eval_filepath)

    # Ensure required columns exist
    if 'text' not in df_eval.columns or 'label' not in df_eval.columns:
        print(f"Error: Evaluation CSV '{eval_filepath}' must contain 'text' and 'label' columns.")
        return

    print(f"Loaded {len(df_eval)} rows from evaluation dataset")
    print(f"Columns found: {list(df_eval.columns)}")
    
    # Check label distribution in the dataset
    print(f"\nOriginal label distribution:")
    print(df_eval['label'].value_counts())

    # Prepare lists for true and predicted labels, skipping bad data
    y_true_clean = []
    y_pred_clean = []
    skipped_rows = 0
    
    print("\nRunning predictions on the evaluation set...")
    for index, row in df_eval.iterrows():
        text = row['text']
        true_label = row['label']
        
        try:
            # Explicitly skip rows with empty or non-string text
            if pd.isna(text) or not str(text).strip():
                print(f"Warning: Skipping row {index} due to empty or invalid text.")
                skipped_rows += 1
                continue

            # Normalize the true label
            normalized_true_label = normalize_label(true_label, label_map)
            if normalized_true_label is None:
                print(f"Warning: Skipping row {index} due to invalid label: {true_label}")
                skipped_rows += 1
                continue

            # Get the prediction
            pred_label = predict_text_with_threshold(text, model, tokenizer, thresholds, label_map)
            
            # Append valid results to our lists
            y_pred_clean.append(pred_label)
            y_true_clean.append(normalized_true_label)

        except Exception as e:
            # Catch any error during prediction for a single row
            print(f"Warning: Skipping row {index} due to an error: {e}")
            skipped_rows += 1

        # Progress indicator
        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1}/{len(df_eval)} rows...")

    print(f"\nPredictions complete. Processed {len(y_true_clean)} rows, skipped {skipped_rows} rows.\n")
    
    if not y_true_clean:
        print("Evaluation could not be completed. No valid rows were processed.")
        return

    # Check final label distribution
    print(f"Final true label distribution:")
    true_label_counts = pd.Series(y_true_clean).value_counts()
    print(true_label_counts)
    
    print(f"\nFinal predicted label distribution:")
    pred_label_counts = pd.Series(y_pred_clean).value_counts()
    print(pred_label_counts)

    # Generate all possible class names from both true and predicted labels
    all_labels = sorted(list(set(y_true_clean + y_pred_clean)))
    
    # Generate and print classification report
    print("\n--- Classification Report ---")
    try:
        report = classification_report(y_true_clean, y_pred_clean, labels=all_labels, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")
        print("This might be due to label inconsistencies. Printing basic accuracy...")
        correct = sum(1 for true, pred in zip(y_true_clean, y_pred_clean) if true == pred)
        accuracy = correct / len(y_true_clean)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{len(y_true_clean)})")

    # Generate and plot confusion matrix
    print("\n--- Confusion Matrix ---")
    try:
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=all_labels)
        plot_confusion_matrix(cm, all_labels)
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    print(f"\nEvaluation completed on {len(y_true_clean)} out of {len(df_eval)} total rows.")
    print("\n" + "="*30)
    print("      EVALUATION COMPLETE")
    print("="*30 + "\n")


if __name__ == "__main__":
    try:
        # === Load Model and Tokenizer ===
        MODEL_PATH = "hate_speech_model"
        
        if not os.path.isdir(MODEL_PATH):
            print(f"Error: Model directory not found at '{MODEL_PATH}'")
            print("Please make sure your fine-tuned model is saved in that directory or update the MODEL_PATH variable.")
            sys.exit(1)

        print(f"Loading model from: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()

        # Check model configuration
        print(f"Model config - num_labels: {model.config.num_labels}")

        # === Configuration ===
        label_map = {0: "hatespeech", 1: "offensive", 2: "normal"}
        thresholds = {
            0: 0.560,  # hatespeech
            1: 0.420,  # offensive
            2: 0.320   # normal
        }

        # Verify model has expected number of classes
        if model.config.num_labels != len(label_map):
            print(f"Warning: Model was trained on {model.config.num_labels} classes, but label_map has {len(label_map)} classes")
            print("This might cause issues. Consider updating the label_map or retraining the model.")

        # === Find Evaluation Data File ===
        EVAL_CSV_PATH = None
        possible_paths = [
            'data/test_data.csv',
            'test_data.csv', 
            'data/val_data.csv',
            'val_data.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                EVAL_CSV_PATH = path
                break
        
        if EVAL_CSV_PATH is None:
            print("\nError: Evaluation data not found.")
            print("Please ensure one of these files exists:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)

        # === Run Evaluation ===
        evaluate_model(model, tokenizer, thresholds, label_map, EVAL_CSV_PATH)

    except Exception as e:
        print(f"\nA critical error occurred during setup: {e}")
        import traceback
        traceback.print_exc()