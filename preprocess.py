import pandas as pd
import re
import emoji
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def regex_clean(text):
    """Clean social media text while preserving important context"""
    
    # Handle @mentions - replace with generic token
    text = re.sub(r'@\w+', '@USER', text)
    
    # Handle hashtags - keep the text, remove #
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle URLs
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    
    # Handle repeated characters (sooooo -> so)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Handle emojis - convert to text description
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def anonymize_with_ner(ner_pipeline, text):
    """Use NER to identify and anonymize personal information"""
    
    # Get entities
    entities = ner_pipeline(text)
    
    # Replace person names, locations, organizations
    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
        if entity['entity'].startswith('B-PER') or entity['entity'].startswith('I-PER'):
            text = text[:entity['start']] + '[PERSON]' + text[entity['end']:]
        elif entity['entity'].startswith('B-LOC') or entity['entity'].startswith('I-LOC'):
            text = text[:entity['start']] + '[LOCATION]' + text[entity['end']:]
    
    return text

def preprocess_dataset(example):
    example['tweet'] = regex_clean(example['tweet'])
    example['tweet'] = anonymize_with_ner(example['tweet'])
    return example


if __name__ == "__main__":
    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")
    val_data = pd.read_csv("data/val_data.csv")

    # Load NER model
    ner_pipeline = pipeline("ner", 
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

    # Process each dataset
    print("Processing training data...")
    train_data['tweet'] = train_data['tweet'].progress_apply(lambda x: regex_clean(x))
    train_data['tweet'] = train_data['tweet'].progress_apply(lambda x: anonymize_with_ner(ner_pipeline, x))

    print("Processing validation data...")
    val_data['tweet'] = val_data['tweet'].progress_apply(lambda x: regex_clean(x))
    val_data['tweet'] = val_data['tweet'].progress_apply(lambda x: anonymize_with_ner(ner_pipeline, x))

    print("Processing test data...")
    test_data['tweet'] = test_data['tweet'].progress_apply(lambda x: regex_clean(x))
    test_data['tweet'] = test_data['tweet'].progress_apply(lambda x: anonymize_with_ner(ner_pipeline, x))

    # Save processed datasets
    train_data.to_csv("data/processed_train_data.csv", index=False)
    val_data.to_csv("data/processed_val_data.csv", index=False)
    test_data.to_csv("data/processed_test_data.csv", index=False)

    print("Processing complete! Saved to processed_*.csv files")