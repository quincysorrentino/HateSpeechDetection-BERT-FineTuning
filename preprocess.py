import pandas as pd
import re
import emoji
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm

def regex_clean(text):
    """Clean social media text while preserving important context"""
    
    # Handle @mentions - replace with generic token
    text = re.sub(r'@\w+', '@USER', text)
    
    # Handle special angle bracket patterns
    text = re.sub(r'<number>', '[NUMBER]', text, flags=re.IGNORECASE)
    text = re.sub(r'<percent>', '[PERCENT]', text, flags=re.IGNORECASE)
    text = re.sub(r'<user>', '@USER', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '[TOKEN]', text)
    
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
    example['text'] = regex_clean(example['text'])
    example['text'] = anonymize_with_ner(example['text'])
    return example

def process_batch(texts, ner_pipeline, batch_size=32):
    """Process a batch of texts with regex cleaning and NER anonymization"""
    # Clean texts and handle NaN values
    cleaned_texts = [regex_clean(str(text)) if pd.notna(text) else "" for text in texts]
    
    # Process all texts in one batch through NER
    entities_list = ner_pipeline(cleaned_texts, batch_size=batch_size)
    
    # Anonymize all texts
    result = []
    for text, entities in zip(cleaned_texts, entities_list):
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            if entity['entity'].startswith('B-PER') or entity['entity'].startswith('I-PER'):
                text = text[:entity['start']] + '[PERSON]' + text[entity['end']:]
            elif entity['entity'].startswith('B-LOC') or entity['entity'].startswith('I-LOC'):
                text = text[:entity['start']] + '[LOCATION]' + text[entity['end']:]
        result.append(text)
    
    return result

if __name__ == "__main__":
    # Load datasets
    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")
    val_data = pd.read_csv("data/val_data.csv")

    # Load NER model with batch processing enabled
    ner_pipeline = pipeline("ner", 
                          model="dbmdz/bert-large-cased-finetuned-conll03-english",
                          tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                          batch_size=32)

    BATCH_SIZE = 32

    # Process each dataset with batching
    for dataset, name in [(train_data, "training"), (val_data, "validation"), (test_data, "test")]:
        print(f"Processing {name} data...")
        texts = dataset['text'].tolist()
        processed_texts = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i:i + BATCH_SIZE]
            processed_batch = process_batch(batch, ner_pipeline, BATCH_SIZE)
            processed_texts.extend(processed_batch)
        
        dataset['text'] = processed_texts
        dataset.to_csv(f"data/processed_{name}_data.csv", index=False)

    print("Processing complete! Saved to processed_*.csv files")
