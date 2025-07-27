import pandas as pd
import re
import emoji
from transformers import pipeline
from tqdm import tqdm


def regex_clean(text: str) -> str:
    """Apply regex-based cleaning to a single text string."""
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'<number>', '[NUMBER]', text, flags=re.IGNORECASE)
    text = re.sub(r'<percent>', '[PERCENT]', text, flags=re.IGNORECASE)
    text = re.sub(r'<user>', '@USER', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '[TOKEN]', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def anonymize_with_ner(ner_pipeline: pipeline, text: str) -> str:
    """Anonymize PERSON and LOCATION entities in text using a NER pipeline."""
    entities = ner_pipeline(text)
    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
        if entity['entity'].startswith('B-PER') or entity['entity'].startswith('I-PER'):
            text = text[:entity['start']] + '[PERSON]' + text[entity['end']:]
        elif entity['entity'].startswith('B-LOC') or entity['entity'].startswith('I-LOC'):
            text = text[:entity['start']] + '[LOCATION]' + text[entity['end']:]
    return text


def process_batch(texts, ner_pipeline, batch_size=32):
    """Process a batch of texts with regex cleaning and NER anonymization."""
    cleaned_texts = [regex_clean(str(text)) if pd.notna(text) else "" for text in texts]
    entities_list = ner_pipeline(cleaned_texts, batch_size=batch_size)

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

    # Initialize the Hugging Face NER pipeline
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
        batch_size=32,
    )

    BATCH_SIZE = 32

    # Process and save each dataset
    for dataset, name in [(train_data, "training"), (val_data, "validation"), (test_data, "test")]:
        print(f"Processing {name} data...")
        texts = dataset['text'].tolist()
        processed_texts = []

        # Process in batches with a progress bar
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i:i + BATCH_SIZE]
            processed_batch = process_batch(batch, ner_pipeline, BATCH_SIZE)
            processed_texts.extend(processed_batch)

        # Update dataset and save
        dataset['text'] = processed_texts
        dataset.to_csv(f"data/processed_{name}_data.csv", index=False)

    print("Processing complete! Saved to processed_*.csv files")
