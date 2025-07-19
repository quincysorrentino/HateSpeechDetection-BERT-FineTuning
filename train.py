import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def prepare_data():
    print("Loading datasets...")
    train_df = pd.read_csv("data/processed_train_data.csv")
    val_df = pd.read_csv("data/processed_val_data.csv")
    test_df = pd.read_csv("data/processed_test_data.csv")

    # Rename 'class' column to 'label'
    for df in [train_df, val_df, test_df]:
        df['label'] = df['class'].astype(int)
        df.drop(columns=['class'], inplace=True)

    print("Converting to HuggingFace datasets...")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset

def tokenize_function(example):
    return tokenizer(
        example['tweet'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    train_dataset, val_dataset, test_dataset = prepare_data()

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # Remove columns not used by the model
    tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained("./hate_speech_model")
    tokenizer.save_pretrained("./hate_speech_model")
    print("Model saved to ./hate_speech_model")
