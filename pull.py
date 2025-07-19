from datasets import load_dataset
import pandas as pd 
from sklearn.model_selection import train_test_split

def pull_data():
    dataset = load_dataset("tdavidson/hate_speech_offensive")

    # convert to dataframe
    dataset_df = dataset['train'].to_pandas()

    # prettyprint all df info
    print(
        """
        count: (Integer) number of users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable,
        hate_speech_annotation: (Integer) number of users who judged the tweet to be hate speech,
        offensive_language_annotation: (Integer) number of users who judged the tweet to be offensive,
        neither_annotation: (Integer) number of users who judged the tweet to be neither offensive nor non-offensive,
        label: (Class Label) class label for majority of CF users (0: 'hate-speech', 1: 'offensive-language' or 2: 'neither'),
        tweet: (string)
        """
    )

    dataset_df.info()
    dataset_df.head()

    return dataset_df


def create_splits(dataset_df):
    # First split: separate out test set (15%)
    train_data, test_data = train_test_split(
        dataset_df,
        test_size=0.15, 
        random_state=42,
        stratify=dataset_df['class']
    )

    # Second split: separate training data into train and validation sets (15% of original = ~17.6% of remaining data)
    train_data, val_data = train_test_split(
        train_data,
        test_size=0.176,  # This gives us approximately 15% of the original data
        random_state=42,
        stratify=train_data['class']
    )

    return train_data, test_data, val_data


if __name__ == "__main__":
    dataset_df = pull_data()
    train_data, test_data, val_data = create_splits(dataset_df)

    # Calculate total size
    total = len(train_data) + len(test_data) + len(val_data)

    print()
    print(f"Training set size: {len(train_data)} ({len(train_data)/total:.2%})")
    print(f"Test set size: {len(test_data)} ({len(test_data)/total:.2%})")
    print(f"Validation set size: {len(val_data)} ({len(val_data)/total:.2%})")

    train_data.to_csv("data/train_data.csv", index=False)
    test_data.to_csv("data/test_data.csv", index=False)
    val_data.to_csv("data/val_data.csv", index=False)
