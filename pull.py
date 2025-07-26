from datasets import load_dataset
import pandas as pd 
from sklearn.model_selection import train_test_split

def pull_data():
    # ========== ALL HATEXPLAIN LOGIC ==========
    # pull hatexplain
    hate_explain = load_dataset("Abhi0072/HateXplain")
    hate_explain = hate_explain['train'].to_pandas()

    
    # need to remap all the labels so that they are integers 
    label_mapping = {
    "normal": 2,
    "hatespeech": 0,
    "offensive": 1

    }

    # Apply the mapping to the label column
    hate_explain['label'] = hate_explain['label'].apply(lambda label: label_mapping.get(label))

    # ========== ALL DAVIDSON LOGIC ==========
    # pull davidson
    davidson = load_dataset("tdavidson/hate_speech_offensive")
    davidson_df = davidson['train'].to_pandas()

    # rename and drop unneeded columns
    davidson_df = davidson_df.rename(columns={'class': 'label', 'tweet': 'text'})
    davidson_df = davidson_df.drop(columns=["neither_count", "offensive_language_count", "hate_speech_count","count"])

    # ========== ALL GAB LOGIC ==========
    gab = load_dataset("juliadollis/The_Gab_Hate_Corpus_ghc_train_original")
    # convert to dataframe
    gab = gab['train'].to_pandas()

    # Filter hate speech (either attacks dignity or calls for violence)
    gab_hate = gab[(gab["hd"] == 1) | (gab["cv"] == 1)].copy()

    gab_hate['label'] = 0
    gab_hate = gab_hate.drop(columns=["hd", "cv", "vo"])


    # Filter only offensive speech (vulgar but not hateful)
    gab_offensive = gab[(gab["vo"] == 1) & (gab["hd"] == 0) & (gab["cv"] == 0)].copy()

    gab_offensive['label'] = 1
    gab_offensive = gab_offensive.drop(columns=["hd", "cv", "vo"])
    gab_offensive.head()

    gab = pd.concat([gab_hate, gab_offensive])

    # ========== ALL FRANKSHU LOGIC ===========
    frankshu = load_dataset("thefrankhsu/hate_speech_twitter")
    # convert to dataframe
    frankshu = frankshu['train'].to_pandas()

    frankshu_hate = frankshu[frankshu['label'] == 1].copy()
    frankshu_hate['label'] = 0

    frankshu_normal = frankshu[frankshu['label'] == 0].copy()
    frankshu_normal['label'] = 2

    frankshu = pd.concat([frankshu_hate, frankshu_normal])

    frankshu = frankshu.drop(columns=['categories'])
    frankshu = frankshu.rename(columns={"tweet": "text"})
    frankshu.info()

    # ========== MERGE LOGIC ==========

    # merge datasets
    dataset_df = pd.concat([hate_explain, davidson_df, gab, frankshu])

    print()
    dataset_df.info()

    return dataset_df


def create_splits(dataset_df):
    # First split: separate out test set (15%)
    train_data, test_data = train_test_split(
        dataset_df,
        test_size=0.15, 
        random_state=42,
        stratify=dataset_df['label']
    )

    # Second split: separate training data into train and validation sets (15% of original = ~17.6% of remaining data)
    train_data, val_data = train_test_split(
        train_data,
        test_size=0.176,  # This gives us approximately 15% of the original data
        random_state=42,
        stratify=train_data['label']
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
