# Hate Speech Detection Using Bert (ToXicBERT)

## Model Description
ToxiBERT [LINK TO HUGGINFACEDATASET] is a fine-tuned RoBERTa fine tuned model specifically designed to detect and classify hate speech, offensive language, and normal content on the X (formerly Twitter) platform.

## Key Technical Achievements
- Collecting data from various sources and compiling a custom dataset for training 
- Fine tuning existing model on cloud computing resources 
- Evaluating model efficacy 

## Core Technologies


## Model Architecture
- Base Model: BERTweet (RoBERTa) [https://github.com/VinAIResearch/BERTweet]
- Fine-tuning: Multi-class classification
- Input: Self compiled dataset from X
- Output: Three-class classification
  - Hate Speech
  - Offensive Content
  - Normal Content

## Key Features 
- Data collection pipeline pulling various datasets from Huggingface

- Data preprocessing workflow
  - Anonymization using NER 
  - Cleaning pretokenization from compiled datasets
  - URL handling 
  - Emoji handling

- Model training (Google Colab)
  - Hyperparameter tuning 
- Model Evaluation
  - Threshold analysis and optimization 

- Interactive CLI inference tool

## Model Performance
Optimal class-specific thresholds:
  hatespeech: 0.560
  offensive: 0.420
  normal: 0.320

Results with class-specific thresholds:
              precision    recall  f1-score   support

  hatespeech      0.624     0.818     0.708      1625
   offensive      0.892     0.759     0.820      3868
      normal      0.803     0.827     0.815      2474

    accuracy                          0.792      7967
   macro avg      0.773     0.801     0.781      7967
weighted avg      0.810     0.792     0.796      7967

## 

