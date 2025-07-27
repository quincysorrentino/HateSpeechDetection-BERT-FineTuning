# Hate Speech Detection Using Bert (ToXiBERT)

## Model Description
[ToXiBERT](https://huggingface.co/QuincySorrentino/toXibert/blob/main/README.md) is a fine-tuned RoBERTa fine tuned model specifically designed to detect and classify hate speech, offensive language, and normal content on the X (formerly Twitter) platform.

## Key Technical Achievements
- Advanced Transfer Learning: Fine-tuned [BERTweet](https://huggingface.co/docs/transformers/en/model_doc/bertweet) (RoBERTa) architecture for domain-specific hate speech detection
- Comprehensive Data Pipeline: Multi-source data collection and compilation
- Privacy-Preserving Preprocessing: NER-based anonymization and advanced text normalization workflows
- Threshold Optimization Framework: Class-specific threshold tuning achieving 79.2% accuracy with balanced precision-recall


## Core Technologies
- Deep Learning Framework: PyTorch with CUDA acceleration on Google Colab
- NLP Library: Hugging Face Transformers ecosystem with custom tokenization
- Base Model: BERTweet (RoBERTa-base optimized for Twitter text)
- Data Processing: Pandas, NumPy, scikit-learn with custom preprocessing pipelines

## Technical Implementations
- Multi-Source Data Collection Pipeline
  - Automated dataset aggregation from Hugging Face Hub
  - Custom data validation and quality assurance workflows
  - Stratified sampling for balanced class representation
  - Data versioning and reproducible dataset compilation

- Advanced anonymization and normalization pipeline
  - Named Entity Recognition (NER) for automatic anonymization
  - Twitter-specific preprocessing (mentions, hashtags, URLs)
  - Emoji standardization and handling
  - Custom tokenization for social media text

- Advanced Transfer Learning Strategy
  - Domain adaptation from general RoBERTa to Twitter-specific BERTweet
  - Layer-wise learning rate scheduling for optimal fine-tuning
  - Gradient accumulation for effective large batch training
  - Early stopping with validation monitoring

- Hyperparameter Optimization Framework
  - Automated hyperparameter search with Weights & Biases (wandb) experiment tracking
  - Multi-objective optimization targeting macro F1-score for balanced class performance
  - Dynamic parameter space exploration including learning rates, batch sizes, epochs, weight decay
  - Best model selection with automatic checkpoint saving and restoration

- Threshold Optimization
  - Class-specific threshold optimization using F1-score maximization
  - Statistical significance testing for threshold selection
  - Cross-validation framework for robust threshold estimation
  -Uncertainty quantification for low-confidence predictions

## Model Performance
Overall Accuracy: 79.2% with balanced class performance

Weighted F1-Score: 79.6% accounting for class imbalance

Hate Speech Detection: 70.8% F1 with 82% recall (prioritizing detection)
Offensive Content: 82.0% F1 with high precision-recall balance
Normal Content: 81.5% F1 with robust classification accuracy

Thresholding Analysis

```
### Optimal Class-Specific Thresholds

| Class      | Threshold |
|------------|-----------|
| hatespeech | 0.560     |
| offensive  | 0.420     |
| normal     | 0.320     |

### Results with Class-Specific Thresholds

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| hatespeech  | 0.624     | 0.818  | 0.708    | 1625    |
| offensive   | 0.892     | 0.759  | 0.820    | 3868    |
| normal      | 0.803     | 0.827  | 0.815    | 2474    |
| **Accuracy**     |           |        | **0.792**    | **7967**   |
| **Macro Avg**    | 0.773     | 0.801  | 0.781    | 7967    |
| **Weighted Avg** | 0.810     | 0.792  | 0.796    | 7967    |
```

## Future Enhancements
- Multi-language support with cross-lingual transfer learning