# AI-news-misinformation-detection-and-summarizer

By : Maarjani Sanghavi 
For course ECE 592 Advanced Deep Learning

## Overview
This project implements a system to classify misinformation in news articles and generate concise summaries. The system combines a BERT-based classification model with a T5-based summarizer and includes a bias detection module to flag potentially biased statements.

---

## Code Organization

### Files and Their Descriptions
1. **`code.py`:**
   - Core implementation of the system, including:
     - Data preprocessing: Cleaning, encoding, tokenization, and integration of auxiliary features.
     - Model definition and training: Custom BERT-based classifier for misinformation detection.
     - T5 summarization and ROUGE evaluation.
     - Bias detection module.
     - End-to-end process for prediction with summary, bias detection, and classification.

2. **Dataset Files (`train.tsv`, `valid.tsv`, `test.tsv`):**
   - LIAR dataset files used for training, validation, and testing.
   - These contain labeled data for the classification task.

3. **Preprocessed Data Directory (`processed_data/`):**
   - Contains preprocessed dataset files (`train_processed.csv`, `valid_processed.csv`, `test_processed.csv`) for efficient and reproducible experimentation.

---

## Code Details and Comments

### Data Preprocessing
1. **Loading the Dataset (`load_dataset(file_path)`):**
   - Reads LIAR dataset files in TSV format and cleans data by handling missing values and converting numerical counts to proper types.

2. **Text Cleaning (`preprocess_text(text)`):**
   - Removes unnecessary elements (e.g., URLs, special characters, numbers) and ensures uniformity by converting text to lowercase.

3. **Label Encoding:**
   - Maps the dataset’s truthfulness labels to numerical categories:
     - `pants-fire: 0`, `false: 1`, `barely-true: 2`, `half-true: 3`, `mostly-true: 4`, `true: 5`.

4. **Tokenization (`tokenize_and_format(dataframe)`):**
   - Prepares text and auxiliary features for input to the BERT model.
   - Handles padding, truncation, and attention mask generation.

---

### Model and Training
1. **Custom BERT Classifier (`CustomBERTClassifier`):**
   - Enhances BERT’s embedding layer with auxiliary features (`barely_true_counts`, etc.).
   - Multi-layer architecture with dropout and ReLU activation.

2. **Training and Validation:**
   - `train_one_epoch()`: Trains the classifier for one epoch.
   - `evaluate_model(loader)`: Evaluates the model on validation or test datasets.
   - Early stopping is implemented to prevent overfitting.

---

### Summarization and Bias Detection
1. **Summarization (`generate_summary(text)`):**
   - Uses a pre-trained T5 model to generate concise summaries of input text.

2. **Bias Detection (`detect_bias(statement)`):**
   - Checks for bias using a predefined set of keywords (e.g., "always", "never", "everyone").

3. **Combined Workflow (`process_example_with_bias(statement)`):**
   - Produces a summary, detects bias, and classifies misinformation for a given statement.

---

## Running Instructions

### Prerequisites
1. Install dependencies:
   ```bash
   pip install torch transformers sklearn rouge_score matplotlib

