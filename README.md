# AI-news-misinformation-detection-and-summarizer

By : Maarjani Sanghavi 
For course ECE 592 Advanced Deep Learning

## Overview
This project implements a system to classify misinformation in news articles and generate concise summaries. The system combines a BERT-based classification model with a T5-based summarizer and includes a bias detection module to flag potentially biased statements.

---

### Code Organization
1. **`code.py`:**
   - Core implementation of the pipeline, including:
     - Data preprocessing:
       - Functions like `load_dataset`, `preprocess_text`, and `tokenize_and_format` prepare the dataset for model training.
     - Custom BERT Classifier:
       - Implements `CustomBERTClassifier` to integrate BERT embeddings with auxiliary features for improved classification.
     - Training Functions:
       - Functions such as `train_one_epoch` and `evaluate_model` handle model training and validation.
     - Summarization:
       - `generate_summary` generates concise summaries using a T5-based model.
     - Bias Detection:
       - `detect_bias` flags potentially biased summaries based on keywords.
     - End-to-End Prediction:
       - `process_example` combines summarization, bias detection, and classification.

2. **Dataset Files:**
   - LIAR Dataset files (`train.tsv`, `valid.tsv`, `test.tsv`) for training, validation, and testing.
   - Saved preprocessed files in `processed_data/` can be reused in future runs.

3. **Outputs:**
   - Trained model checkpoints (`best_classifier_model.pt`).
   - ROUGE score evaluations for summaries.
   - Classification reports for validation and testing datasets.


---

### Key Classes and Functions (Developed by me)
1. **`Data preprocessing functions`:**
   - A comprehensive pipeline that cleans text, encodes labels, and integrates auxiliary features (e.g., barely_true_counts) with tokenized inputs for model training.
     
2.  **`CustomBERTClassifier`:**
   - A multi-layer BERT-based classifier that incorporates auxiliary features (e.g., truthfulness counts) with BERT embeddings for better prediction accuracy.

3. **`train_one_epoch`:**
   - Handles a single epoch of model training, including gradient updates and loss computation.

4. **`evaluate_model`:**
   - Evaluates the model on a validation or test dataset and provides metrics like accuracy and a classification report.

5. **`generate_summary`:**
   - Uses a pre-trained T5 model to generate concise summaries from input text.

6. **`process_example`:**
   - Combines summarization, bias detection, and misinformation classification for a single input statement.

---

### Developed Components
The following components were developed as part of this project:
- **Data Preprocessing Pipeline:**
  - Added auxiliary feature integration (`barely_true_counts`, etc.) into tokenization.
  - Improved text cleaning logic to handle noisy inputs.

- **Custom BERT Classifier:**
  - Enhanced BERT embeddings with auxiliary features.
  - Added dropout layers and ReLU activations to improve model generalization.

- **Bias Detection Module:**
  - Designed a keyword-based approach to flag potentially biased summaries.

- **Integration of Summarization Module:**
  - Integrated a T5 model for summarization and evaluated its quality using ROUGE scores.

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

3. **Combined Workflow (`process_example(statement, reference_summary= None)`):**
   - Produces a summary, detects bias, and classifies misinformation for a given statement.

---

## Running Instructions

### Prerequisites
1. Install dependencies:
   ```bash
   pip install torch transformers sklearn rouge_score matplotlib
2. Running the code for training:
   Ensure the dataset files (train.tsv, valid.tsv, test.tsv) are in the dataset_liar directory.
   ```bash
   python code.py
4. For seperate testing input:
   ```bash
   Modify the `example_statement` variable in the script to test a new input. Also add a reference summary if you want to calculate the rogue score.
5. Reuse preprocessed data: Preprocessed files are saved in the processed_data/ directory after your initial run. Adjust the script to load these files to skip preprocessing during future runs.

