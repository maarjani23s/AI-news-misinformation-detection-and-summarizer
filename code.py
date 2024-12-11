from google.colab import drive
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, AutoModel, AutoTokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import re
import string
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

# Step 1: Define Dataset Path
# Define the folder containing the LIAR dataset files
data_directory = './dataset_liar'

# Step 2: Load LIAR Dataset
data_columns = [
    "id", "label", "statement", "subjects", "speaker",
    "speaker_job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

# Load dataset and process numeric fields
def load_dataset(file_path):
    dataframe = pd.read_csv(file_path, sep='\t', header=None, names=data_columns)
    dataframe = dataframe.dropna(subset=["label", "statement"])  # Drop rows with missing labels/statements
    count_fields = ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]
    for column in count_fields:
        dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce').fillna(0)  # Convert counts to numeric
    return dataframe

# Load training, validation, and test datasets
train_data = load_dataset(os.path.join(data_directory, 'train.tsv'))
val_data = load_dataset(os.path.join(data_directory, 'valid.tsv'))
test_data = load_dataset(os.path.join(data_directory, 'test.tsv'))

# Step 3: Text Cleaning
# Preprocessing to remove noise and normalize text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Apply preprocessing to all datasets
for dataset in [train_data, val_data, test_data]:
    dataset['statement'] = dataset['statement'].apply(preprocess_text)

# Step 4: Encode Labels
# Map text labels to numeric classes
label_to_class = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

# Apply label encoding to all datasets
for dataset in [train_data, val_data, test_data]:
    dataset['label'] = dataset['label'].map(label_to_class)

# Step 5: Save Preprocessed Data
# Save processed data for future use
preprocessed_dir = os.path.join(data_directory, 'processed_data')
os.makedirs(preprocessed_dir, exist_ok=True)
train_data.to_csv(os.path.join(preprocessed_dir, 'train_processed.csv'), index=False)
val_data.to_csv(os.path.join(preprocessed_dir, 'valid_processed.csv'), index=False)
test_data.to_csv(os.path.join(preprocessed_dir, 'test_processed.csv'), index=False)

# Step 6: Tokenization
# Tokenize statements using BertTokenizer
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
max_length = 128  # Maximum sequence length for BERT

def tokenize_and_format(dataframe, tokenizer, max_len=128):
    tokenized_output = tokenizer(
        dataframe['statement'].tolist(),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = torch.tensor(dataframe['label'].values)
    count_fields = ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]
    additional_data = torch.tensor(
        dataframe[count_fields].values, dtype=torch.float
    )
    return {
        "input_ids": tokenized_output["input_ids"],
        "attention_mask": tokenized_output["attention_mask"],
        "labels": labels,
        "additional_features": additional_data
    }

# Tokenize training, validation, and test datasets
train_tokens = tokenize_and_format(train_data, bert_tokenizer)
val_tokens = tokenize_and_format(val_data, bert_tokenizer)
test_tokens = tokenize_and_format(test_data, bert_tokenizer)

# Step 7: DataLoader Setup
batch_size = 16  # Batch size for training and evaluation

# Define DataLoader for train, validation, and test sets
train_loader = DataLoader(
    TensorDataset(
        train_tokens["input_ids"],
        train_tokens["attention_mask"],
        train_tokens["labels"],
        train_tokens["additional_features"]
    ),
    sampler=RandomSampler(train_tokens["input_ids"]), batch_size=batch_size
)

val_loader = DataLoader(
    TensorDataset(
        val_tokens["input_ids"],
        val_tokens["attention_mask"],
        val_tokens["labels"],
        val_tokens["additional_features"]
    ),
    sampler=SequentialSampler(val_tokens["input_ids"]), batch_size=batch_size
)

test_loader = DataLoader(
    TensorDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"],
        test_tokens["labels"],
        test_tokens["additional_features"]
    ),
    sampler=SequentialSampler(test_tokens["input_ids"]), batch_size=batch_size
)

# Step 8: Define BERT Model with Additional Features
class CustomBERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(CustomBERTClassifier, self).__init__()
        self.bert = bert_model  # BERT base model
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(768 + 5, 768)  # Incorporate additional features
        self.dense2 = nn.Linear(768, 512)
        self.dense3 = nn.Linear(512, len(label_to_class))  # Final layer for classification
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, additional_features):
        _, cls_hidden_state = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        combined_input = torch.cat((cls_hidden_state, additional_features), dim=1)
        output = self.dropout(combined_input)
        output = self.activation(self.dense1(output))
        output = self.dropout(output)
        output = self.activation(self.dense2(output))
        output = self.dropout(output)
        output = self.dense3(output)
        return self.softmax(output)

# Load Pre-trained BERT Model
bert_model = AutoModel.from_pretrained('bert-base-uncased')
classifier = CustomBERTClassifier(bert_model).to(device)

# Define loss function with class weights
train_labels_array = train_tokens["labels"].numpy()
computed_weights = compute_class_weight('balanced', classes=np.unique(train_labels_array), y=train_labels_array)
class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float).to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer and Scheduler
optimizer = AdamW(classifier.parameters(), lr=2e-5, weight_decay=0.01)
training_epochs = 10
training_steps = training_epochs * len(train_loader)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=50, num_training_steps=training_steps)

# Train and Evaluate the Model
for epoch in range(training_epochs):
    print(f"Epoch {epoch + 1}/{training_epochs}")
    epoch_train_loss = train_one_epoch()
    validation_loss, validation_accuracy, validation_report = evaluate_model(val_loader)
    print(f"Training Loss: {epoch_train_loss:.3f}, Validation Loss: {validation_loss:.3f}, Validation Accuracy: {validation_accuracy:.3f}")
    print("Validation Classification Report:\n", validation_report)

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        torch.save(classifier.state_dict(), 'best_classifier_model.pt')
        stopping_counter = 0
    else:
        stopping_counter += 1

    if stopping_counter >= max_patience:
        print("Early stopping activated. No further improvements.")
        break

# Test the Model
test_trained_model()

# Summarization and Bias Detection
def process_example(statement, reference_summary=None):
    # Generate summary
    summary = generate_summary(statement)
    # Detect bias
    bias_detected = detect_bias(summary)
    # Classify misinformation
    predictions = make_prediction([summary])
    # Calculate ROUGE scores if a reference summary is provided
    rouge_scores = None
    if reference_summary:
        rouge_scores = evaluate_rouge([reference_summary], [summary])
    return summary, bias_detected, predictions[0], rouge_scores


example_statement = "Vaccines are harmful and always cause severe side effects, according to recent reports."
reference_summary = "Vaccines are safe and effective."
summary, bias, prediction, rouge_scores = process_example(example_statement, reference_summary)

# Print Results
print(f"Original Statement: {example_statement}")
print(f"Generated Summary: {summary}")
print(f"Contains Bias: {bias}")
print(f"Misinformation Prediction: {prediction}")
if rouge_scores:
    print(f"ROUGE Scores: {rouge_scores}")
