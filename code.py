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

# Step 1: Mount Google Drive
#drive.mount('/content/drive')

# Step 2: Define Dataset Path
data_directory = './dataset_liar'

# Step 3: Load LIAR Dataset
data_columns = [
    "id", "label", "statement", "subjects", "speaker",
    "speaker_job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

def load_dataset(file_path):
    dataframe = pd.read_csv(file_path, sep='\t', header=None, names=data_columns)
    dataframe = dataframe.dropna(subset=["label", "statement"])
    count_fields = ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]
    for column in count_fields:
        dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce').fillna(0)
    return dataframe

train_data = load_dataset(os.path.join(data_directory, 'train.tsv'))
val_data = load_dataset(os.path.join(data_directory, 'valid.tsv'))
test_data = load_dataset(os.path.join(data_directory, 'test.tsv'))

# Step 4: Text Cleaning
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for dataset in [train_data, val_data, test_data]:
    dataset['statement'] = dataset['statement'].apply(preprocess_text)

# Step 5: Encode Labels
label_to_class = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}
for dataset in [train_data, val_data, test_data]:
    dataset['label'] = dataset['label'].map(label_to_class)

# Step 6: Save Preprocessed Data
preprocessed_dir = os.path.join(data_directory, 'processed_data')
os.makedirs(preprocessed_dir, exist_ok=True)
train_data.to_csv(os.path.join(preprocessed_dir, 'train_processed.csv'), index=False)
val_data.to_csv(os.path.join(preprocessed_dir, 'valid_processed.csv'), index=False)
test_data.to_csv(os.path.join(preprocessed_dir, 'test_processed.csv'), index=False)

# Step 7: Tokenization
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
max_length = 128

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

train_tokens = tokenize_and_format(train_data, bert_tokenizer)
val_tokens = tokenize_and_format(val_data, bert_tokenizer)
test_tokens = tokenize_and_format(test_data, bert_tokenizer)

# Step 8: DataLoader Setup
batch_size = 16

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

# Define BERT Model with Additional Features
class CustomBERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(CustomBERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(768 + 5, 768)
        self.dense2 = nn.Linear(768, 512)
        self.dense3 = nn.Linear(512, len(label_to_class))
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

# Load Pre-trained BERT
bert_model = AutoModel.from_pretrained('bert-base-uncased')
classifier = CustomBERTClassifier(bert_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = classifier.to(device)

train_labels_array = train_tokens["labels"].numpy()
computed_weights = compute_class_weight('balanced', classes=np.unique(train_labels_array), y=train_labels_array)
class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float).to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = AdamW(classifier.parameters(), lr=2e-5, weight_decay=0.01)
training_epochs = 10
training_steps = training_epochs * len(train_loader)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=50, num_training_steps=training_steps)

def train_one_epoch():
    classifier.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels, extra_features = [item.to(device) for item in batch]
        classifier.zero_grad()
        predictions = classifier(input_ids, attention_mask, extra_features)
        loss = loss_function(predictions, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
    return total_loss / len(train_loader)

def evaluate_model(loader):
    classifier.eval()
    eval_loss = 0
    predictions_list = []
    ground_truth_list = []
    for batch in loader:
        input_ids, attention_mask, labels, extra_features = [item.to(device) for item in batch]
        with torch.no_grad():
            predictions = classifier(input_ids, attention_mask, extra_features)
            loss = loss_function(predictions, labels)
            eval_loss += loss.item()
            predictions_list.append(predictions.detach().cpu().numpy())
            ground_truth_list.append(labels.cpu().numpy())
    predictions_array = np.concatenate(predictions_list, axis=0)
    ground_truth_array = np.concatenate(ground_truth_list, axis=0)
    predicted_classes = np.argmax(predictions_array, axis=1)
    return eval_loss / len(loader), accuracy_score(ground_truth_array, predicted_classes), classification_report(ground_truth_array, predicted_classes, target_names=list(label_to_class.keys()))

best_val_loss = float('inf')
stopping_counter = 0
max_patience = 3

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

classifier.load_state_dict(torch.load('best_classifier_model.pt'))

def test_trained_model():
    test_loss, test_accuracy, test_report = evaluate_model(test_loader)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")
    print("Test Classification Report:\n", test_report)

test_trained_model()

# Add T5 Summarizer
summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-small")
summarizer_tokenizer = AutoTokenizer.from_pretrained("t5-small")

def generate_summary(text):
    inputs = summarizer_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Prediction Function
def make_prediction(sentences):
    classifier.eval()
    tokenized = bert_tokenizer(
        sentences,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    additional_features = torch.zeros((len(sentences), 5)).to(device)  # Placeholder for additional features

    with torch.no_grad():
        logits = classifier(input_ids, attention_mask, additional_features)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions

# Add ROUGE Score Evaluation
def evaluate_rouge(reference_texts, generated_summaries):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, gen in zip(reference_texts, generated_summaries):
        score = scorer.score(ref, gen)
        scores["rouge1"].append(score["rouge1"].fmeasure)
        scores["rouge2"].append(score["rouge2"].fmeasure)
        scores["rougeL"].append(score["rougeL"].fmeasure)
    avg_scores = {key: np.mean(val) for key, val in scores.items()}
    return avg_scores

def detect_bias(statement):
    bias_keywords = ["always", "never", "everyone", "no one"]
    statement_lower = statement.lower()
    for keyword in bias_keywords:
        if keyword in statement_lower:
            return True
    return False

# Updated process_example to include Bias Detection
def process_example_with_bias(statement):
    summary = generate_summary(statement)
    bias_detected = detect_bias(summary)
    predictions = make_prediction([summary])
    return summary, bias_detected, predictions[0]

# Example Usage with Bias Detection
example_statement = "Vaccines are harmful and always cause severe side effects, according to recent reports."
summary, bias, prediction = process_example_with_bias(example_statement)
print(f"Original Statement: {example_statement}")
print(f"Generated Summary: {summary}")
print(f"Contains Bias: {bias}")
print(f"Prediction: {prediction}")
