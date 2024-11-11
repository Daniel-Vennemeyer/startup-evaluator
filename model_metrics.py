import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import Dataset

# Load the fine-tuned model and tokenizer
model_path = "./augmented_fine_tuned_roberta_model"  # Path to your fine-tuned model
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Load the test dataset
test_data_path = "data/test_data/test_dataset.csv"  # Path to your test data in CSV format
df_test = pd.read_csv(test_data_path)

# Prepare the test dataset
# Create Hugging Face Dataset and tokenize
test_dataset = Dataset.from_pandas(df_test)
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Evaluate the model
def get_predictions(model, dataset):
    predictions, labels = [], []
    for batch in dataset:
        input_ids = batch['input_ids'].unsqueeze(0)  # Add batch dimension
        attention_mask = batch['attention_mask'].unsqueeze(0)
        label = batch['label'].item()

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

        predictions.append(pred)
        labels.append(label)

    return predictions, labels

# Get model predictions and true labels
predictions, labels = get_predictions(model, test_dataset)

# Calculate evaluation metrics
accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)