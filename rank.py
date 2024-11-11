import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from itertools import combinations
import numpy as np

# Load the fine-tuned model and tokenizer
model_path = "./augmented_fine_tuned_roberta_model"  # Update to your model path
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Load project ideas
data_path = "augmented_test_dataset.csv"  # Update to your CSV file containing project ideas
df = pd.read_csv(data_path)

# Define a function to get the model's score for an idea
def get_score(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        score = torch.softmax(logits, dim=1)[0][1].item()  # Get probability of 'good' class
    return score

# Calculate scores for each project idea
df['score'] = df['text'].apply(get_score)

# Rank projects based on their scores
df_sorted = df.sort_values(by='score', ascending=False).reset_index(drop=True)

# Save or display final ranking
df_sorted[['text', 'score']].to_csv("ranked_projects.csv", index=False)
print(df_sorted[['text', 'score']])