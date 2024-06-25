import pandas as pd
import torch
from transformers import pipeline

def classify_comment(comment):
    result = classifier(comment, labels)
    return result['labels'][0]

input_file = 'comments.csv'
comments_df = pd.read_csv(input_file)

print(comments_df.head())

classifier = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")
labels = ["informative", "frustrated", "afraid", "skeptical", "critical", "supportive", "concerned", "unconcerned", "optimistic"]

comments_df['Type'] = comments_df['Comment'].apply(classify_comment)

output_file = 'csv_files/classified_data.csv'
comments_df.to_csv(output_file, index=False)