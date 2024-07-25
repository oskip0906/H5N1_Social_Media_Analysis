import pandas as pd
from transformers import pipeline
import os

# Load the zero-shot classification model
classifier = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")

# Define the labels
labels = ["Fear", "Anger", "Sadness", "Joy", "Neutral"]

def classify_comment(comment):

    # Perform zero-shot classification
    result = classifier(comment, labels)
    
    # Extract the labels and scores
    label_scores = {}
    for label, score in zip(result["labels"], result["scores"]):
        label_scores[label] = score
    
    # Determine the activated label based on score
    return max(label_scores, key=label_scores.get)

def classify_file(file_name):

    input_file = f'csv_files/comments_by_state/{file_name}'
    output_file = f'csv_files/classified_comments_by_state/{file_name}'
    comments_df = pd.read_csv(input_file)

    # print(comments_df)

    comments_df['Sentiment'] = comments_df['Comment'].apply(classify_comment)
    comments_df.to_csv(output_file, index=False)

folder = 'csv_files/comments_by_state'
files = os.listdir(folder)

# print(files)

for file_name in files:
    classify_file(file_name)

data = pd.concat([pd.read_csv(f'csv_files/classified_comments_by_state/{file}') for file in files])
data.to_csv('csv_files/classified_comments.csv', index=False)