import pandas as pd
from transformers import pipeline
import os

# Load the zero-shot classification model
classifier = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")

# Define the labels
labels = ["Fear", "Anger", "Joy", "Neutral"]

def classify_comment(comment, classifier, labels):
    # Perform zero-shot classification
    result = classifier(comment, labels)
    
    # Extract the labels and scores
    label_scores = {}
    for label, score in zip(result["labels"], result["scores"]):
        label_scores[label] = score
    
    # Determine the activated label based on the highest score
    activated_label = max(label_scores, key=label_scores.get)

    return activated_label, label_scores[activated_label]

def classify_file(file_name):
    input_file = f'csv_files/comments_by_state/{file_name}'
    output_file = f'csv_files/classified_comments_by_state/{file_name}'
    comments_df = pd.read_csv(input_file)

    # Apply the classifier to each comment
    sentiments = comments_df['Comment'].apply(lambda comment: classify_comment(comment, classifier, labels))
    
    # Extract sentiment and intensity from the results
    comments_df['Sentiment'] = sentiments.apply(lambda x: x[0])
    comments_df['Intensity'] = sentiments.apply(lambda x: x[1])

    # Save the classified comments
    comments_df.to_csv(output_file, index=False)

# Specify the folder containing the files
folder = 'csv_files/comments_by_state'
files = os.listdir(folder)

# Classify comments in each file
for file_name in files:
    classify_file(file_name)

# Concatenate all classified data into a single CSV
data = pd.concat([pd.read_csv(f'csv_files/classified_comments_by_state/{file}') for file in files])
data.to_csv('csv_files/classified_comments.csv', index=False)