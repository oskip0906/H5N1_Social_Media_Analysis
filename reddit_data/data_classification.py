import pandas as pd
from transformers import pipeline
import os

classifier = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")
labels = ["informative", "frustrated", "afraid", "skeptical", "critical", "supportive", "concerned", "unconcerned", "optimistic"]

def classify_comment(comment):
    
    result = classifier(comment, labels)
    return result['labels'][0]

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