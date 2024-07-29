import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional as F
import torch
import os
import zipfile
import shutil
import gc

device = torch.device('cuda')

# Unzip the zip files
with zipfile.ZipFile('sentiment_analysis/model/saved_model.zip', 'r') as zip_ref:
    zip_ref.extractall('sentiment_analysis/model')

with zipfile.ZipFile('sentiment_analysis/model/saved_tokenizer.zip', 'r') as zip_ref:
    zip_ref.extractall('sentiment_analysis/model')

# Load the model and tokenizer

# Load the model and tokenizer
classifier = AutoModelForSequenceClassification.from_pretrained('sentiment_analysis/model/saved_model').to(device)
tokenizer = AutoTokenizer.from_pretrained('sentiment_analysis/model/saved_tokenizer')

# Define the labels
labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise', 'Neutral']

def classify_comment(comment):

    tokenized_comment = tokenizer(comment, return_tensors='pt', padding='max_length', truncation=True).to(device)

    # Perform classification
    with torch.no_grad():
        outputs = classifier(**tokenized_comment)
        logits = outputs.logits

    # Get the predicted class and the maximum score
    score = F.softmax(logits, dim=-1)
    prediction = torch.argmax(score, dim=-1).item()
    max_score = torch.max(score).item()

    # If the maximum score low, classify as 'Neutral'
    if max_score < 0.5:
        prediction = 6 

    print(f'Prediction: {labels[prediction]} | Score: {max_score}')
    
    return labels[prediction], max_score

def classify_file(file_name):
    input_file = f'csv_files/comments_by_state/{file_name}'
    output_file = f'csv_files/classified_comments_by_state/{file_name}'
    comments_df = pd.read_csv(input_file)

    # Apply the classifier to each comment
    sentiments = comments_df['Comment'].apply(lambda comment: classify_comment(comment))
    
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

del classifier
del tokenizer
gc.collect()

# Delete the extracted folders
shutil.rmtree('sentiment_analysis/model/saved_model')
shutil.rmtree('sentiment_analysis/model/saved_tokenizer')