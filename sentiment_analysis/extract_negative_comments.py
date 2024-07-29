import pandas as pd
import os

folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

for file in files:

    data = pd.read_csv(f'{folder}/{file}')
    state = file[:-4].split('/')[-1]
    
    # Extract fear comments
    fear_comments = data[data["Sentiment"].isin(["Sadness", "Anger", "Fear"])] 

    fear_comments.to_csv(f"csv_files/negative_comments_by_state/{state}.csv", index=False)