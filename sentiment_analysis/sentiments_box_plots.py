import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from statannotations.Annotator import Annotator

# Define labels
sentiment_labels = ['Sadness', 'Joy', 'Anger', 'Fear']

folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

# Dataset for all states combined
comprehensive_data = pd.DataFrame();

# For each state
for file in files:

    data = pd.read_csv(f'{folder}/{file}')
    state = file[:-4].split('/')[-1]

    # Add data from each state to comprehensive data
    data['State'] = state
    comprehensive_data = pd.concat([comprehensive_data, data])

    data = data[data['Sentiment'].isin(sentiment_labels)]
    
    plt.figure(figsize=(12, 6))
    axis = sns.boxplot(x='Sentiment', y='Intensity', data=data)  
    plt.title(f'{state.capitalize()}')
    plt.xlabel('Sentiment Types')
    plt.ylabel('Sentiment Intensity')
    
    # Create pairs of sentiment types
    pairs = []
    for i, sentiment_1 in enumerate(sentiment_labels):
        for sentiment_2 in sentiment_labels[i + 1:]:
            pairs.append((sentiment_1, sentiment_2))
    
    # Add statistical annotations
    annotator = Annotator(axis, pairs, data=data, x='Sentiment', y='Intensity')
    annotator.configure(test='Mann-Whitney', text_format='simple', verbose=True).apply_and_annotate()
    
    plt.savefig(f'graphs/sentiments_intensities_box_plots/{state}.png')

# For overall data
data = pd.read_csv('csv_files/classified_comments.csv')
data = data[data['Sentiment'].isin(sentiment_labels)]

plt.figure(figsize=(12, 6))
axis = sns.boxplot(x='Sentiment', y='Intensity', data=data) 
plt.xlabel('Sentiment Types)')
plt.ylabel('Sentiment Intensity')
plt.legend(loc='upper left')

# Create pairs of sentiment types between states
pairs = []
for i, sentiment_1 in enumerate(sentiment_labels):
    for sentiment_2 in sentiment_labels[i + 1:]:
        pairs.append((sentiment_1, sentiment_2))

# Add statistical annotations
annotator = Annotator(axis, pairs, data=data, x='Sentiment', y='Intensity')
annotator.configure(test='Mann-Whitney', text_format='simple', verbose=True).apply_and_annotate()

plt.savefig('graphs/sentiments_intensities_box_plot_overall.png')

# For overall data with states seperated
comprehensive_data = comprehensive_data[comprehensive_data['Sentiment'].isin(sentiment_labels)]

plt.figure(figsize=(24, 24))
axis = sns.boxplot(x='State', y='Intensity', hue='Sentiment', data=comprehensive_data) 
plt.xlabel('States (With Sentiment Types)')
plt.ylabel('Sentiment Intensity')
plt.legend(loc='upper left')

# Create pairs of sentiment types between states
pairs = []
for i, state_1 in enumerate(comprehensive_data['State'].unique()):
    for state_2 in comprehensive_data['State'].unique()[i + 1:]:
        pairs.append((state_1, state_2))

# Add statistical annotations
annotator = Annotator(axis, pairs, data=comprehensive_data, x='State', y='Intensity')
annotator.configure(test='Mann-Whitney', text_format='simple', verbose=True).apply_and_annotate()

plt.savefig('graphs/sentiments_intensities_box_plot_all_states.png')