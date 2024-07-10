from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan 
import nltk
import string
import pandas as pd
import json
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')  

def preprocess_comment(comment):

    # consider context of words
    lemmatizer = WordNetLemmatizer()
    comment = lemmatizer.lemmatize(comment)

    # remove punctuation and lowercase all words
    translator = str.maketrans('', '', string.punctuation)
    comment = comment.translate(translator).lower()

    # tokenize and remove stopwords
    tokens = nltk.word_tokenize(comment)

    filtered_tokens = []
    for word in tokens:
        if word not in stopwords.words('english'):
            filtered_tokens.append(word)

    return ' '.join(filtered_tokens)

def topic_modeling(comments, output):

    processed_comments = []
    for comment in comments:
        processed_comments.append(preprocess_comment(comment))

    # print(processed_comments)

    # HDBSCAN model for clustering 
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=5, 
        min_samples=5, 
        cluster_selection_epsilon=0.5
    )

    # BERTopic model with adjusted parameters
    topic_model = BERTopic(
        hdbscan_model=hdbscan_model, 
        embedding_model='all-MiniLM-L6-v2', 
        vectorizer_model=CountVectorizer(ngram_range=(2, 5)),
        low_memory=True
    )

    # fit-transform the model
    topic_model.fit_transform(processed_comments)

    topics_list = {}

    i = 0
    while topic_model.get_topic(i) != False:
        subtopics = topic_model.get_topic(i)
        subtopics_list = []
        for subtopic in subtopics:
            subtopics_list.append(subtopic[0])
        topics_list[f"Topic {i + 1}"] = subtopics_list
        i += 1

    # save the topics to a JSON file
    with open(f'json_files/{output}.json', 'w') as f:
        json.dump(topics_list, f, indent=4)

    if output == 'all_topics':
        topic_model.save(f'reddit_data/topic_modeling/topic_model')

# all data
data = pd.read_csv('csv_files/classified_comments.csv')
comments = data['Comment'].tolist()
topic_modeling(comments, 'all_topics')

# data by state
folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

for file in files:
    state = file[:-4].split('/')[-1]
    state_data = pd.read_csv(f'{folder}/{file}')
    state_comment = state_data['Comment'].tolist()
    topic_modeling(state_comment, state)