from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import hdbscan 
import nltk
import string
import pandas as pd
import json

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')  

data = pd.read_csv('csv_files/classified_comments.csv')
comments = data['Comment'].tolist()

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

processed_comments = []

for comment in comments:
    processed_comments.append(preprocess_comment(comment))

# hdbscan model for clustering 
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10, 
    min_samples=10, 
    cluster_selection_epsilon=0.1
)

# bertopic model with adjusted parameters
topic_model = BERTopic(
    hdbscan_model=hdbscan_model, 
    embedding_model='all-MiniLM-L6-v2', 
    nr_topics=10,
    n_gram_range=(2, 5),
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

# print(topics_list)

with open('reddit_data/topic_modeling/topics.json', 'w') as f:
    json.dump(topics_list, f, indent=4)

topic_model.save('reddit_data/topic_modeling/topic_model')