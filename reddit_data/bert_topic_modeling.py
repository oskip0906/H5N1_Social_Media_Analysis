from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import json

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('csv_files/classified_comments.csv')
comments = data['Comment'].tolist()

def preprocess_comment(comment):
    comment = re.sub(r'[^\w\s]', '', comment).lower()
    tokens = nltk.word_tokenize(comment)
    filtered_tokens = []
    for word in tokens:
        if word not in stopwords.words('english'):
            filtered_tokens.append(word)
    return ' '.join(filtered_tokens)

processed_comments = []

for comment in comments:
    processed_comments.append(preprocess_comment(comment))

# 3 to 5 words per subtopic
vectorizer_model = CountVectorizer(ngram_range=(3, 5), stop_words='english')

topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)
topic_model.fit_transform(processed_comments)

topics_list = {}

i = 0
while topic_model.get_topic(i) != False:
    subtopics = topic_model.get_topic(i)
    subtopics_list = []
    for subtopic in subtopics:
        subtopics_list.append(subtopic[0])
    topics_list[f"Topics Group {i + 1}"] = subtopics_list
    i += 1

# print(topics_list)

with open('json_files/topics.json', 'w') as f:
    json.dump(topics_list, f, indent=4)