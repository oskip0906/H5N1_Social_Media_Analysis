import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
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

    vect = TfidfVectorizer(
        stop_words=stopwords.words('english'), 
        max_features=1000,
        ngram_range=(2, 5)
    )
    vect_text = vect.fit_transform(processed_comments)

    # Initialize LDA model
    lda_model = LatentDirichletAllocation(
        learning_method='online', 
        random_state=42, 
        max_iter=10
    )
    
    lda_model.fit_transform(vect_text)

    vocab = vect.get_feature_names_out()

    topics_list = {}

    for i, comp in enumerate(lda_model.components_):
        topic_words = [vocab[id] for id in comp.argsort()[-10:]]
        topics_list[f"Topic {i + 1}"] = topic_words

    # save the topics to a JSON file
    with open(f'topic_modeling/LDA_method/topics_by_state/{output}.json', 'w') as f:
        json.dump(topics_list, f, indent=4)

    # save the models
    with open(f'topic_modeling/LDA_method/models_by_state/{output}_topic_model', 'wb') as model_file:
        pickle.dump(lda_model, model_file)

# data by state
folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

for file in files:
    state = file[:-4].split('/')[-1]
    state_data = pd.read_csv(f'{folder}/{file}')
    state_comment = state_data['Comment'].tolist()
    topic_modeling(state_comment, state)