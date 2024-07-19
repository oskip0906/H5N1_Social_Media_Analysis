import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

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