import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")


stopwords = stopwords.words("portuguese")
stm = PorterStemmer()
wnl = WordNetLemmatizer()


def custom_tokenizer(text, reduc="stemmer", min_len=3, stopwords=stopwords):
    text = str(text)
    words = wordpunct_tokenize(text)
    # Removendo pontuação e tornando lowercase
    words = [word.lower() for word in words if word.isalpha()]
    # Removendo stopwords
    words = [word for word in words if word not in stopwords]

    if reduc.lower() == "lemmatizer":
        words = [wnl.lemmatize(word) for word in words]
    else:
        words = [stm.stem(word) for word in words]

    words = [word for word in words if min_len <= len(word)]
    return words


# Funções separadas serão úteis posteriormente
def tokenizer_lemma(text, min_len=3):
    return custom_tokenizer(text, reduc="lemmatizer", min_len=min_len)


def tokenizer_stem(text, min_len=3):
    return custom_tokenizer(text, reduc="stemmer", min_len=min_len)


def map_progress(func, iterable):
    tqdm_iter = tqdm(iterable)
    return list(map(func, tqdm_iter))
