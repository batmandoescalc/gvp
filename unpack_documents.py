import os
import glob
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Custom Porter Stemmer that protects specific pronouns from being stemmed
class CustomPorterStemmer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        # Define the exact words you want to protect
        self.exceptions = {
            "his": "him",
            "himself": "him",
            "hisself": "him",
            "hers": "her",
            "herself": "her",
            "he'd": "he",
            "he'll": "he",
            "he's": "he",
            "she'd": "she",
            "she'll": "she",
            "she's": "she"
        }

    def stem(self, word):
        # Normalize to lowercase to match exceptions
        lowered_word = word.lower()
        if lowered_word in self.exceptions:
            return self.exceptions[lowered_word]
        return self.stemmer.stem(word)


# Load documents
def load_documents(filepaths):
    corpus = []
    filenames = []
    for fp in filepaths:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            corpus.append(f.read())
            filenames.append(os.path.basename(fp))
    print (f"Loaded {len(corpus)} documents")

    return corpus, filenames


def clean_titles_from_filenames(filenames):
    """clean up the filenames to extract titles for data merges"""
    titles = []
    
    no_num = [re.sub(r'^[^A-Za-z]+', '', s) for s in filenames]  # remove leading non-letters

    no_symbols = [re.sub(r'[^A-Za-z0-9 ]', '', s) for s in no_num]  # remove punctuation

    titles = [s[:-3].lower() for s in no_symbols]  # remove "txt"

    return titles


def clean_titles_from_csv(df):
    df['cleaned_title'] = (
        df['Title']
        .str.lower()
        .apply(
            lambda x: unicodedata.normalize('NFKD', x)
            .encode('ascii', 'ignore')
            .decode('ascii')
            if isinstance(x, str) else x
        )
        .str.replace(r'[^A-Za-z0-9 ]', '', regex=True)
    )
    return df


# Preprocess text: lowercase, remove punctuation, tokenize, stem
def preprocess(text, stemmer=CustomPorterStemmer()):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
    tokens = word_tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    return " ".join(stems)

