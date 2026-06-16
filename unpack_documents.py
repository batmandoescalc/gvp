import os
import glob
import re
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
    titles = []
    for fp in filepaths:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            corpus.append(f.read())
            filenames.append(os.path.basename(fp))
    print (f"Loaded {len(corpus)} documents")

    # clean up the filenames to extract titles for data merges
    no_num = [re.sub(r'^[^A-Za-z]+', '', s) for s in filenames]  # remove leading non-letters
    titles = [s[:-4] for s in no_num]  # remove ".txt"
    return corpus, filenames, titles


# Preprocess text: lowercase, remove punctuation, tokenize, stem
def preprocess(text, stemmer=CustomPorterStemmer()):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
    tokens = word_tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    return " ".join(stems)

