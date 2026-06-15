import os
import glob
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer



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

# Preprocess text: lowercase, remove punctuation, tokenize, stem
def preprocess(text, stemmer=PorterStemmer()):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
    tokens = word_tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    return " ".join(stems)

