import string
import re
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook

from scipy.sparse import csr_matrix, hstack, issparse, coo_matrix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import OneHotEncoder

from IPython.display import clear_output
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV



punct = string.punctuation + '…–—‘“‚„«»'

stopwords_eng = set(["them", "she", "wasn", "wasn't", "having", "ll", "hadn", "nor", "why", "these", "she's", "both",
                     "you're", "an", "between", "myself", "because", "being", "ve", "while", "over", "whom", "isn",
                     "shouldn", "wouldn't", "been", "d", "themselves", "does", "most", "below", "his", "you'll",
                     "further", "there", "was", "ain", "doesn't", "each", "couldn", "which", "that'll", "down", "won't",
                     "than", "y", "should've", "have", "until", "their", "through", "ma", "before", "is", "yours", "so",
                     "up", "hasn", "doesn", "him", "very", "if", "mustn", "or", "it's", "too", "re", "mustn't", "as",
                     "now", "isn't", "mightn't", "those", "other", "above", "who", "do", "the", "wouldn", "some",
                     "this", "for", "don", "me", "any", "what", "theirs", "weren't", "mightn", "aren", "ours", "your",
                     "didn't", "shan", "shouldn't", "off", "has", "just", "himself", "herself", "m", "we", "by",
                     "aren't", "yourselves", "again", "after", "you've", "you", "how", "such", "can", "o", "were",
                     "not", "they", "out", "few", "with", "i", "be", "haven't", "are", "s", "hadn't", "had", "our",
                     "don't", "in", "needn't", "but", "during", "weren", "it", "will", "a", "did", "of", "shan't",
                     "ourselves", "and", "no", "doing", "yourself", "at", "then", "you'd", "he", "once", "about",
                     "where", "more", "only", "into", "same", "my", "hers", "t", "when", "its", "own", "here", "all",
                     "won", "needn", "under", "br", "that", "couldn't", "from", "against", "itself", "am", "should",
                     "her", "on", "to", "didn", "haven", "hasn't","i", "me", "my", "myself", "we", "our", "ours",
                     "ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't", "br"])


def preprocess(text, stop_words=False, punctuation=False):
    """
    Preprocess the data

    stop_words: flag to remove stopwords
    punctuation: flag to remove punctuation
    """
    new_text = []
    for num, value in enumerate(text):
        value = value.lower()

        for char in set(string.punctuation):
            value = value.replace(char, " " + char + " ")

        if stop_words:
            pp = " ".join([word for word in value.split() if word not in stopwords_eng])
            if punctuation:
                punctn = string.punctuation + '…–—‘“‚„«»'
                pp = pp.translate(str.maketrans('', '', punctn))
                new_text.append(" ".join(pp.split()))
            else:
                new_text.append(pp)
        else:
            new_text.append(" ".join(value.split()))  # remove multiple spaces

    return new_text


def ngrams(words, gram_range):
    pp = []
    for igram in range(gram_range[0], gram_range[1] + 1):
        for i in range(len(words) - igram + 1):
            pp.append(" ".join(words[i:i + igram]))

    return pp

def tokenization_gram(text, gram_range=(1, 1)):
    return [ngrams(line.split(), gram_range=gram_range) for line in text]



def train(train_texts, train_labels):
    emb_dim = 300
    train_texts = preprocess(train_texts, punctuation=True)
    train_texts_tok = tokenization_gram(train_texts)

    y_train = np.array([1 if label == 'pos' else 0 for label in train_labels ])
    
    documents_train = [TaggedDocument(words=doc, tags=[tag]) for doc,tag in zip(train_texts_tok,y_train)]
    
    model_dbow = Doc2Vec(vector_size=emb_dim,negative=5, min_count=2, epochs=10)
    model_dbow.build_vocab(documents_train)

    model_dbow.train(documents_train, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

    X_train = [model_dbow.infer_vector(doc.words) for doc in documents_train]
 
    logreg = LogisticRegression(random_state=42, n_jobs=-1, C=0.1)
    logreg.fit(X_train,y_train)
    
    params = [model_dbow,logreg]
    return params


def classify(texts,params):
#def classify(texts,

    model_dbow=params[0]
    logreg=params[1]
    
    probs = []
    texts = preprocess(texts, stop_words=True, punctuation=True)
    texts_tok = tokenization_gram(texts, gram_range = (1,1))
    
    X_test = [model_dbow.infer_vector(doc) for doc in texts_tok]
    

    y_pred_all = logreg.predict(X_test)
        
    for pred in y_pred_all:
        if pred:
            probs.append('pos')
        else:
            probs.append('neg')
    return probs