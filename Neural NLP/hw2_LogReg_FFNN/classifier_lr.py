import string
import re
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook

from scipy.sparse import csr_matrix, hstack, issparse, coo_matrix

from IPython.display import clear_output
import matplotlib.pyplot as plt


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
    for value in text:
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


def make_vocab_ngram(text, mode='ber',gram_range=(1,1), min_cnt=0):
    vocab = dict()
    
    if mode == 'ber':    
        for line in text:
            for word in set(ngrams(line,gram_range)):     
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] +=1
    else:
        for line in text:
            for word in ngrams(line,gram_range):     
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] +=1
                    
    if min_cnt:
        new_dict = dict()
        for key,val in vocab.items():
            if val > min_cnt:
                new_dict[key] = val
        vocab = new_dict
    return vocab

def ngrams(words, gram_range):
    pp = []
    for igram in range(gram_range[0], gram_range[1] + 1):
        for i in range(len(words) - igram + 1):
            pp.append(" ".join(words[i:i + igram]))

    return pp

def tokenization_gram(text, gram_range=(1, 1)):
    return [ngrams(line.split(), gram_range=gram_range) for line in text]

def to_mtx(text, vocab,with_vocab=False):
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for line in text:
        for word in line:
            if word in vocab:
                index = vocabulary.setdefault(word, len(vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))
    if with_vocab:
        return csr_matrix((data, indices, indptr), dtype=int),vocabulary
    return csr_matrix((data, indices, indptr), dtype=int)

def to_mtx_test(text, vocab, voc_train, with_vocab=False):
    indptr = [0]
    indices = []
    data = []
    vocabulary = voc_train
    for line in text:
        for word in line:
            if word in vocab:
                index = vocabulary.setdefault(word, len(vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))
    if with_vocab:
        return csr_matrix((data, indices, indptr), dtype=int),vocabulary
    return csr_matrix((data, indices, indptr),shape=[len(text),len(vocab)], dtype=int)

def sigmoid(z): 
    z = np.exp(-z)  
    return np.divide(1,1+z)

def initialize_weights(X):
    N = X.shape[1]
    return np.zeros(N)

def SGD(W, gradient, lr):
    return W - lr*gradient

def gradient(W, X, y,alpha, with_reg=True):
    return X.T@(sigmoid(X@W)-y)/X.shape[0] +2*alpha*W

def accuracy(pred,y_true):
    return len(np.where(np.array(y_true) == np.array(pred))[0])/len(y_true)

def train(train_texts, train_labels, mode='br', gram_range=(1, 1), alpha=1e-3, lr=0.01,n_epoch=30000, early_stop= 500, with_plot=True):


    train_texts = preprocess(train_texts, stop_words=True, punctuation=True)
    train_texts_tok = tokenization_gram(train_texts)

    vocab_1_1 = make_vocab_ngram(train_texts_tok,gram_range=gram_range)

    X_train, vocabulary = to_mtx(train_texts_tok, vocab_1_1,with_vocab=True)
    
    y_train = np.array([1 if label == 'pos' else 0 for label in train_labels ])
    
    
    W = initialize_weights(X_train)
    
    losses_tr = []
    accs_tr = []
    
    best__val_loss = 1
    diff_val_loss = 0
    
    for epoch in tqdm_notebook(range(n_epoch)):
        #train 
        z = X_train@W
        h = sigmoid(z)
        
        loss = -np.mean(y_train*np.log(h+1e-10)+(1-y_train)*np.log(1-h+1e-10))+np.sum(alpha*W*W)
        grads = gradient(W, X_train, y_train,alpha)
        W = SGD(W,grads,lr) 
        
        
        if loss < best__val_loss:
            best__val_loss = loss
            
            diff_val_loss = 0
        else:
            diff_val_loss +=1
        
        if diff_val_loss == early_stop:
            print(f'Early stopping on epoch = {epoch} with best accuracy on dev = {np.max(accs_tr)}')
            
        if(epoch % 100 == 0):
            losses_tr.append(loss)
            accs_tr.append(accuracy(np.round(h),y_train))
            


    params = [W, vocab_1_1, vocabulary]
    return params

def classify(texts,params):
    W=params[0]
    vocab=params[1]
    vocabulary=params[2]
    gram_range=(1, 1)

    probs = []
    texts = preprocess(texts, stop_words=True, punctuation=True)
    texts_tok = tokenization_gram(texts, gram_range = gram_range)
    
    X_dev = to_mtx_test(texts_tok, vocab, vocabulary)
    
    z = X_dev@W
    preds = sigmoid(z)
    
    preds = np.round(preds)
        
    for pred in preds:
        if pred:
            probs.append('pos')
        else:
            probs.append('neg')
    return probs