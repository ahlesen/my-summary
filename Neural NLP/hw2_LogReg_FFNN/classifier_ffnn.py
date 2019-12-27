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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

device = torch.device( 'cpu')


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

# Fully connected neural network with one hidden layer
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size) 
        self.relu = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, num_classes)  
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        
        self.hidden_layer.weight.data.uniform_(-initrange, initrange)
        self.hidden_layer.bias.data.zero_()
        
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        
    def forward(self, x):
        out = self.hidden_layer(x)
        out = self.relu(out)
        out = self.output_layer(out)
        return out
    


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(targets))
    #for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
    for batch_idx in range(int(len(targets)/batchsize)):        
        start_idx = batch_idx*batchsize
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    
def evaluate_minibatch(model,X_test, y_test, criterion):
    test_loss_history = []
    test_acc = []
    y_pred_all = np.array([])
    
    model.eval()
    with torch.no_grad():
        for num_batch, (x_batch,y_batch) in enumerate(tqdm_notebook(iterate_minibatches(X_test, y_test, batchsize=batch_size, shuffle=False))):
            x_batch = torch.from_numpy(x_batch.todense()).float()
            y_batch = torch.from_numpy(y_batch).float()
            
            y_test_pred = model(x_batch)
            
            y_pred_all = np.append(y_pred_all,np.argmax(y_test_pred,axis=1).cpu().detach().numpy())
            
            test_loss = criterion(y_test_pred,y_batch)
            
            accuracy_batch = accuracy_score(np.argmax(y_batch, axis=1),np.argmax(y_test_pred,axis=1))
            test_acc.append(accuracy_batch)

            test_loss_history.append(test_loss)
#             print(test_acc[-1])
    return y_pred_all,test_loss_history,test_acc

def iterate_minibatch_predict(inputs, batchsize, shuffle=False):
    M, N = inputs.shape
    if shuffle:
        indices = np.random.permutation(M)
    #for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
    for batch_idx in range(int(M/batchsize)):        
        start_idx = batch_idx*batchsize
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]
        
def predict_minibatch(model,X_test,batch_size):
    y_pred_all = np.array([])
    model.eval()
    with torch.no_grad():
        for x_batch in tqdm_notebook(iterate_minibatch_predict(X_test,batchsize=batch_size, shuffle=False)):
            x_batch = torch.from_numpy(x_batch.todense()).float()
            
            y_test_pred = model(x_batch)
            
            y_pred_all = np.append(y_pred_all,np.argmax(y_test_pred,axis=1).cpu().detach().numpy())
    return y_pred_all


def train(train_texts, train_labels, hidden_size = 1024, learning_rate = 1e-3, batch_size = 100,n_epoch = 5):


    train_texts = preprocess(train_texts, stop_words=True, punctuation=True)
    train_texts_tok = tokenization_gram(train_texts)

    vocab_1_1 = make_vocab_ngram(train_texts_tok)

    X_train, vocabulary = to_mtx(train_texts_tok, vocab_1_1,with_vocab=True)
    
    y_train = np.array([1 if label == 'pos' else 0 for label in train_labels ])
    
    encoder = OneHotEncoder(categories=[range(2)], sparse=False)
    y_enc_train = encoder.fit_transform(y_train.reshape(-1, 1))
    
    _,input_size = X_train.shape
    num_classes = len(set(y_train))
    
    
    model = FFNN(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    
    train_loss_history = []
    
    train_acc = []

    
    for epoch in tqdm_notebook(range(n_epoch)):
        model.train()

        train_loss_epoch = []
        train_acc_epoch = []
        
        for num_batch, (x_batch,y_batch) in enumerate(tqdm_notebook(iterate_minibatches(X_train, y_train, batchsize=batch_size, shuffle=True))):
#             print(x_batch.shape,y_batch.shape)
            
            x_batch = torch.from_numpy(x_batch.todense()).float()
            y_batch = torch.from_numpy(y_batch).float()
            optimizer.zero_grad()
            
             # Forward
            predictions = model(x_batch)
#             print(predictions)
            loss = criterion(predictions, y_batch)
    
            # Backward
            loss.backward()
            optimizer.step()
            acc =accuracy_score(np.argmax(y_batch.detach().numpy(), axis=1),np.argmax(predictions.detach().numpy(),axis=1))
            
            train_acc_epoch.append(acc)
            train_loss_epoch.append(loss.item())

        train_loss_history.append(np.mean(train_loss_epoch))
        train_acc.append(np.mean(train_acc_epoch))
            
   
    params = [model,batch_size,vocab_1_1,vocabulary]
    return params

def classify(texts,params):
    model=params[0]
    batch_size = params[1]
    vocab=params[2]
    vocabulary=params[3]

    gram_range=(1, 1)
    
    probs = []
    texts = preprocess(texts, stop_words=True, punctuation=True)
    texts_tok = tokenization_gram(texts, gram_range =gram_range)
    
    X_test = to_mtx_test(texts_tok, vocab, vocabulary)
    
    y_pred_all = predict_minibatch(model,X_test,batch_size)
        
    for pred in y_pred_all:
        if pred:
            probs.append('pos')
        else:
            probs.append('neg')
    return probs