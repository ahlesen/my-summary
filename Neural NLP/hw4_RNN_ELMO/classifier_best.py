
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

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from torchtext.vocab import Vectors

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR


# device = torch.device( 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# load the embeddings
def load_embeddings(emb_path, vocab):
    clf_embeddings = {}
    emb_vocab = set()
    for line in open(emb_path):
        line = line.strip('\n').split()
        word, emb = line[0], line[1:]
        emb = [float(e) for e in emb]
        if word in vocab:
            clf_embeddings[word] = emb
    for w in vocab:
        if w in clf_embeddings:
            emb_vocab.add(w)
    word2idx = {w: idx for (idx, w) in enumerate(emb_vocab)}
    max_val = max(word2idx.values())
    
    word2idx['UNK'] = max_val + 1
    word2idx['EOS'] = max_val + 2
    emb_dim = len(list(clf_embeddings.values())[0])
    clf_embeddings['UNK'] = [0.0 for i in range(emb_dim)]
    clf_embeddings['EOS'] = [0.0 for i in range(emb_dim)]
    
    embeddings = [[] for i in range(len(word2idx))]
    for w in word2idx:
        embeddings[word2idx[w]] = clf_embeddings[w]
    embeddings = torch.Tensor(embeddings)
    return embeddings, word2idx

def to_matrix(lines, vocab, max_len=None, dtype='int32'):
    """Casts a list of lines into a matrix"""
    pad = vocab['EOS']
    max_len = max_len or max(map(len, lines))
    lines_ix = np.zeros([len(lines), max_len], dtype) + pad
    for i in range(len(lines)):
        line_ix = [vocab.get(l, vocab['UNK']) for l in lines[i]]
        lines_ix[i, :len(line_ix)] = line_ix
    lines_ix = torch.LongTensor(lines_ix)
    return lines_ix

def generate_data(train_tok,vocab,label_enc=None,with_label=True):
    data = []
    if with_label:
        for t, l in zip(train_tok,label_enc):
            t = to_matrix([t], vocab)
            l = torch.Tensor([l])
            data.append((t, l))
    else:
        for t in train_tok:
            t = to_matrix([t], vocab)
            data.append(t)
    return data

def binary_accuracy(preds, y):
    # y is either [0, 1] or [1, 0]
    # get the class (0 or 1)
    y = torch.argmax(y, dim=1)
    
    # get the predicted class
    preds = torch.argmax(torch.sigmoid(preds), dim=1)
    
    correct = (preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

class BiLSTM(nn.Module):
    def __init__(self, embeddings, hidden_dim=128, lstm_layer=1, output=2):
        
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # load pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        # embeddings are not fine-tuned
        self.embedding.weight.requires_grad = False
        
        # RNN layer with LSTM cells
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            bidirectional=True)
        # dense layer
        # self.linear = nn.Linear(hidden_dim*2, hidden_dim//2)
        self.output = nn.Linear(hidden_dim*2, output)

    
    def forward(self, sents):
        x = self.embedding(sents)
        
        # the original dimensions of torch LSTM's output are: (seq_len, batch, num_directions * hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # reshape to get the tensor of dimensions (seq_len, batch, num_directions, hidden_size)
        lstm_out = lstm_out.view(x.shape[0], -1, 2, self.hidden_dim)#.squeeze(1)
        
        # lstm_out[:, :, 0, :] -- output of the forward LSTM
        # lstm_out[:, :, 1, :] -- output of the backward LSTM
        # we take the last hidden state of the forward LSTM and the first hidden state of the backward LSTM
        dense_input = torch.cat((lstm_out[-1, :, 0, :], lstm_out[0, :, 1, :]), dim=1)
        
        y = self.output(dense_input).view([1, 2])

        # output = self.linear(dense_input)
        # y = self.output(output).view([1, 2])
        return y

def train_epoch(model, train_data, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    # set the model to the training mode
    model.train(mode=True)
    
    for t, l in train_data:
        # reshape the data to n_words x batch_size (here batch_size=1)
        t = t.view((-1, 1))
        # transfer the data to GPU to make it accessible for the model and the loss
        t = t.to(device)
        l = l.to(device)
        
        # set all gradients to zero
        optimizer.zero_grad()
        
        # forward pass of training
        # compute predictions with current parameters
        predictions = model(t)
        # compute the loss
        loss = criterion(predictions, l)
        # compute the accuracy (this is only for report)
        acc = binary_accuracy(predictions, l)
        
        # backward pass (fully handled by pytorch)
        loss.backward()
        # update all parameters according to their gradients
        optimizer.step()
        
        # data for report
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(train_data), epoch_acc / len(train_data)


def train(train_texts, train_labels):
    # my params
    hidden_dim = 128
    layers = 1
    LEARNING_RATE = 1e-3
    N_EPOCHS = 3
    emb_dim = 300
    ##############

    train_texts = preprocess(train_texts, punctuation=True)
    train_texts_tok = tokenization_gram(train_texts)

    y_train = np.array([1 if label == 'pos' else 0 for label in train_labels ])
    
    encoder = OneHotEncoder(categories=[range(2)], sparse=False)
    y_enc_train = encoder.fit_transform(y_train.reshape(-1, 1))

    dict_train = make_vocab_ngram(train_texts_tok,mode='any')
    vocab_train = set(dict_train.keys())

    embeddings, vocab_train = load_embeddings(f'glove.6B.{emb_dim}d.txt', vocab_train)
    # glove_model =  Vectors(f'glove.6B.{emb_dim}d.txt', cache='./embeddings/')
    train_data = generate_data(train_texts_tok, vocab=vocab_train, label_enc=y_enc_train)

    model = BiLSTM(embeddings, hidden_dim, lstm_layer=layers)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()


    model = model.to(device)
    criterion = criterion.to(device)    

    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train_epoch(model, train_data, optimizer, criterion)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    params = [model,vocab_train]
    return params


def predict(model, test_data):
    model.eval()

    y_pred_all = np.array([])

    for t in test_data:
        t = t.view((-1, 1))
        t = t.to(device)

        predictions = model(t)

        y_pred_all =  np.append(y_pred_all,np.argmax(predictions.cpu().detach().numpy(),axis=1))
    
    return y_pred_all

def classify(texts, params):
#def classify(texts, model=params[0], BATCH_SIZE = param[1]):
    model = params[0]
    vocab = params[1]
    probs = []

    texts = preprocess(texts, stop_words=True, punctuation=True)
    texts_tok = tokenization_gram(texts, gram_range = (1,1))

    test_data = generate_data(texts_tok, vocab,with_label=False)


    y_pred_all = predict(model,test_data)
        
    for pred in y_pred_all:
        if pred:
            probs.append('pos')
        else:
            probs.append('neg')
    return probs