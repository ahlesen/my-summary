
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
import torch.utils.data as data_utils

from sklearn.metrics import accuracy_score
from allennlp.modules.elmo import Elmo, batch_to_ids


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

class BiLSTMTagger(nn.Module):
    def __init__(self, emb_dim, hidden_dim=128, lstm_layer=1, output=1):
        
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        # RNN layer with LSTM cells
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            bidirectional=True)
        # dense layer
        self.linear = nn.Linear(hidden_dim*2, hidden_dim//2)
        self.output = nn.Linear(hidden_dim//2, output)

    
    def forward(self, sents):
        # x = self.embedding(sents)
        
        # the original dimensions of torch LSTM's output are: (seq_len, batch, num_directions * hidden_size)
        lstm_out, _ = self.lstm(sents)
        
        # reshape to get the tensor of dimensions (seq_len, batch, num_directions, hidden_size)
        lstm_out = lstm_out.view(sents.shape[1], -1, 2, self.hidden_dim)#.squeeze(1)#sents.shape[0]
        
        # lstm_out[:, :, 0, :] -- output of the forward LSTM
        # lstm_out[:, :, 1, :] -- output of the backward LSTM
        # we take the last hidden state of the forward LSTM and the first hidden state of the backward LSTM
        # dense_input = torch.cat((lstm_out[-1, :, 0, :], lstm_out[0, :, 1, :]), dim=1)
        dense_input = torch.cat((lstm_out[:,-1,0,:], lstm_out[:,0,1,:]), dim=1)
        # y = self.output(dense_input).view([1, 2])

        output = self.linear(dense_input)
        y = self.output(output)
        return y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_epoch(model,elmo, train_data, optimizer, criterion,epoch_acc=[],epoch_loss=[]):
    
    # set the model to the training mode
    model.train(mode=True)
    
    for t, l in train_data:
        # reshape the data to n_words x batch_size (here batch_size=1)
        # t = t.view((-1, 1))
        # transfer the data to GPU to make it accessible for the model and the loss
        t = batch_to_ids(t).to(device)
        t_elmo = elmo(t)['elmo_representations'][0]
        l = l.to(device)
        
        # set all gradients to zero
        optimizer.zero_grad()
        
        # forward pass of training
        # compute predictions with current parameters
        predictions = model(t_elmo)

        # compute the loss
        loss = criterion(predictions, l)
        # compute the accuracy (this is only for report)
        y_pred=torch.sigmoid(predictions).detach().cpu().numpy().round()
        acc = accuracy_score(y_pred,l.detach().cpu().numpy())
        # print('y_pred,l,acc ',y_pred,l, acc)
        # backward pass (fully handled by pytorch)
        loss.backward()
        # update all parameters according to their gradients
        optimizer.step()
        
        # data for report
        epoch_loss.append(loss.item())
        epoch_acc.append(acc.item())
        
    return epoch_loss, epoch_acc 


def train(train_texts, train_labels):
    # my params
    hidden_dim = 128
    layers = 1
    LEARNING_RATE = 1e-3
    N_EPOCHS = 100
    emb_dim = 512
    ##############

    train_texts = preprocess(train_texts, punctuation=True)
    train_texts_tok = tokenization_gram(train_texts)

    y_train = np.array([1 if label == 'pos' else 0 for label in train_labels ])
    
    encoder = OneHotEncoder(categories=[range(2)], sparse=False)
    y_enc_train = encoder.fit_transform(y_train.reshape(-1, 1))

    dict_train = make_vocab_ngram(train_texts_tok,mode='any')
    vocab_train = set(dict_train.keys())

    ELMO_OPTIONS = "models/elmo_2x2048_256_2048cnn_1xhighway_options.json"
    ELMO_WEIGHT = "models/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
    elmo = Elmo(ELMO_OPTIONS, ELMO_WEIGHT, num_output_representations = 1)
    elmo.to(device)

    model = BiLSTMTagger(emb_dim=emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss().to(device)

    train_data=[[text, torch.Tensor([y])] for (text, y) in zip(train_texts_tok, y_train)]

    train_dataloader=DataLoader(train_data,batch_size=256,shuffle=False)


    model = model.to(device)
    criterion = criterion.to(device)    

    best_valid_loss = float('inf')

    train_loss_history = []
    valid_loss_history = []

    train_acc_history = []
    valid_acc_history = []

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model,elmo, train_data, optimizer, criterion)
        
        train_loss_history.append(np.mean(train_loss))
        train_acc_history.append(np.mean(train_acc))
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        

        clear_output(True)
        plt.figure(figsize=(14, 7))
        plt.subplot(121)
        plt.plot(np.arange(len(train_loss_history)) + 1, train_loss_history, label='loss on training')
        plt.ylabel('loss')
        plt.xlabel('epoch number')
        plt.legend()
        plt.grid()

        plt.subplot(122)
        plt.plot(np.arange(len(train_acc_history)) + 1, train_acc_history, label='accuracy on training')
        plt.ylabel('accuracy')
        plt.xlabel('epoch number')
        plt.legend()
        plt.grid()
        plt.show()

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {np.mean(train_loss):.3f} | Train Acc: {np.mean(train_acc)*100:.2f}%')

    params = [model]
    return params


def predict(model, elmo, test_data):
    
    epoch_acc=[]
    epoch_loss=[]
    
    model.eval()
    all_outputs = np.array([])

    with torch.no_grad():
        for t in test_data:
            # print(t)
            # reshape the data to n_words x batch_size (here batch_size=1)
            # t = t.view((-1, 1))
            # transfer the data to GPU to make it accessible for the model and the loss
            t = batch_to_ids(t).to(device)
            t_elmo = elmo(t)['elmo_representations'][0]
            
            # forward pass of training
            # compute predictions with current parameters
            predictions = model(t_elmo)

            preds=torch.sigmoid(predictions).detach().cpu().numpy().round()

            all_outputs = np.append(all_outputs,preds)
        
    return all_outputs

def classify(texts, params):
#def classify(texts, model=params[0], BATCH_SIZE = param[1]):
    model = params[0]
    probs = []

    texts = preprocess(texts, stop_words=True, punctuation=True)
    texts_tok = tokenization_gram(texts, gram_range = (1,1))
    ELMO_OPTIONS = "models/elmo_2x2048_256_2048cnn_1xhighway_options.json"
    ELMO_WEIGHT = "models/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
    
    elmo = Elmo(ELMO_OPTIONS, ELMO_WEIGHT, num_output_representations = 1)
    elmo.to('cuda')
    test_dataloader=DataLoader(texts_tok,batch_size=256,shuffle=False)


    y_pred_all = predict(model,elmo,test_dataloader)
        
    for pred in y_pred_all:
        if pred:
            probs.append('pos')
        else:
            probs.append('neg')
    return probs