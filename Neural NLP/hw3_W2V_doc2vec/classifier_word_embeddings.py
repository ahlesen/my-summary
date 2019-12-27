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

from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR


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


def ngrams(words, gram_range):
    pp = []
    for igram in range(gram_range[0], gram_range[1] + 1):
        for i in range(len(words) - igram + 1):
            pp.append(" ".join(words[i:i + igram]))

    return pp

def tokenization_gram(text, gram_range=(1, 1)):
    return [ngrams(line.split(), gram_range=gram_range) for line in text]


class СurrentDataset(Dataset):
    def __init__(self, text_data, model, target_data=None):
        self.target_data = target_data

        self.model = model

        self.data = torch.tensor([list(self.get_phrase_embedding(l)) for l in text_data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.target_data is not None:
            return self.data[i], self.target_data[i]
        else:
            return self.data[i]

    def get_phrase_embedding(self, train_texts_token):
        """
        Convert phrase to a vector by aggregating it's word embeddings. See description above.
        """

        vector = np.zeros(self.model.vectors.shape[1], dtype=np.float32)

        used_words = 0

        for word in train_texts_token:
            if word in self.model:
                vector += self.model[word]
                used_words += 1

        if used_words > 0:
            vector = vector / used_words

        return vector

class LogRegEmb(nn.Module):
    def __init__(self, num_labels,vocab_size):
        super(LogRegEmb,self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, inputs):
        return self.linear(inputs)
    
class FFNNBow(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_layer_size=8):
        super().__init__()
        self.hidden_layer = nn.Linear(vocab_size, hidden_layer_size)
        self.relu = nn.Tanh()
        self.output_layer = nn.Linear(hidden_layer_size, num_classes)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        
        self.hidden_layer.weight.data.uniform_(-initrange, initrange)
        self.hidden_layer.bias.data.zero_()
        
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
    
    def forward(self, features):
        intermid = self.hidden_layer(features)
        intermid = self.relu(intermid)
        intermid = self.output_layer(intermid)
        output = F.log_softmax(intermid, dim=1) #output = intermid #
        return output


def train(train_texts, train_labels):
    # my params
    hidden_size = 32
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    N_EPOCHS = 500
    emb_dim = 300
    ##############

    train_texts = preprocess(train_texts, punctuation=True)
    train_texts_tok = tokenization_gram(train_texts)

    y_train = np.array([1 if label == 'pos' else 0 for label in train_labels ])
    
    encoder = OneHotEncoder(categories=[range(2)], sparse=False)
    y_enc_train = encoder.fit_transform(y_train.reshape(-1, 1))

    # glove_model =  Vectors(f'glove.6B.{emb_dim}d.txt', cache='./embeddings/')
    glove_input = 'glove.6B.300d.txt'
    word2vec_output = 'glove.6B.300d.w2vformat.txt'
    glove2word2vec(glove_input, word2vec_output)
    glove_model = KeyedVectors.load_word2vec_format("glove.6B.300d.w2vformat.txt", binary=False)
    NUM_LABELS = 2

    train_dataset = СurrentDataset(text_data=train_texts_tok, target_data=torch.FloatTensor(y_enc_train),model=glove_model)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # model = FFNNBow(emb_dim, NUM_LABELS,hidden_size) !!!!!!!!!!!!!!!!!!!!!ЕСЛИ ХОЧЕТСЯ ЗДЕСЬ ЗАПУСТИТЬ FFNN!!!!!!!!!!!
    model = LogRegEmb(NUM_LABELS, emb_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()


    sched1 = StepLR(optimizer, step_size=N_EPOCHS//2, gamma = 0.5)
    
    train_loss_history = []
    
    train_acc = []


    for epoch in range(N_EPOCHS):
        if epoch%50 == 0:
            print('epoch =',epoch)
        sched1.step()

        model.train()
        train_loss_epoch = []
        train_acc_epoch = []
        
        for X_train,Y_train in train_dataloader:       
            optimizer.zero_grad()

            log_probs = model(X_train)
            loss = criterion(log_probs, Y_train)
            loss.backward()
            optimizer.step()

            acc =accuracy_score(np.argmax(Y_train.detach().numpy(), axis=1),np.argmax(log_probs.detach().numpy(),axis=1))

            train_acc_epoch.append(acc)
            train_loss_epoch.append(loss.item())

        train_loss_history.append(np.mean(train_loss_epoch))
        train_acc.append(np.mean(train_acc_epoch))


    params = [model,BATCH_SIZE,glove_model]
    return params

def predict(model, test_dataloader, with_label=True):
    model.eval()

    y_pred_all = np.array([])

    if with_label:
        for X_test,y_test in test_dataloader:
            outputs = model(X_test)

            y_pred_all =  np.append(y_pred_all,np.argmax(outputs.cpu().detach().numpy(),axis=1))
    else:
        for X_test in test_dataloader:
            outputs = model(X_test)

            y_pred_all =  np.append(y_pred_all,np.argmax(outputs.cpu().detach().numpy(),axis=1))
    
    return y_pred_all

def classify(texts, params):
#def classify(texts, model=params[0], BATCH_SIZE = param[1]):
    model=params[0]
    BATCH_SIZE = params[1]
    glove_model = params[2]
    probs = []

    texts = preprocess(texts, stop_words=True, punctuation=True)
    texts_tok = tokenization_gram(texts, gram_range = (1,1))

    test_dataset = СurrentDataset(text_data=texts_tok,model=glove_model)
    
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_pred_all = predict(model,test_dataloader, with_label=False)
        
    for pred in y_pred_all:
        if pred:
            probs.append('pos')
        else:
            probs.append('neg')
    return probs