from os import listdir
from math import log
import string
import re
from statistics import mean, median
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook

from collections import Counter, defaultdict


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


def make_vocab_ngram(text, mode='ber', gram_range=(1, 1)):
    vocab = dict()

    if mode == 'ber':
        for line in text:
            for word in set(ngrams(line, gram_range)):
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
    else:
        for line in text:
            for word in ngrams(line, gram_range):
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
    return vocab

def ngrams(words, gram_range):
    pp = []
    for igram in range(gram_range[0], gram_range[1] + 1):
        for i in range(len(words) - igram + 1):
            pp.append(" ".join(words[i:i + igram]))

    return pp

def tokenization_gram(text, gram_range=(1, 1)):
    return [ngrams(line.split(), gram_range=gram_range) for line in text]


def train(train_texts, train_labels, mode='br', gram_range=(1, 1), alpha=1):
    def pos_neg_reviews(sentiment, train_texts, train_labels):
        tr_labl = np.array(train_labels)
        tr_text = np.array(train_texts)
        return list(tr_text[np.where(tr_labl == sentiment)[0]])

    train_pos = pos_neg_reviews('pos', train_texts, train_labels)
    train_neg = pos_neg_reviews('neg', train_texts, train_labels)

    train_texts_stop_punc = preprocess(train_texts, stop_words=True, punctuation=True)
    train = tokenization_gram(train_texts_stop_punc)
    flag = True
    train_pos = preprocess(train_pos, stop_words=flag, punctuation=flag)
    train_neg = preprocess(train_neg, stop_words=flag, punctuation=flag)

    train_pos = tokenization_gram(train_pos)
    train_neg = tokenization_gram(train_neg)

    dict_all = make_vocab_ngram(train, mode=mode, gram_range=gram_range)
    dict_pos = make_vocab_ngram(train_pos, mode=mode, gram_range=gram_range)
    dict_neg = make_vocab_ngram(train_neg, mode=mode, gram_range=gram_range)

    global train_pos_all, train_neg_all
    train_pos_all, train_neg_all = len(train_pos), len(train_neg)

    if mode == 'br':
        df = pd.DataFrame({'pos_cnt': pd.Series(dict_pos), 'neg_cnt': pd.Series(dict_neg)})
        df['pos_probs'] = (df['pos_cnt'].fillna(0) + alpha) / (train_pos_all + 2)
        df['neg_probs'] = (df['neg_cnt'].fillna(0) + alpha) / (train_neg_all + 2)
        df.fillna(1.0, inplace=True)
        df['NB_weight'] = np.log(df['pos_probs'] / df['neg_probs'])

    else:

        df = pd.DataFrame({'pos_cnt': pd.Series(dict_pos), 'neg_cnt': pd.Series(dict_neg)})
        all_pos_mul, all_neg_mul = np.sum(list(dict_pos.values())), np.sum(list(dict_neg.values()))

        df['pos_probs'] = (df['pos_cnt'].fillna(0) + alpha) / (all_pos_mul)
        df['neg_probs'] = (df['neg_cnt'].fillna(0) + alpha) / (all_neg_mul)
        df.fillna(1.0, inplace=True)
        df['NB_weight'] = np.log(df['pos_probs'] / df['neg_probs'])

    vocab = set(dict_all.keys())
    vocab_pos = set(dict_pos.keys())
    vocab_neg = set(dict_neg.keys())

    params = [df, vocab, vocab_pos, vocab_neg]
    return params

def classify(texts, params):
    df=params[0]
    vocab=params[1]
    vocab_pos=params[2]
    vocab_neg=params[3]
    gram_range=(1, 4)

    tokens = tokenization_gram(texts, gram_range=gram_range)
    probs = []
    p_c_pos = np.log(train_pos_all / (train_pos_all + train_neg_all))
    p_c_neg = np.log(train_neg_all / (train_pos_all + train_neg_all))
    pos_probs = df['pos_probs']
    neg_probs = df['neg_probs']
    for line in tqdm_notebook(tokens):
        pos_prob = p_c_pos
        neg_prob = p_c_neg
        line = set(line)
        for word in line:  # for accelerating I change loop words in vocab -> loop  words in tokens w
            flag_pos = 0
            flag_neg = 0
            if word in vocab:
                if word in vocab_pos:
                    flag_pos = 1
                if word in vocab_neg:
                    flag_neg = 1

                p_prob, n_prob = pos_probs[word], neg_probs[word]
                pos_prob += np.log(flag_pos * p_prob + (1 - flag_pos) * (p_prob))
                neg_prob += np.log(flag_neg * n_prob + (1 - flag_neg) * (n_prob))
        if pos_prob > neg_prob:
            probs.append('pos')
        else:
            probs.append('neg')
    return probs