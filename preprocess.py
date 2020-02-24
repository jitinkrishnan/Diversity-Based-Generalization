import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk import punkt
import numpy as np
import re, random
from nltk.chunk import RegexpParser
import nltk, scipy, emoji
from nltk.corpus import wordnet
import csv, sys, random, math, re, itertools
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize

tknzr = TweetTokenizer()


CONTRACTIONS = { 
" aint": " am not",
"ain't": "am not",
"aren't": "are not",
" arent": " are not",
"can't": "cannot",
" cant": " cannot",
"can't've": "cannot have",
" cant've": " cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
" couldnt": " could not",
"couldn't've": "could not have",
"didn't": "did not",
" didnt": " did not",
"doesn't": "does not",
" doesnt": " does not",
"don't": "do not",
" dont": " do not",
"hadn't": "had not",
" hadnt": " had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"isnt": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": " you will have",
"you're": "you are",
"you've": "you have"
}

def remove_adjacent_duplicates(word_list):
    curr = None
    new_word_list = []
    for i in range(len(word_list)):
        if curr is None:
            curr = word_list[i]
            new_word_list.append(curr)
            continue
        if word_list[i] != curr:
            curr = word_list[i]
            new_word_list.append(curr)
    return new_word_list

def remove_adjacent_duplicates_fromline(line):
    #word_list = nltk.word_tokenize(line.split()
    tknzr = TweetTokenizer()
    word_list = tknzr.tokenize(line)
    #new_word_list = [word for word in word_list if len(word) > 2]
    return ' '.join(remove_adjacent_duplicates(word_list))

def preprocess_1(sentence):

    if type(sentence) != str:
        return ""
    
    sentence = (sentence.encode('ascii', 'ignore')).decode("utf-8")
    
    # URLs
    sentence = re.sub(r'http\S+', ' <URL> ', sentence)
    
    # emoji
    for c in sentence:
        if c in emoji.UNICODE_EMOJI:
            sentence = re.sub(c, emoji.demojize(c), sentence)
    
    sentence = re.sub("([!]){1,}", " ! ", sentence)
    sentence = re.sub("([.]){1,}", " . ", sentence)
    sentence = re.sub("([?]){1,}", " ? ", sentence)
    sentence = re.sub("([;]){1,}", " ; ", sentence)
    sentence = re.sub("([:]){2,}", " : ", sentence)
    
    # numerical values
    #sentence = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " <NUMBER> ", sentence)
    
    # convert words such as "goood" to "good"
    sentence = ''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence))
    
    # symbols
    sentence = re.sub('&', " and ", sentence)
    
    # convert to lower case
    words = tknzr.tokenize(sentence)
    
    # expand contractions
    words = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    
    sentence =  " ".join(words)
    
    sentence = re.sub('[^ a-zA-Z0-9.!?:;<>_#@]', ' ', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    
    return remove_adjacent_duplicates_fromline(sentence)