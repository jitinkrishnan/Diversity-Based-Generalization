import numpy as np
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from gensim.models.keyedvectors import KeyedVectors

snowball_stemmer = SnowballStemmer('english')
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer()

def wordvec_dict(bin_file):
	model = KeyedVectors.load_word2vec_format(bin_file, binary=True)
	return model.wv

def lenOfdict(wv_dict):
    return len(wv_dict.vocab.keys())


def get_wordvector(word, model_wv,dim):
	vocab_keys = model_wv.vocab.keys()

	if word in vocab_keys:
		return model_wv[word]
	elif word.lower() in vocab_keys:
		return model_wv[word.lower()]
	elif word.upper() in vocab_keys:
		return model_wv[word.upper()]
	elif snowball_stemmer.stem(word) in vocab_keys:
		return model_wv[snowball_stemmer.stem(word)]
	elif wordnet_lemmatizer.lemmatize(word) in vocab_keys:
		return model_wv[wordnet_lemmatizer.lemmatize(word)]
	return None