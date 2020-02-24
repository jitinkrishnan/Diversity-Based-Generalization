from preprocess import *
from wordvec_utils import *
import numpy as np
import nltk, random


def create_one_training_example(full_text_example, max_len, wv_dict):
    text = preprocess_1(full_text_example)

    words = nltk.word_tokenize(text)
    bag = []
    mywords = []
    count = 0
    for word in words:
        if count == max_len:
            break
        if word in wv_dict.vocab.keys():
            v = get_wordvector(word,wv_dict,300)
            if v is not None:
                count += 1
                bag.append(list(v))
                mywords.append(word)
    
    for i in range(max_len-count):
        bag.append(list(np.zeros(300)))

    return mywords, np.asarray(bag)

def inplace_shuffle(a,b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a,b

def create_data4lstm(train_category, test_category, wv_dict, Tx=75, Ty=1):

	# TRAIN
	f_bags_pos = open("raw_data/"+train_category+"/review_positive")
	f_bags_neg = open("raw_data/"+train_category+"/review_negative")

	pos = f_bags_pos.readlines()[:10]
	neg = f_bags_neg.readlines()[:10]
	bags = pos + neg

	f_bags_pos.close()
	f_bags_neg.close()

	min_num = min(len(pos), len(neg))

	bag_pos = []
	for text in pos[:min_num]:
		bag_pos.append(create_one_training_example(text, Tx, wv_dict)[1])

	bag_neg = []
	for text in neg[:min_num]:
		bag_neg.append(create_one_training_example(text, Tx, wv_dict)[1])
	
	pos_labels = []
	for i in range(len(bag_pos)):
		pos_labels.append([1,0])

	neg_labels = []
	for i in range(len(bag_neg)):
		neg_labels.append([0,1])

	X_train = bag_pos + bag_neg
	Y_train = pos_labels + neg_labels
	(X_train,Y_train) = inplace_shuffle(X_train,Y_train)

	Xoh = np.asarray(X_train)
	Yoh = np.asarray(Y_train)

	Yoh = np.reshape(Yoh, (Yoh.shape[0],1,2))

	# TEST

	f_bags_pos = open("raw_data/"+test_category+"/review_positive")
	f_bags_neg = open("raw_data/"+test_category+"/review_negative")

	pos = f_bags_pos.readlines()[:10]
	neg = f_bags_neg.readlines()[:10]
	bags = pos + neg

	f_bags_pos.close()
	f_bags_neg.close()

	min_num = min(len(pos), len(neg))

	bag_pos = []
	for text in pos[:min_num]:
		bag_pos.append(create_one_training_example(text, Tx, wv_dict)[1])

	bag_neg = []
	for text in neg[:min_num]:
		bag_neg.append(create_one_training_example(text, Tx, wv_dict)[1])
	
	pos_labels = []
	for i in range(len(bag_pos)):
		pos_labels.append([1,0])

	neg_labels = []
	for i in range(len(bag_neg)):
		neg_labels.append([0,1])

	X_test = bag_pos + bag_neg
	Y_test = pos_labels + neg_labels
	(X_test,Y_test) = inplace_shuffle(X_test,Y_test)

	Xoh_test = np.asarray(X_test)
	Yoh_test = np.asarray(Y_test)

	return Xoh, Yoh, Xoh_test, Yoh_test

def create_data4lstm_DA_oneclass(domain_A, wv_dict, Tx=75, Ty=1):

    # TRAIN
    f_bags_pos = open("raw_data/"+domain_A+"/review_unlabeled")

    pos = f_bags_pos.readlines()[:10]

    f_bags_pos.close()

    bag_pos = []
    for text in pos:
        bag_pos.append(create_one_training_example(text, Tx, wv_dict)[1])

    Xoh = np.asarray(bag_pos)

    return Xoh
