from wordvec_utils import *
from models import *
from dataset_utils import *
import sys

word2vec_file = 'GoogleNews-vectors-negative300.bin'
wv_dict = wordvec_dict(word2vec_file)

Tx = 200
Ty = 1
epochs = 40

def bilstm_attention_classifier(domainA, domainB):

	Xoh_domainA, Yoh_domainA, Xoh_domainB, Yoh_domainB = create_data4lstm(domainA, domainB, wv_dict, Tx, Ty)
	modelx = train_bilstm_attention(Xoh_domainA, Yoh_domainA, Tx, Ty, epochs=epochs)
	acc= evaluate_bilstm_attention(modelx, Xoh_domainB, Yoh_domainB)
	print(acc)

bilstm_attention_classifier(sys.argv[1], sys.argv[2])

