
from wordvec_utils import *
from models import *
from dataset_utils import *
import sys

word2vec_file = 'GoogleNews-vectors-negative300.bin'
wv_dict = wordvec_dict(word2vec_file)

Tx = 200
Ty = 1
epochs = 40

def bilstm_mhad_classiifer(domainA, domainB, drop=0.4, heads=5, diversity_weight=0.01):

	Xoh_domainA, Yoh_domainA, Xoh_domainB, Yoh_domainB = create_data4lstm(domainA, domainB, wv_dict, Tx, Ty)

	modelx = train_bilstm_mhad(Xoh_domainA, Yoh_domainA, Tx, Ty, epochs=epochs,drop=drop,heads=heads,diversity_weight=diversity_weight)
	acc = evaluate_bilstm_mhad(modelx, Xoh_domainB, Yoh_domainB)
	print(acc)

bilstm_mhad_classiifer(sys.argv[1], sys.argv[2])



