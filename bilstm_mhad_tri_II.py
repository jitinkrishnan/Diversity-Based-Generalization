from wordvec_utils import *
from models import *
from dataset_utils import *
import sys

word2vec_file = 'GoogleNews-vectors-negative300.bin'
wv_dict = wordvec_dict(word2vec_file)

Tx = 200
Ty = 1
epochs = 40

def bilstm_mhad_fulltri_classifier(domainA, domainB,zeta=0.7,drop=0.4,heads=5,ortho_loss_weight=0.01,diversity_weight=0.01):

	Xoh_domainA, Yoh_domainA, Xoh_domainB, Yoh_domainB = create_data4lstm(domainA, domainB, wv_dict, Tx, Ty)
	domainA_unlabelled = create_data4lstm_DA_oneclass(domainA, wv_dict, Tx, Ty)

	modelx = train_bilstm_mhad_fulltri(Xoh_domainA, Yoh_domainA, domainA_unlabelled, Tx, Ty, epochs=epochs,zeta=zeta,drop=drop,heads=heads,ortho_loss_weight=ortho_loss_weight, diversity_weight=diversity_weight)
	acc = evaluate_mhad_fulltri(modelx, Xoh_domainB, Yoh_domainB)
	print(acc)

bilstm_mhad_fulltri_classifier(sys.argv[1], sys.argv[2])

