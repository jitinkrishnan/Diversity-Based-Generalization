## Diversity-Based Generalization for Neural Unsupervised Text Classification under Domain Shift

**Purpose of the model**: Train a classifier in one domain where plenty of data is available and generalize it to another domain with **zero** data.

### Paper/Cite
https://arxiv.org/pdf/2002.10937.pdf

### Why use this method? (See paper for detailed performance comparison)
- Unlike most state-of-the-art methods, no unlabeled target data is needed to train the model (which means no gradient reversal or manual pivot extractions). Out-of-the-box adaptable to any domain. 
- Computationally much cheaper than the state-of-the-art methods which uses unlabeled target data, with no trade-off in performance.

### Requirements
Python3.6, Keras, Tensorflow.
Or ```pip install -r requirements.txt``` to install necessary packages.

### Additional Requirements
Download [GoogleNews-vectors-negative300.bin](https://code.google.com/archive/p/word2vec/)

### Data
All datasets in **[raw_data](https://github.com/jitinkrishnan/Diversity-Based-Generalization/tree/master/raw_data)** folder. @user mentions anonymized for twitter data.

#### To add new dataset:
Place your positive/negative/unlabeled in the **[raw_data](https://github.com/jitinkrishnan/Diversity-Based-Generalization/tree/master/raw_data)** folder (no preprocessing needed) and name the files accordingly.

### Sample Runs of all models
#### BiLSTM
```python bilstm.py 'electronics' 'kitchen```

#### BiLSTM + Attention
```python bilstm_attention.py 'electronics' 'kitchen```

#### BiLSTM + MHA
```python bilstm_mha.py 'electronics' 'kitchen'```

#### BiLSTM + MHAD
```python bilstm_mhad.py 'electronics' 'kitchen'```

#### BiLSTM + MHAD + Tri-I
```python bilstm_mhad_tri_I.py 'electronics' 'kitchen'```

#### BiLSTM + MHAD + Tri-II
```python bilstm_mhad_tri_II.py 'electronics' 'kitchen'```


