## Diversity-Based Generalization for Unsupervised Text Classification under Domain Shift

**Purpose of the model**: Train a classifier in one domain where plenty of data is available and generalize it to another domain that has **no** (labeled or unlabeled) data.

### Paper/Cite
https://arxiv.org/abs/2002.10937 (To appear at [ECML-PKDD 2020](https://ecmlpkdd2020.net))
```
@article{krishnanDiversity,
  title={Diversity-Based Generalization for Unsupervised Text Classification under Domain Shift},
  author={Krishnan, Jitin and Purohit, Hemant and Rangwala, Huzefa},
  journal={In Proceedings of the 19th European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year={2020}
}
```

### Why use this method? (See paper for detailed performance comparison)
- Unlike the existing state-of-the-art unsupervised methods, no **unlabeled** target data is needed to train the model. Our model is out-of-the-box adaptable to any domain. 
- Computationally much cheaper (days to a few hours on a cpu) as it does not use unlabeled target data (which means no gradient reversal or manual pivot extractions), with no trade-off in performance.

#### Results on [Blitzer Dataset](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/)

| Task  | Accuracy in %  |
 :-: |  :-:
| books-dvd           | 87.46 |
| books-electronics   | 86.08 |
| books-kitchen       | 87.68 |
| kitchen-books       | 84.23 |
| kitchen-dvd         | 83.34 |
| kitchen-electronics | 89.22 |
| electronics-books   | 84.33 |
| electronics-kitchen | 91.05 |
| electronics-dvd     | 82.81 |
| dvd-books           | 88.74 |
| dvd-electronics     | 86.21 |
| dvd-kitchen         | 87.37 |
| **Average**		          | **86.54** |

#### Results on the newly labeled [Crisis Dataset](https://github.com/jitinkrishnan/Diversity-Based-Generalization/tree/master/raw_data)

| Task  | Accuracy in %  |
 :-: |  :-:
| Harvey-Florence     | 78.11 |
| Harvey-Irma         | 64.38 |

#### Results on very divergent datasets such as [Yelp](https://www.yelp.com/dataset/challenge) and [IMDb](https://ai.stanford.edu/~amaas/data/sentiment/), in addition to Amazon Reviews. (3 randomly selected combinations).

| Task  | Accuracy in %  |
 :-: |  :-:
| Electronics-Yelp  | 89.15 |
| Kitchen-IMDb      | 78.33 |
| Yelp-IMDb         | 77.28 |

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

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
