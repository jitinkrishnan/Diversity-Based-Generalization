import keras
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import nltk, scipy, random
import pandas as pd
import sys, random, math, re, itertools
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk import word_tokenize
import operator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from numpy import linalg as LA
import math
from dataset_utils import *

######################## helper functions ########################
def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

######################## BILSTM ########################
def bilstm_model(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.4):

    X = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    A = Bidirectional(LSTM(units=n_a,dropout=drop))(X)
    A = Dense(10, activation = "relu",name='hidden')(A)
    out = Dense(out_dim, activation=softmax, name='classification')(A)
    model = Model(inputs=[X, s0, c0], outputs=out)
    return model

def train_bilstm(Xoh, Yoh, Tx, Ty, n_a=32, n_s=64, out_dim = 2, wv_dim=300, epochs=10, drop=0.4):

    model = bilstm_model(Tx, Ty, n_a, n_s, wv_dim, out_dim,drop=drop)

    #print(model.summary())

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'classification': 'categorical_crossentropy'},loss_weights={'classification': 1.0},optimizer=opt,metrics={'classification': 'accuracy'})
 
    s0 = np.zeros((len(Xoh), n_s))
    c0 = np.zeros((len(Xoh), n_s))
    outputs = list(Yoh.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    model.fit([Xoh, s0, c0], {'classification': outputs},batch_size=32,epochs=epochs,validation_split=0.15,shuffle=True, callbacks=callbacks)
    
    return model

def evaluate_bilstm(model, Xoh_test, Yoh_test, n_s=64):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred = []
    y_prob = []
    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        y_pred.append(np.argmax(prediction.squeeze()))
        if np.argmax(prediction.squeeze()) == 1:
            y_prob.append(np.max(prediction.squeeze()))
        else: 
            y_prob.append(1 - np.max(prediction.squeeze()))
    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))
    
    acc = accuracy_score(y_true, y_pred)
    
    return round(acc,4)

######################## BILSTM + ATTENTION ########################

def bilstm_attention_model(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.4):
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(softmax)
    dotor = Dot(axes = 1)
    post_activation_LSTM_cell = LSTM(n_s, return_state = True)

    X = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    attn_outputs = []
    a = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))(X)
    s_prev = repeator(s)
    concat = concatenator([a,s_prev])
    e = densor1(concat)
    e = Dropout(drop)(e)
    energies = densor2(e)
    alphas = activator(energies)

    context = dotor([alphas,a])

    s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])
    out_pre = Dense(n_s, activation = "relu")(s)
    output_layer = Dense(out_dim, activation=softmax, name='classification')(out_pre)

    word_attention = Lambda(lambda x: x[:, :,0,])(alphas)
    word_attention = Activation(None, name='word_attention')(word_attention)
    word_attention_copy = Lambda(lambda x: x[:, :,0,])(alphas)
    word_attention_copy = Activation(None, name='word_attention_copy')(word_attention_copy)

    model = Model(inputs=[X, s0, c0], outputs=[output_layer,word_attention,word_attention_copy])
    return model

def train_bilstm_attention(Xoh, Yoh, Tx, Ty, n_a=32, n_s=64, out_dim = 2, wv_dim=300,epochs=10,drop=0.4):

    model = bilstm_attention_model(Tx, Ty, n_a, n_s, wv_dim, out_dim, drop=drop)

    #print(model.summary())

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'classification': 'categorical_crossentropy'},loss_weights={'classification': 1.0},optimizer=opt,metrics={'classification': 'accuracy'})
 
    s0 = np.zeros((len(Xoh), n_s))
    c0 = np.zeros((len(Xoh), n_s))
    outputs = list(Yoh.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    model.fit([Xoh, s0, c0], {'classification': outputs},batch_size=32,epochs=epochs,validation_split=0.15, shuffle=True, callbacks=callbacks)
    
    return model

def evaluate_bilstm_attention(model, Xoh_test, Yoh_test, n_s=64):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred = []
    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        y_pred.append(np.argmax(prediction[0].squeeze()))

    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))

    acc = accuracy_score(y_true, y_pred)

    return round(acc,4)

######################## BILSTM + Multi-Head ATTENTION ########################

def bilstm_mha_model(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.4, heads=5):

    X = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    attn_outputs = []
    a = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))(X)

    attn_weights = []

    for h in range(heads):
        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(10, activation = "tanh")
        densor2 = Dense(1, activation = "relu")
        activator = Activation(softmax)
        dotor = Dot(axes = 1)
        post_activation_LSTM_cell = LSTM(n_s, return_state = True)

        s_prev = repeator(s)
        concat = concatenator([a,s_prev])
        e = densor1(concat)
        e = Dropout(drop)(e)
        energies = densor2(e)
        alphas = activator(energies)
        context = dotor([alphas,a])
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])
        attn_weights.append(alphas)

    out_pre = Dense(n_s, activation = "relu")(s)
    output_layer = Dense(out_dim, activation=softmax, name='classification')(out_pre)

    outputs2 = []

    for index in range(len(attn_weights)):
        attn_w = attn_weights[index]
        attn_w = Lambda(lambda x: x[:, :,0,])(attn_w)
        attn_w = Activation(None, name='word_attention_'+str(index))(attn_w)
        outputs2.append(attn_w)

    outputs=[output_layer]
    outputs.extend(outputs2)

    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model

def train_bilstm_mha(Xoh, Yoh, Tx, Ty, n_a=32, n_s=64, out_dim = 2, wv_dim=300,epochs=10,drop=0.4,heads=5):

    model = bilstm_mha_model(Tx, Ty, n_a, n_s, wv_dim, out_dim, drop=drop,heads=heads)

    #print(model.summary())

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'classification': 'categorical_crossentropy'},loss_weights={'classification': 1.0},optimizer=opt,metrics={'classification': 'accuracy'})
 
    s0 = np.zeros((len(Xoh), n_s))
    c0 = np.zeros((len(Xoh), n_s))
    outputs = list(Yoh.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    model.fit([Xoh, s0, c0], {'classification': outputs},batch_size=32,epochs=epochs,validation_split=0.15, shuffle=True, callbacks=callbacks)
    
    return model

def evaluate_bilstm_mha(model, Xoh_test, Yoh_test, n_s=64):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred = []
    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        y_pred.append(np.argmax(prediction[0].squeeze()))

    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))

    acc = accuracy_score(y_true, y_pred)

    return round(acc,4)

######################## BILSTM + Multi-Head ATTENTION + DIVERSITY########################
def bilstm_mhad_model(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.4, heads=5):

    X = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    attn_outputs = []
    a = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))(X)

    attn_weights = []

    for h in range(heads):
        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(10, activation = "tanh")
        densor2 = Dense(1, activation = "relu")
        activator = Activation(softmax)
        dotor = Dot(axes = 1)
        post_activation_LSTM_cell = LSTM(n_s, return_state = True)

        s_prev = repeator(s)
        concat = concatenator([a,s_prev])
        e = densor1(concat)
        e = Dropout(drop)(e)
        energies = densor2(e)
        alphas = activator(energies)
        context = dotor([alphas,a])
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])
        attn_weights.append(alphas)

    out_pre = Dense(n_s, activation = "relu")(s)
    output_layer = Dense(out_dim, activation=softmax, name='classification')(out_pre)

    outputs2 = []

    for index in range(len(attn_weights)):
        attn_w = attn_weights[index]
        attn_w = Lambda(lambda x: x[:, :,0,])(attn_w)
        attn_w = Activation(None, name='word_attention_'+str(index))(attn_w)
        outputs2.append(attn_w)

    outputs=[output_layer]
    outputs.extend(outputs2)

    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model

def custom_loss_mhad(layers, diversity_weight=0.01):
    def loss(y_true,y_pred):
        ce_loss = K.categorical_crossentropy(y_true,y_pred)
        ortho_loss = 0
        count = 0
        for i in range(len(layers)-2):
            for j in range(i+1,len(layers)-1):
                x = layers[i].output
                y = layers[j].output
                ortho_loss += K.sum(K.square(K.dot(x,K.transpose(y))))
                count+=1
        ortho_loss = ortho_loss/count
        return ce_loss + diversity_weight*ortho_loss
    return loss

def train_bilstm_mhad(Xoh, Yoh, Tx, Ty, n_a=32, n_s=64, out_dim = 2, wv_dim=300,epochs=10,drop=0.4,heads=5,diversity_weight=0.01):

    model = bilstm_mhad_model(Tx, Ty, n_a, n_s, wv_dim, out_dim, drop=drop,heads=heads)

    #print(model.summary())

    attn_layers = []
    for i in range(heads):
        lyrname = 'word_attention_'+str(i)
        attn_layers.append(model.get_layer(lyrname))

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss={'classification': custom_loss_mhad(attn_layers,diversity_weight=diversity_weight)},loss_weights={'classification': 1.0},optimizer=opt,metrics={'classification': 'accuracy'})
 
    s0 = np.zeros((len(Xoh), n_s))
    c0 = np.zeros((len(Xoh), n_s))
    outputs = list(Yoh.swapaxes(0,1))[0]

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    model.fit([Xoh, s0, c0], {'classification': outputs},batch_size=32,epochs=epochs,validation_split=0.15, shuffle=True, callbacks=callbacks)
    
    return model

def evaluate_bilstm_mhad(model, Xoh_test, Yoh_test, n_s=64):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred = []
    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        y_pred.append(np.argmax(prediction[0].squeeze()))

    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))

    acc = accuracy_score(y_true, y_pred)

    return round(acc,4)

######################## BILSTM + Multi-Head ATTENTION + DIVERSITY + ONE STEP TRI-TRAINING########################
def bilstm_mhad_onstri_model(Tx, Ty, n_a, n_s, vocab_size, out_dim, drop=0.4, heads=5):

    X = Input(shape=(Tx, vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    a = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))(X)

    attn_weights = []
    outputs = []
    attn_outputs = []
    s = s0
    c = c0

    for h in range(heads):
        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(10, activation = "tanh")
        densor2 = Dense(1, activation = "relu")
        activator = Activation(softmax)
        dotor = Dot(axes = 1)
        post_activation_LSTM_cell = LSTM(n_s, return_state = True)

        s_prev = repeator(s)
        concat = concatenator([a,s_prev])
        e = densor1(concat)
        e = Dropout(drop)(e)
        energies = densor2(e)
        alphas = activator(energies)
        context = dotor([alphas,a])
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])
        attn_weights.append(alphas)

    out_pre = Dense(n_s, activation = "relu")(s)
    classification_1 = Dense(out_dim, activation=softmax, name='classification_1')(out_pre)
    attn_w = Lambda(lambda x: x[:, :,0,])(attn_weights[-1])
    word_attention_1 = Activation(None, name='word_attention_1')(attn_w)
    
    for index in range(len(attn_weights)):
        attn_w = attn_weights[index]
        attn_w = Lambda(lambda x: x[:, :,0,])(attn_w)
        attn_w = Activation(None, name='word_attention_1_'+str(index))(attn_w)
        attn_outputs.append(attn_w)

    outputs.extend(attn_outputs)

    ######### m2 ##########
    a = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))(X)
    attn_weights_2 = []
    outputs_2 = []
    attn_outputs_2 = []
    s = s0
    c = c0

    for h in range(heads):
        repeator_2 = RepeatVector(Tx)
        concatenator_2 = Concatenate(axis=-1)
        densor1_2 = Dense(10, activation = "tanh")
        densor2_2 = Dense(1, activation = "relu")
        activator_2 = Activation(softmax)
        dotor_2 = Dot(axes = 1)
        post_activation_LSTM_cell_2 = LSTM(n_s, return_state = True)

        s_prev = repeator_2(s)
        concat = concatenator_2([a,s_prev])
        e = densor1_2(concat)
        e = Dropout(drop)(e)
        energies = densor2_2(e)
        alphas = activator_2(energies)
        context = dotor_2([alphas,a])
        s, _, c = post_activation_LSTM_cell_2(context, initial_state = [s,c])
        attn_weights_2.append(alphas)

    out_pre = Dense(n_s, activation = "relu")(s)
    classification_2 = Dense(out_dim, activation=softmax, name='classification_2')(out_pre)
    attn_w = Lambda(lambda x: x[:, :,0,])(attn_weights_2[-1])
    word_attention_2 = Activation(None, name='word_attention_2')(attn_w)
    
    for index in range(len(attn_weights_2)):
        attn_w = attn_weights_2[index]
        attn_w = Lambda(lambda x: x[:, :,0,])(attn_w)
        attn_w = Activation(None, name='word_attention_2_'+str(index))(attn_w)
        attn_outputs_2.append(attn_w)

    outputs.extend(attn_outputs_2)

    ######### m3 ##########
    a = Bidirectional(LSTM(units=n_a, return_sequences=True, dropout=drop))(X)
    attn_weights_3 = []
    outputs_3 = []
    attn_outputs_3 = []
    s = s0
    c = c0

    for h in range(heads):
        repeator_3 = RepeatVector(Tx)
        concatenator_3 = Concatenate(axis=-1)
        densor1_3 = Dense(10, activation = "tanh")
        densor2_3 = Dense(1, activation = "relu")
        activator_3 = Activation(softmax)
        dotor_3 = Dot(axes = 1)
        post_activation_LSTM_cell_3 = LSTM(n_s, return_state = True)

        s_prev = repeator_3(s)
        concat = concatenator_3([a,s_prev])
        e = densor1_3(concat)
        e = Dropout(drop)(e)
        energies = densor2_3(e)
        alphas = activator_3(energies)
        context = dotor_3([alphas,a])
        s, _, c = post_activation_LSTM_cell_3(context, initial_state = [s,c])
        attn_weights_3.append(alphas)

    out_pre = Dense(n_s, activation = "relu")(s)
    classification_3 = Dense(out_dim, activation=softmax, name='classification_3')(out_pre)
    attn_w = Lambda(lambda x: x[:, :,0,])(attn_weights_3[-1])
    word_attention_3 = Activation(None, name='word_attention_3')(attn_w)

    for index in range(len(attn_weights_3)):
        attn_w = attn_weights_3[index]
        attn_w = Lambda(lambda x: x[:, :,0,])(attn_w)
        attn_w = Activation(None, name='word_attention_3_'+str(index))(attn_w)
        attn_outputs_3.append(attn_w)

    outputs.extend(attn_outputs_3)

    full_outputs = [classification_1,classification_2,classification_3,word_attention_1,word_attention_2,word_attention_3]
    full_outputs.extend(outputs)

    model = Model(inputs=[X, s0, c0], outputs=full_outputs)
    return model

def custom_loss_mhad_onstri(layers1, layers2, ortho_loss_weight=0.01, diversity_weight=0.01):
    def loss(y_true,y_pred):
        ce_loss = K.categorical_crossentropy(y_true,y_pred)
        ortho_loss = 0.0
        diversity_loss = 0.0
        ortho_count = 0
        diversity_count = 0
        for i in range(len(layers1)):
            if i < len(layers1)-2:
                for j in range(i+1,len(layers1)-1):
                    x = layers1[i].output
                    y = layers1[j].output
                    diversity_loss += K.sum(K.square(K.dot(x,K.transpose(y))))
                    diversity_count+=1
            for j in range(len(layers2)):
                x = layers1[i].output
                y = layers2[j].output
                ortho_loss += K.sum(K.square(K.dot(x,K.transpose(y))))
                ortho_count+=1
        ortho_loss = ortho_loss/ortho_count
        diversity_loss = diversity_loss/diversity_count
        return ce_loss + ortho_loss_weight*ortho_loss + diversity_weight*diversity_loss
    return loss

def train_bilstm_mhad_ostri(Xoh, Yoh, Tx, Ty, n_a=32, n_s=64, out_dim = 2, wv_dim=300,epochs=10, zeta=0.7,drop=0.4,heads=5, ortho_loss_weight=0.01, diversity_weight=0.01):

    model = bilstm_mhad_onstri_model(Tx, Ty, n_a, n_s, wv_dim, out_dim, heads=heads)

    attn_layers_1 = []
    for i in range(heads):
        lyrname = 'word_attention_1_'+str(i)
        attn_layers_1.append(model.get_layer(lyrname))

    attn_layers_2 = []
    for i in range(heads):
        lyrname = 'word_attention_2_'+str(i)
        attn_layers_2.append(model.get_layer(lyrname))

    attn_layers_3 = []
    for i in range(heads):
        lyrname = 'word_attention_3_'+str(i)
        attn_layers_3.append(model.get_layer(lyrname))

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    s0 = np.zeros((len(Xoh), n_s))
    c0 = np.zeros((len(Xoh), n_s))
    outputs = list(Yoh.swapaxes(0,1))[0]
    model.compile(loss={'classification_1': custom_loss_mhad_onstri(attn_layers_1, attn_layers_2,ortho_loss_weight=ortho_loss_weight,diversity_weight=diversity_weight),'classification_2': custom_loss_mhad_onstri(attn_layers_2, attn_layers_1,ortho_loss_weight=ortho_loss_weight,diversity_weight=diversity_weight), 'classification_3': custom_loss_mhad(attn_layers_3,diversity_weight=diversity_weight)},loss_weights={'classification_1': 1.0,'classification_2': 1.0, 'classification_3': 1.0},optimizer=opt,metrics={'classification_1': 'accuracy','classification_2': 'accuracy', 'classification_3': 'accuracy'})
    model.fit([Xoh, s0, c0], {'classification_1': outputs,'classification_2': outputs,'classification_3': outputs},batch_size=32,epochs=epochs,validation_split=0.15,shuffle=True, callbacks=callbacks)

    return model

def evaluate_mhad_ostri(model, Xoh_test, Yoh_test, n_s=64):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred = []

    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        p1 = np.argmax(prediction[0].squeeze())
        p2 = np.argmax(prediction[1].squeeze())
        p3 = np.argmax(prediction[2].squeeze())
        p_sum = p1+p2+p3
        ans = 0
        if p_sum > 1:
            ans = 1
        y_pred.append(ans)

    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))

    acc = accuracy_score(y_true, y_pred)

    return round(acc,4)

######################## BILSTM + Multi-Head ATTENTION + DIVERSITY + FULL TRI-TRAINING########################
def construct_pseudo_labelled_data(model, Xoh_unl, zeta, Xoh_len, i, j, n_s=64):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))

    Xoh_pseudo_pos = []
    Yoh_pseudo_pos = []
    Xoh_pseudo_neg = []
    Yoh_pseudo_neg = []
    count = 0
    for sample in Xoh_unl:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        if np.argmax(prediction[i].squeeze()) == np.argmax(prediction[j].squeeze()):
            p1 = np.max(prediction[i].squeeze())
            p2 = np.max(prediction[j].squeeze())
            if p1 > zeta or p2 > zeta:
                count += 1
                #Xoh = np.append(Xoh,np.array([sample]), axis=0)
                lab = np.argmax(prediction[i].squeeze())
                if lab == 0:
                    Xoh_pseudo_pos.append(sample)
                    ans = np.array([0,0])
                    ans[lab] = 1
                    ans = np.expand_dims(ans, axis=0)
                    Yoh_pseudo_pos.append(ans)
                else:
                    Xoh_pseudo_neg.append(sample)
                    ans = np.array([0,0])
                    ans[lab] = 1
                    ans = np.expand_dims(ans, axis=0)
                    Yoh_pseudo_neg.append(ans)
    if len(Xoh_pseudo_pos) > 0 and len(Xoh_pseudo_neg) > 0:
        (Xoh_pseudo_pos,Yoh_pseudo_pos) = inplace_shuffle(Xoh_pseudo_pos,Yoh_pseudo_pos)
        (Xoh_pseudo_neg,Yoh_pseudo_neg) = inplace_shuffle(Xoh_pseudo_neg,Yoh_pseudo_neg)

    min_len = min(len(Yoh_pseudo_pos), len(Yoh_pseudo_neg), Xoh_len//2)

    Xoh_pseudo = Xoh_pseudo_pos[:min_len] + Xoh_pseudo_neg[:min_len]
    Yoh_pseudo = Yoh_pseudo_pos[:min_len] + Yoh_pseudo_neg[:min_len]

    return np.asarray(Xoh_pseudo), np.asarray(Yoh_pseudo)

def check_mttri_convergence(model, Xoh_unl, n_s=64):

    max_limit = len(Xoh_unl)

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))

    count = 0

    np.random.shuffle(Xoh_unl)
    for sample in Xoh_unl[:max_limit]:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        if np.argmax(prediction[0].squeeze()) == np.argmax(prediction[1].squeeze()) == np.argmax(prediction[2].squeeze()):
            count += 1

    if count / len(Xoh_unl[:max_limit]) > 0.85:
        return True

    return False

def train_bilstm_mhad_fulltri(Xoh, Yoh, Xoh_unl, Tx, Ty, n_a=32, n_s=64, out_dim = 2, wv_dim=300,epochs=10, zeta=0.7,drop=0.4,heads=5,ortho_loss_weight=0.01, diversity_weight=0.01):

    Xoh_unl = Xoh_unl[:len(Xoh)]

    model = bilstm_mhad_onstri_model(Tx, Ty, n_a, n_s, wv_dim, out_dim, heads=heads)

    attn_layers_1 = []
    for i in range(heads):
        lyrname = 'word_attention_1_'+str(i)
        attn_layers_1.append(model.get_layer(lyrname))

    attn_layers_2 = []
    for i in range(heads):
        lyrname = 'word_attention_2_'+str(i)
        attn_layers_2.append(model.get_layer(lyrname))

    attn_layers_3 = []
    for i in range(heads):
        lyrname = 'word_attention_3_'+str(i)
        attn_layers_3.append(model.get_layer(lyrname))

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    #base training m1 and m2
    s0 = np.zeros((len(Xoh), n_s))
    c0 = np.zeros((len(Xoh), n_s))
    outputs = list(Yoh.swapaxes(0,1))[0]
    model.compile(loss={'classification_1': custom_loss_mhad_onstri(attn_layers_1, attn_layers_2,ortho_loss_weight=ortho_loss_weight,diversity_weight=diversity_weight),'classification_2': custom_loss_mhad_onstri(attn_layers_2, attn_layers_1,ortho_loss_weight=ortho_loss_weight,diversity_weight=diversity_weight), 'classification_3': custom_loss_mhad(attn_layers_3,diversity_weight=diversity_weight)},loss_weights={'classification_1': 1.0,'classification_2': 1.0, 'classification_3': 0.0},optimizer=opt,metrics={'classification_1': 'accuracy','classification_2': 'accuracy'})
    model.fit([Xoh, s0, c0], {'classification_1': outputs,'classification_2': outputs,'classification_3': outputs},batch_size=32,epochs=epochs,validation_split=0.15,shuffle=True, callbacks=callbacks,verbose=0)

    #m3
    Xoh_pseudo, Yoh_pseudo = construct_pseudo_labelled_data(model, Xoh_unl, zeta, len(Xoh), 0, 1)

    if len(Xoh_pseudo) == 0:
        return model

    s0 = np.zeros((len(Xoh_pseudo), n_s))
    c0 = np.zeros((len(Xoh_pseudo), n_s))
    outputs = list(Yoh_pseudo.swapaxes(0,1))[0]

    model.compile(loss={'classification_1': 'categorical_crossentropy','classification_2': 'categorical_crossentropy', 'classification_3': custom_loss_mhad(attn_layers_3,diversity_weight=diversity_weight)},loss_weights={'classification_1': 0.0,'classification_2': 0.0, 'classification_3': 1.0},optimizer=opt,metrics={'classification_3': 'accuracy'})
    model.fit([Xoh_pseudo, s0, c0], {'classification_1': outputs,'classification_2': outputs,'classification_3': outputs},batch_size=32,epochs=epochs,validation_split=0.15,shuffle=True, callbacks=callbacks,verbose=0)
    
    del Xoh_pseudo
    del Yoh_pseudo

    convergence_count = 0

    while (not check_mttri_convergence(model, Xoh_unl)) and convergence_count < 5:

        #m1
        Xoh_pseudo, Yoh_pseudo = construct_pseudo_labelled_data(model, Xoh_unl, zeta, len(Xoh), 1, 2)

        Xoh_new = np.concatenate((Xoh, np.array(Xoh_pseudo)))
        Yoh_new = np.concatenate((Yoh, np.array(Yoh_pseudo)))

        s0 = np.zeros((len(Xoh_new), n_s))
        c0 = np.zeros((len(Xoh_new), n_s))
        outputs = list(Yoh_new.swapaxes(0,1))[0]

        model.compile(loss={'classification_1': custom_loss_mhad_onstri(attn_layers_1, attn_layers_2,ortho_loss_weight=ortho_loss_weight,diversity_weight=diversity_weight),'classification_2': custom_loss_mhad_onstri(attn_layers_2, attn_layers_1,ortho_loss_weight=ortho_loss_weight,diversity_weight=diversity_weight), 'classification_3': 'categorical_crossentropy'},loss_weights={'classification_1': 1.0,'classification_2': 0.5, 'classification_3': 0.0},optimizer=opt,metrics={'classification_1': 'accuracy'})
        model.fit([Xoh_new, s0, c0], {'classification_1': outputs,'classification_2': outputs,'classification_3': outputs},batch_size=32,epochs=epochs,validation_split=0.15,shuffle=True, callbacks=callbacks,verbose=0)

        del Xoh_pseudo
        del Yoh_pseudo
        del Xoh_new
        del Yoh_new

        #m2
        Xoh_pseudo, Yoh_pseudo = construct_pseudo_labelled_data(model, Xoh_unl, zeta, len(Xoh), 0, 2)
        Xoh_new = np.concatenate((Xoh, np.array(Xoh_pseudo)))
        Yoh_new = np.concatenate((Yoh, np.array(Yoh_pseudo)))

        s0 = np.zeros((len(Xoh_new), n_s))
        c0 = np.zeros((len(Xoh_new), n_s))
        outputs = list(Yoh_new.swapaxes(0,1))[0]

        model.compile(loss={'classification_1': custom_loss_mhad_onstri(attn_layers_1, attn_layers_2,ortho_loss_weight=ortho_loss_weight,diversity_weight=diversity_weight),'classification_2': custom_loss_mhad_onstri(attn_layers_2, attn_layers_1,ortho_loss_weight=ortho_loss_weight,diversity_weight=diversity_weight), 'classification_3': 'categorical_crossentropy'},loss_weights={'classification_1': 0.5,'classification_2': 1.0, 'classification_3': 0.0},optimizer=opt,metrics={'classification_2': 'accuracy'})
        model.fit([Xoh_new, s0, c0], {'classification_1': outputs,'classification_2': outputs,'classification_3': outputs},batch_size=32,epochs=epochs,validation_split=0.15,shuffle=True, callbacks=callbacks,verbose=0)

        del Xoh_pseudo
        del Yoh_pseudo
        del Xoh_new
        del Yoh_new
        
        #m3
        Xoh_pseudo, Yoh_pseudo = construct_pseudo_labelled_data(model, Xoh_unl, zeta, len(Xoh), 0, 1)

        s0 = np.zeros((len(Xoh_pseudo), n_s))
        c0 = np.zeros((len(Xoh_pseudo), n_s))
        outputs = list(Yoh_pseudo.swapaxes(0,1))[0]

        model.compile(loss={'classification_1': 'categorical_crossentropy','classification_2': 'categorical_crossentropy', 'classification_3': custom_loss_mhad(attn_layers_3,diversity_weight=diversity_weight)},loss_weights={'classification_1': 0.0,'classification_2': 0.0, 'classification_3': 1.0},optimizer=opt,metrics={'classification_3': 'accuracy'})
        model.fit([Xoh_pseudo, s0, c0], {'classification_1': outputs,'classification_2': outputs,'classification_3': outputs},batch_size=32,epochs=epochs,validation_split=0.15,shuffle=True, callbacks=callbacks,verbose=0)
        
        del Xoh_pseudo
        del Yoh_pseudo

        convergence_count += 1

    return model

def evaluate_mhad_fulltri(model, Xoh_test, Yoh_test, n_s=64):

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    y_pred = []

    for sample in Xoh_test:
        prediction = model.predict([np.expand_dims(sample, axis=0), s0, c0])
        p1 = np.argmax(prediction[0].squeeze())
        p2 = np.argmax(prediction[1].squeeze())
        p3 = np.argmax(prediction[2].squeeze())
        p_sum = p1+p2+p3
        ans = 0
        if p_sum > 1:
            ans = 1
        y_pred.append(ans)

    y_true = []
    for label in Yoh_test:
        y_true.append(np.argmax(label))

    acc = accuracy_score(y_true, y_pred)

    return round(acc,4)