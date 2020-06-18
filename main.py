# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:14:10 2020

@author: Selly
"""
# from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from scipy.stats import spearmanr
from sklearn.utils import resample
from sklearn.feature_selection import SelectPercentile#, SelectKBest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

import keras.backend as K
from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD
# from keras.utils import to_categorical

raw = pd.read_csv('interviewData.csv')  # , index_col='ecid')
raw = raw.sort_values(by=['wafername_WS1', 'touchdownseq_WS1'])
raw = raw.reset_index(drop=True)

sec_col = raw.filter(regex='seconds', axis=1).columns
raw[sec_col] = raw[sec_col]/60/60/24
# %% die-wafer mapping
wafer_group = raw.groupby(by=['wafername_WS1'])
wafer_names = []
max_diex, max_diey = 0, 0
for name, group in wafer_group:
    wafer_names.append(name)
    min_x = group['diex_WS1'].min()
    raw.loc[group.index, 'diex_WS1'] -= min_x
    group.loc[:, 'diex_WS1'] = group['diex_WS1']-min_x

    max_x = group['diex_WS1'].max()
    if max_x > max_diex:
        max_diex = max_x
    max_y = group['diey_WS1'].max()
    if max_y > max_diey:
        max_diey = max_y

# graph
wafers = np.zeros((len(wafer_names), max_diex+1, max_diey+1))
for i, (name, group) in enumerate(wafer_group):
    for ind, row in group.iterrows():
        wafers[i, row['diex_WS1'], row['diey_WS1']] = ind

# %% pre-process functions
# normalization
def z_norm(data, axis=0):
    mu, sigma = data.mean(axis=axis).values, data.std(axis=axis).values
    if axis == 1:
        mu, sigma = map(np.transpose, ([mu], [sigma]))
    return (data - mu) / sigma
def minmax_norm(data):
    l, u = data.min().min(), data.max().max()
    return (data - l) / (u-l)

# group and split columns
def split(df):
    y = df['hardbin_FT1'].apply(lambda x: x != 1)
    measure = df.filter(regex=r'^[A-Z]_[0-9]+$', axis=1)
    cols_pre = measure.columns.map(lambda x: x.split('_')[0])
    measure.columns = pd.MultiIndex.from_tuples(zip(cols_pre, measure.columns))
    for c in measure.columns.get_level_values(0).unique():
        measure[c] = minmax_norm(measure[c]).values
    # measure = z_norm(measure)
    meta = df.filter(regex=r'(die|site|touchdown)\w*_(WS|FT)', axis=1)
    meta.columns = pd.MultiIndex.from_product([['meta'], meta.columns])
    meta = z_norm(meta)
    return y, (measure, meta)
def split_x(x):
    return (x.filter(regex='[A-Z]_', axis=1), x[['meta']])

# over-sampling
def simple_sample(x, y, ratio=0.5):
    sampled = resample(x[y], n_samples=int(sum(~y)*ratio))

    x = pd.concat([x, pd.DataFrame(sampled)])
    y = y.append(pd.Series([True, ]*len(sampled)))
    return x, y
def SMOTE_sample(x, y, ratio=0.5):
    sm = SMOTE(ratio, k_neighbors=2)
    return sm.fit_resample(x, y)

# feature
def feat_reduction(x, THRE=0.8):
    corr = x.corr()
    high_corr_ind = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if abs(corr.iloc[i,j]) >= THRE:
                if high_corr_ind[j]:
                    high_corr_ind[j] = False
    return high_corr_ind
def plot_feat_corr(df, title=''):
    corr = df.corr()

    group_cnt = df.columns.get_level_values(0).value_counts(sort=False)
    label, loc = zip(*group_cnt.sort_index().items())
    loc = [sum(loc[:i]) for i, l in enumerate(loc,0)]

    fig = plt.figure(figsize=(7.2, 6))
    cm = plt.pcolormesh(corr)
    plt.title(title)
    plt.xticks(loc, label)
    plt.yticks(loc, label)
    fig.colorbar(cm)
    fig.tight_layout()
    plt.show()

from my_eval import plotCM
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    plotCM(cm, ['pass','fail'])
    print('acc:{:.4f}\tuar:{:.4f}\n{}'.format(acc, uar, cm))


sample_fn = simple_sample
# sample_fn = SMOTE_sample
# train_x_sampled, train_y_sampled = sample_fn(train_x, train_y,0.5)


class BaseModel():
    def __init__(self, C_measure=1, C_meta=1, perc=100):
        self.clf_measure = SVC(C=C_measure,
                               kernel='linear',
                               class_weight='balanced',
                               probability=True)
        self.clf_meta = SVC(C=C_meta,
                            kernel='linear',
                            class_weight='balanced',
                            probability=True)
        self.selector = SelectPercentile(percentile=perc)

    def selector_support(self):
        return self.selector.get_support()

    def fit(self, x_measure, x_meta, y):
        self.measure_ind = feat_reduction(x_measure)
        x_measure_sel = x_measure.loc[:, self.measure_ind]
        # x_measure_sel = z_norm(x_measure_sel)
        # ANOVA f
        x_measure_sel = self.selector.fit_transform(x_measure_sel, y)

        self.clf_measure.fit(x_measure_sel, y)
        self.clf_meta.fit(x_meta, y)
        return self

    def decision_function(self, x_measure, x_meta, y, alpha=0.5):
        x_measure_sel = x_measure.loc[:, self.measure_ind]
        # x_measure_sel = z_norm(x_measure_sel)
        # ANOVA f
        x_measure_sel = self.selector.transform(x_measure_sel)

        s1 = self.clf_measure.decision_function(x_measure_sel)
        evaluate(y, s1 > 0)
        s2 = self.clf_meta.decision_function(x_meta)
        evaluate(y, s2 > 0)
        return ((1-alpha)*s1+alpha*s2) > 0

    def predict(self, x_measure, x_meta, y):
        x_measure_sel = x_measure.loc[:, self.measure_ind]
        # x_measure_sel = z_norm(x_measure_sel)
        # ANOVA f
        x_measure_sel = self.selector.transform(x_measure_sel)

        pred1 = self.clf_measure.predict(x_measure_sel)
        evaluate(y, pred1)
        pred2 = self.clf_meta.predict(x_meta)
        evaluate(y, pred2)
        return pred1, pred2


class DNNModel():
    def __init__(self, input_shape, lr=0.001):
        inputs = Input(shape=input_shape)
        x = Dense(512, activation='relu')(inputs)
        # x = Dense(256, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(2, activation='softmax')(x)

        self.model = Model(inputs, outputs)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=SGD(lr=lr),
                           metrics=['accuracy'])

    def fit(self, x, y, batch_size=8, epochs=10):
        return self.model.fit(x, y,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.1,
                              shuffle=True)

    def predict(self, x, y):
        pred = self.model.predict(x)
        evaluate(y, np.argmax(pred, axis=1))
        return pred

# %% Cross-Validation
K.clear_session()
all_true, all_pred = np.array([]), np.array([])
all_supports = []
for w in wafer_names:#[:1]:
    test_ind = raw['wafername_WS1'] == w
    test = raw.loc[test_ind].reset_index(drop=True)
    train = raw.loc[~test_ind].reset_index(drop=True)

    # split data ( with normalization )
    train, test = map(split, (train, test))
    train_y, train_x = train  # train_x: (train_measure, train_meta)
    test_y, test_x = test

    # sample
    # train_x, train_y = sample_fn(pd.concat(train_x,axis=1), train_y, 1)
    # train_x = split_x(train_x)

    # train_y_hot = to_categorical(train_y)
    test_y = test_y.values

    # =============================================================================
    # model
    # =============================================================================
    model = BaseModel(C_measure=1, C_meta=1, perc=60).fit(*train_x, train_y)
    # _,pred = model.predict(*test_x, test_y)  # direct prediction
    pred = model.decision_function(*test_x, test_y, alpha=0.8)  # fusion prediction

    # record high ranked features
    sup = train_x[0].columns[model.measure_ind][model.selector_support()].get_level_values(1)
    sup_v = model.selector.scores_[model.selector_support()]
    all_supports.append(sorted(zip(sup, sup_v), key=lambda x: x[1], reverse=True))

    # =============================================================================
    # model
    # =============================================================================
    # pred = np.ones((len(test_y), 2))
    # for a, b in zip(train_x, test_x):
    #     if a.shape[1] > 10:
    #         measure_ind = feat_reduction(a)
    #         a = a.loc[:, measure_ind]
    #         b = b.loc[:, measure_ind]
    #         # ANOVA f
    #         selector = SelectPercentile(percentile=40)
    #         a = selector.fit_transform(a, train_y)
    #         b = selector.transform(b)


    #     model = DNNModel(a.shape[1:], lr=0.001)
    #     model.fit(a, train_y_hot, epochs=16)
    #     pred *= model.predict(b, test_y)
    # pred = np.argmax(pred, axis=1)

    all_true = np.append(all_true, test_y)
    all_pred = np.append(all_pred, pred)

print("----- Overall -----")
evaluate(all_true, all_pred)
