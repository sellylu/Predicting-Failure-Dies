# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:09:48 2018

@author: selly
"""
import matplotlib.pyplot as plt
import itertools
import numpy as np

params = {'legend.fontsize': 20,
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize': 20,
         'ytick.labelsize': 20}
plt.rcParams.update(params)

def plotCM(cm, classes,
           normalize=False,
           title='Confusion matrix',
           cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(5.5,5))
    # plt.pcolormesh(cm, cmap=cmap)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)#, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim((1.5, -0.5))

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=20)

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

def calAccuracy(true, pred, class_names, title='', draw=True):
	from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
	acc = accuracy_score(true, pred)
	uar = recall_score(true, pred, average='macro')
	cm  = confusion_matrix(true, pred, range(len(class_names)))
	print("acc: {}\tuar: {}\ncm: \n{}".format(acc, uar, cm))
	if draw:
		plotCM(cm, classes=class_names, title='Confusion matrix('+title+')')
	return (acc, uar, cm)