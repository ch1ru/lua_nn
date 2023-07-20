from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import random
import pandas as pd



X = []
y = []
loss = []
acc = []
epoch = []

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title('Classification Predictions')

df_output = pd.read_csv('output.csv')
for v in df_output.values:
    X.append( [v[0], v[1]] )
    y.append( v[2] )



for i in range(len(X)):
    if y[i] == 0:
      marker = 'ro'
      label='Class 1'
    elif y[i] == 1:
      marker = 'bo'
      label = 'Class 2'
    elif y[i] == 2: 
      marker = 'yo'
      label='Class 3'
    else:
        marker = 'go'
        label='Class 4'

    ax1.plot(X[i][0], X[i][1], marker, label=label)


handles, labels = ax1.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys())
ax1.set_xticks([])
ax1.set_yticks([])


df_training = pd.read_csv('trainingStats.csv')
for v in df_training.values:
    epoch.append( v[0] )
    acc.append( v[1] )
    loss.append( v[2] )


ax2.set_title('Loss and Accuracy through epochs')
ax2.set_xlabel('Epoch')
ax2.plot(acc, c='r', label='accuracy')
ax2.plot(loss, c='b', label='loss')
ax2.legend()

plt.show()




