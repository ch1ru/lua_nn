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


def classify_graph():
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

def regression_graph():
  f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))
  ax2.set_title('Regression Predictions')

  df_output = pd.read_csv('output.csv')
  df_input = pd.read_csv('input.csv')

  y_original = []
  x_original = []

  for v in df_output.values:
      X.append( [v[0]] )
      y.append( v[1] )

  for v in df_input.values:
     x_original.append(v[0])
     y_original.append(v[1])

  ax2.plot(X, y, color='r')
  ax1.set_title('y=sin(x)')
  ax1.plot(x_original, y_original)


  df_training = pd.read_csv('trainingStats.csv')
  for v in df_training.values:
      epoch.append( v[0] )
      acc.append( v[1] )
      loss.append( v[2] )
  ax3.set_title('Loss and Accuracy through epochs')
  ax3.set_xlabel('Epoch')
  ax3.plot(acc, c='r', label='accuracy')
  ax3.plot(loss, c='b', label='loss')
  ax3.legend()

  plt.show()



regression_graph()

