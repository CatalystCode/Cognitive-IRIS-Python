# http://scikit-learn.org/stable/modules/model_evaluation.html
# PRECISION: What percent of positive predictions were correct? PR=TP/Total Predicted positive
# RECALL: What percent of the positive cases did you catch? R=TP/Total Real Positive

from sklearn import metrics
import requests
import os
import base64
import pandas as pd
import sys
import urllib2
import random


import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

from sklearn.metrics import confusion_matrix

from PIL import Image, ImageDraw
# try:
#     import cv
# except ImportError:
#     print 'Could not import cv, trying opencv'
#     import opencv.cv as cv

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def crop_images(inputDirectory, numOfCrops, outputDirectory):
  # Generate crop images from failed image
  for file in os.listdir(inputDirectory):
    if file.endswith('.jpg'):
      filePath = os.path.join(inputDirectory, file)
      print(filePath)
      crop_image(filePath, numOfCrops, outputDirectory, file)

def crop_image(filePath, numOfCrops, outputDirectory, fileName):
  # Generate crop images from image
  source_image = Image.open(filePath)
  source_width, source_height = source_image.size
  print 'Image was {}x{}'.format(source_width, source_height)
  for x in range(0, numOfCrops):
    ratio = round(random.uniform(0.8, 1),2)
    print 'cropping ratio', ratio
    target_width = source_width * ratio
    target_height = source_height * ratio

    target_x1 = (source_width - target_width)/2
    target_y1 = (source_height - target_height)/2

    print 'Image new {}x{}'.format(target_width, target_height)
    coords = (target_x1, target_y1, target_width, target_height)
    print 'Cropping to', coords
    newFilePath = '{}{}_{}'.format(outputDirectory, x, fileName)
    final_image = source_image.crop(coords)
    final_image.save(newFilePath)
    return newFilePath

def plot_project(y_true, y_pred):
  print(metrics.classification_report(y_true, y_pred))
  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_true, y_pred)
  np.set_printoptions(precision=2)

  # Plot non-normalized confusion matrix
  plt.figure()
  class_names = sorted(set(y_true))
  plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title='Confusion matrix, without normalization')

  # Plot normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

  plt.show()

