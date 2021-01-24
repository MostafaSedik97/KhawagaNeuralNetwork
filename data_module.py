import glob
import matplotlib.image as mpimg 
import numpy as np
from numpy import savetxt
import os
import pandas as pd 
import pickle
import webbrowser


def load_images_from_folder(folder):
    np.images = []
    np.y=[]
    np.x=[]
    i=0
    for root, dirs, files in os.walk(folder):
        for filename in files:
          np.image = mpimg.imread(folder+ '/' + filename)
          np.image1 = np.image.flatten() 
          np.pixel_size=np.image.shape   
          if np.image is not None:
            np.images.append(np.image1)
            temp = np.zeros(10)
            print(filename)
            temp[int(filename[6])] = 1
            np.y.append(temp)
        #print(np.images)
    np.x=np.array(np.y)   
    return np.images, np.x


def normalization(fetaure_matrix):
  x=np.max(np.fetaure_matrix)
  np.y=np.fetaure_matrix/x
  return np.y


    
def load_images_from_folder_compressed(folder):
    np.images = []
    np.y=[]
    i=0
    for filename in os.listdir(folder):
      np.image = mpimg.imread(os.path.join(folder,filename))
      
      np.pixel_size=np.image.shape
      np.fetaure_matrix=np.zeros((np.pixel_size))
      np.dot(np.image[...,:3], [0.2989, 0.5870, 0.1140])
      bottom_half = np.image[np.image.shape[0]//2:, :]
      np.image = np.sum(bottom_half, axis=0)
      
 
  
      if np.image is not None:
        np.images.append(np.image)
    return np.images

def getlabel(folder):
  np.y=[]
  np.x=[]
  for root, dirs, files in os.walk(folder):
      for filename in files:
        if np.image is not None:
          temp = np.zeros(10)
          print(filename)
          temp[int(filename[6])] = 1
          np.y.append(temp)
  np.x=np.array(np.y)
  return np.x

def split(feature,label):
  indices = np.random.permutation(feature.shape[0])
  training_idx, test_idx = indices[:int(0.8*feature.shape[0])], indices[int(0.8*feature.shape[0]):]
  traininglabels, testlabels,trainingfeatures,testfeatures = label[training_idx,:],label[test_idx,:],feature[training_idx,:],feature[test_idx,:]
  return traininglabels,testlabels,trainingfeatures,testfeatures





'''np.labels=getlabel(folder)
np.normalized_features=[]
np.normalized_features=normalization(np.fetaure_matrix)
np.traininglabels,np.testlabels,np.trainingfeatures,np.testfeatures=split(np.normalized_features,np.labels)

x_train = np.trainingfeatures
y_train = np.traininglabels
x_val = np.testfeatures
y_val = np.testlabels'''

'''pd.DataFrame(np.fetaure_matrix).to_csv("features4.csv")
pd.DataFrame(np.labels).to_csv("labels.csv")
pd.DataFrame(np.normalized_features).to_csv("normalized_features.csv")
pd.DataFrame(np.trainingfeatures).to_csv("trainingfeatures.csv")
pd.DataFrame(np.testfeatures).to_csv("testfeatures.csv")
pd.DataFrame(np.traininglabels).to_csv("traininglabels.csv")
pd.DataFrame(np.testlabels).to_csv("testlabels.csv")'''