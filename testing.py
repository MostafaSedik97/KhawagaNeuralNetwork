from softmax_layer import SoftmaxLayer
from evaluation_module import Evaluation_Module
import time
import numpy as np
import pandas as pd
import FC
from optimizer import optimizer
from activations import relu, sigmoid, softmax
from deepNN import DeepNeuralNetwork
from data_module import getlabel, split, normalization, load_images_from_folder

np.fetaure_matrix, np.labels=load_images_from_folder('C:/Users/Lenovo/Desktop/New folder/TrainSampleLast')
np.normalized_features=[]
np.normalized_features=normalization(np.fetaure_matrix)
np.traininglabels,np.testlabels,np.trainingfeatures,np.testfeatures=split(np.normalized_features,np.labels)


'''pd.DataFrame(np.fetaure_matrix).to_csv("features4.csv")
pd.DataFrame(np.labels).to_csv("labels.csv")
pd.DataFrame(np.normalized_features).to_csv("normalized_features.csv")
pd.DataFrame(np.trainingfeatures).to_csv("trainingfeatures.csv")
pd.DataFrame(np.testfeatures).to_csv("testfeatures.csv")
pd.DataFrame(np.traininglabels).to_csv("traininglabels.csv")
pd.DataFrame(np.testlabels).to_csv("testlabels.csv")'''

x_train = np.trainingfeatures
y_train = np.traininglabels
x_val = np.testfeatures
y_val = np.testlabels


dnn = DeepNeuralNetwork(epochs=30,l_rate=0.001,optimizer_name="adaDelta",raw=0.99,epsilon=10^-8)
dnn.add(size=[784,128],activation=relu)
dnn.add(size=[128,64],activation=relu)
dnn.addout(size=[64, 10],activation=softmax)
predict = dnn.train(x_train, y_train, x_val, y_val)
eval = Evaluation_Module(y_val, predict)
print(eval.confusion_matrix())
print(eval.percision())
print(eval.recall())
print(eval.f1_score())
print(eval.accuracy())