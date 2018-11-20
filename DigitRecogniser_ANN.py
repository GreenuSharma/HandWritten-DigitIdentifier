# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:38:24 2018

@author: Greenu
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier 
from mnist import MNIST

def main():
    print("Loading training and test data..........") 
    mndata=MNIST("samples")
    images_Training,labels_Training=mndata.load_training()
    images_Testing,labels_Testing=mndata.load_testing()
    
    print("Creating the Neural Network model with 3 hidden layers of 200 neurons each.....")
    ANN = MLPClassifier(solver='adam', alpha=1e-5,
          hidden_layer_sizes=(200,200,200),activation='relu', random_state=1)
    print("Training the model............")
    ANN.fit(images_Training,labels_Training)
    print("Predicting with the test data...........")
    Ypredict=ANN.predict(images_Testing)
    #print(Ypredict)
    print("Scoring the prediction..........")
    Score=ANN.score(np.array(images_Testing),np.array(labels_Testing))
    print("Score for the model =",Score)
    
    print("Writing prediction result to Digits_ANN.csv")
    ImageId=pd.Series(range(1, np.array(images_Testing).shape[0]+1))
    Ypredict=pd.DataFrame(Ypredict)
    frames=[ImageId,Ypredict[0]]
    Predictions=pd.DataFrame(pd.concat(frames,axis=1,ignore_index=True))
    Predictions.columns=['ImageId','Label']
    Predictions.to_csv("Digits_ANN.csv",index=False,columns=['ImageId','Label'])
    
    print("Code Complete")

if __name__=="__main__":
    main()