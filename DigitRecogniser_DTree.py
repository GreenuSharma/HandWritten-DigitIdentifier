# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 09:08:35 2018

@author: Greenu
"""
def main():
    from mnist import MNIST
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    
    mndata=MNIST("samples")
    images_Training,labels_Training=mndata.load_training()
    images_Testing,labels_Testing=mndata.load_testing()
    
    DigitRecogniser_model= DecisionTreeClassifier()
    print("training...................")
    DigitRecogniser_model.fit(images_Training,labels_Training)
    print("Evaluating....................")
    preds=DigitRecogniser_model.predict(np.array(images_Testing))
    score=DigitRecogniser_model.score(np.array(images_Testing),np.array(labels_Testing))
    
    print("Score = ", score)
    
    r=0
    w=0
    num=len(images_Testing)
    for i in range(0,num):
        if preds[i] == labels_Testing[i]:
            r+=1
        else:
            w+=1
    print("tested ", num, " digits")
    print("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
    print("got correctly ", float(r)*100/(r+w), "%")
    
    print("Writing prediction result to digits_DTree.txt")
    f=open("Digits_DTree.txt","a+")
    for i in range(0 ,num):
        f.write(str(preds[i]))
        f.write(",")
        f.write(str(labels_Testing[i]) + "\n")
    f.close()
    print("Code Complete")
#print("score")
#print(preds)
#print(labels_Testing)

#print(labels_Training)
#print (labels_Training[1])

#print(score)
if __name__=="__main__":
    main()