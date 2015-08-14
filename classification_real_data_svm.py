import copy
import csv
#from sklearn import tree
import numpy as np
from sklearn import cross_validation
from sklearn.externals.six import StringIO
import pydot
import os
from sklearn.svm import SVC
import glob
import time

class classficationOfRealDataUsingSVM:
    
    def __init__(self,test_size):
        self.test_size = test_size

    def loadExperimentData(self):
        path = "./tuft_real_data/24June/extractedFeatures/"
        list_of_data_files = glob.glob(path+'data/*.csv')
        list_of_data_files = sorted(list_of_data_files)
        flagInitial = True
        
        for file_name in list_of_data_files:
            featureFileName = os.path.splitext(file_name)[0].split("/")[-1]
            #print featureFileName
            data = np.loadtxt(fname=file_name,delimiter=',')
            if flagInitial:
                flagInitial = False
                trainData = data
            else:
                trainData = np.vstack((trainData,data))

        #For reading the labels
        list_of_label_files = glob.glob(path+'labels/*.csv')
        list_of_label_files = sorted(list_of_label_files)
        flagInitial = True        
        for file_name in list_of_label_files:
            #print file_name
            featureFileName = os.path.splitext(file_name)[0].split("/")[-1]
            #print featureFileName
            labels = np.loadtxt(fname=file_name,delimiter=',')
            if flagInitial:
                flagInitial = False
                trainLabel = labels
            else:
                trainLabel = np.concatenate((trainLabel,labels),axis=0)

        return trainData,trainLabel
        
    def svmClassifier(self):
        trainData,trainLabel = self.loadExperimentData()

        print "total available data",len(trainData)
        data_train,data_test,label_train,label_test = cross_validation.train_test_split(trainData,trainLabel,test_size=self.test_size)
        
        clf = SVC(C=1.9,gamma=0.001)
        clf = clf.fit(data_train,label_train)
        print "prediction Accuracy",clf.score(data_test,label_test)
        print "Number of support vectors used:",len(clf.support_vectors_)

        #Use the cross_validation score
        clf2 = SVC(C=1.9,gamma=0.001)
        cv = cross_validation.ShuffleSplit(len(trainData), n_iterations=10,test_size=self.test_size, random_state=0)
        scores = cross_validation.cross_val_score(clf2, trainData, trainLabel, cv=cv)
        print scores
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100 / 2)

if __name__ == "__main__":
    test_size = 0.3
    classifier = classficationOfRealDataUsingSVM(test_size) 
    start_time = time.time()
    classifier.svmClassifier()
    print("--- %s seconds ---" % (time.time() - start_time))
