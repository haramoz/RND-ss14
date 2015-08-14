import glob
import os
import numpy as np
import copy
import csv

'''This file collects the allready extracted features from the mentioned path and put them all 
together in same order in a single file for data and labels'''
class unifyingMultipleDatasets:
    def __init__(self,learningProblem,readPath,writePath):
        self.readPath = readPath
        self.writePath = writePath
        self.learningProblem = learningProblem

    def loadExperimentData(self):
        list_of_data_files = glob.glob(self.readPath+'data/*.csv')
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
        list_of_label_files = glob.glob(self.readPath+'labels/*.csv')
        list_of_label_files = sorted(list_of_label_files)
        flagInitial = True        
        for file_name in list_of_label_files:
            featureFileName = os.path.splitext(file_name)[0].split("/")[-1]
            #print featureFileName
            labels = np.loadtxt(fname=file_name,delimiter=',')
            if flagInitial:
                flagInitial = False
                trainLabel = labels
            else:
                trainLabel = np.concatenate((trainLabel,labels),axis=0)

        return trainData,trainLabel
    
    def normalizeColumnwiseData(self):
    
        trainingData,desiredLabel = self.loadExperimentData()
        #Step 1: Check how many columns are there
        noOfColumns = len(trainingData[0])
        trainDataArray = np.asarray(trainingData)
        print trainingData.shape , noOfColumns
        normalizedData = np.zeros(trainingData.shape)
        for col in range(noOfColumns):
            columnVal = np.asarray(trainingData[:,col])
            #print len(columnVal) , len(trainingData)
            #Step 2: For all the rows and specific column do the normalization 
            meanSubstracted = columnVal - np.mean(columnVal)
            normalizedColumn = meanSubstracted/np.std(columnVal)
            #Step 3: Stack them vertically one by one
            normalizedData[:,col] = normalizedColumn

        return normalizedData,desiredLabel

    '''This function unifies the data from multiple datasets and unifies them and creates
    resulting csv file'''

    def writeUnifiedFeatures(self):
        #path = self.writePath
        trainData,trainLabel = self.normalizeColumnwiseData()

        with open(self.writePath+'data/unifiedTrainData.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in trainData: 
                writer.writerow(i)


        with open(self.writePath+'labels/unifiedTrainDatalabel.csv', 'w') as csvfile2:
            writer2 = csv.writer(csvfile2,delimiter=',')
            for i in trainLabel:
                    writer2.writerow([i])    
    
if __name__ == "__main__":
    readPath = './tuft_real_data/24June/extractedFeatures/'
    writePath = './tuft_real_data/24June/extractedFeatures/unified/'
    learningProblem = "regression"
    unifyMultipleDatasets = unifyingMultipleDatasets(learningProblem,readPath,writePath) 
    #dataProcessor = processRealData("classification")
    unifyMultipleDatasets.writeUnifiedFeatures()
    print "Files are successfully written!!"