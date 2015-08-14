from sklearn import tree
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

from pylab import *

class regressionUsingDtree:
    def __init__(self):
        pass

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
            #print "alles gut"
            #Step 3: Stack them vertically one by one
            normalizedData[:,col] =normalizedColumn
            #print normalizedData
        return normalizedData,desiredLabel

    def loadExperimentData(self):
        path = "./tuft_real_data/22June/extractedFeatures/"
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
            featureFileName = os.path.splitext(file_name)[0].split("/")[-1]
            #print featureFileName
            labels = np.loadtxt(fname=file_name,delimiter=',')
            if flagInitial:
                flagInitial = False
                trainLabel = labels
            else:
                trainLabel = np.concatenate((trainLabel,labels),axis=0)

        return trainData,trainLabel

    def dtreeRegressor(self):

        #trainingData,desiredLabel = self.loadExperimentData()
        trainingData,desiredLabel = self.normalizeColumnwiseData()
        coordinates_train, coordinates_test, windspeed_train, windspeed_test = cross_validation.train_test_split(trainingData,desiredLabel,test_size=0.30)
        _, coordinates_predict, _, windspeed_predict = cross_validation.train_test_split(coordinates_test, windspeed_test,test_size=0.04)
        curveFit = tree.DecisionTreeRegressor()
        #curveFit = tree.DecisionTreeClassifier(max_leaf_nodes=6)
        curveFit = curveFit.fit(coordinates_train, windspeed_train)
        predicted_speed = curveFit.predict(coordinates_predict)
        print "Prediction Score :", curveFit.score(coordinates_test, windspeed_test)
        print curveFit.predict([.8,.4,.4,.5,.3,.2])
        #predicted_speed = curveFit.predict(coordinates_predict)
        mse = mean_squared_error(windspeed_test, curveFit.predict(coordinates_test))
        rms = sqrt(mse)
        print "mse on the test data: ",mse

        errorbarValues = []
        #errorbins = [-4,-3,-2,-1,0,1,2,3,4,5]
        errorbins = np.arange(-30,30,1)
        print "errorbins:", errorbins
        for threshold in errorbins:
            correct_estimation = 0
            for i in range(len(predicted_speed)):
                if (windspeed_predict[i] - predicted_speed[i] <= threshold) and (windspeed_predict[i] - predicted_speed[i] > threshold-1):
                    correct_estimation += 1
            print "for threshold between: ", threshold ," and ",threshold-1," estimation: ", correct_estimation, " out of : ", len(windspeed_predict)       
            errorbarValues.append(correct_estimation)
        ###############################################################################
        # look at the results
        #plt.plot(np.arange(0,len(predicted_speed),1),predicted_speed, c='g')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 0.4
        ax.bar([i - width for i in errorbins],errorbarValues,width,color="y",alpha=0.7)
        #ax.bar(errorbins,errorbarValues,width,color="y",alpha=0.7)
        plt.xlabel("Estimation error(kmph)")
        plt.ylabel("Number of observation")
        plt.title("Error histogram DTR")
        ax.set_xlim(-25,25)
        plt.grid()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(0,len(predicted_speed),1),predicted_speed, c='g',marker='+', label='predicted speed')
        #plt.hold('on')
        ax.scatter(np.arange(0,len(windspeed_predict),1),windspeed_predict, c='r',marker='x', label='Actual data')
        ax.set_xlim(-2,len(windspeed_predict))
        ax.set_ylim(8,45)   

        plt.xlabel('Number of test cases')
        plt.ylabel('wind speed')
        plt.title('Decision tree Regression')
        ax.legend()
        
        for i in range(len(predicted_speed)):
            ax.annotate('', xy=(i, windspeed_predict[i]), xytext=(i, predicted_speed[i]),
                    arrowprops=dict(facecolor='b',alpha=0.5, shrink=0.03,headwidth=4.5,width=1.5,frac=0.4),
                    )
        plt.show()

if __name__ == "__main__":
    regressor = regressionUsingDtree() 
    start_time = time.time()
    regressor.dtreeRegressor()
    print("--- %s seconds ---" % (time.time() - start_time))