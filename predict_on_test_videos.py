#Step1 Read the files
#Step2 Process the files in chunks
#Step3 Replace the chunks with stats data
#Step3.5 Train the classifer with the stats data approach
#Step4 Predict the result for each of the stats data
#Step5 Interpolate the results
import copy
import csv
import numpy as np
from sklearn import cross_validation
import pydot
import os
import matplotlib.pyplot as plt
import glob,math
from sklearn.svm import SVC,SVR
import time
#TO DOs
#Train it with speed data
#Output the speed predicted
class predictRealData:
    def __init__(self,learningProblem,seriesLengthInSeconds,readPath,writePath,test_size,slidingWindow=False,flag=True):
        self.frameRatePerSecond = 120
        self.seriesLengthInSeconds = seriesLengthInSeconds #0.5 #0.0625
        self.featuresPerSeries = np.round(self.seriesLengthInSeconds * self.frameRatePerSecond)
        self.framesInTheSlidingWindow = int(self.featuresPerSeries/3)
        self.learningProblem = learningProblem
        #This is the path from where the experiment results will be read
        self.readPath = readPath
        self.writePath = writePath
        self.slidingWindow = slidingWindow
        #print "features per series",2*self.featuresPerSeries
        self.numberOfFeatures = 6
        self.test_size = test_size
        if learningProblem != "regression":
            self.clf = SVC(C=1.6,gamma=0.002)
        else:
            self.clf = SVR(kernel='rbf',C=1.2, epsilon=1.38)
        self.flagPredict = flag
        if flag:
            self.writePath = './tuft_real_data/17June/extractedFeatures/'

    def outlierDetection(self,listItem):
        #Reference http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
        listItem = copy.deepcopy(listItem)
        #Step1: Sort the list
        sortedListItem = sorted(listItem)
        #Step2: Find the median value
        medianValue = self.calculateMedian(sortedListItem)
        #Step3: Find the lower quartile
        lowerQuartile = sortedListItem[:len(listItem)//2]
        lowerQuartileValue = self.calculateMedian(lowerQuartile)
        #Step4: Find the upper quartile
        upperQuartile = sortedListItem[len(listItem)//2:]
        upperQuartileValue = self.calculateMedian(upperQuartile)
        #Step5 find interquartile range
        interquartileRange = (upperQuartileValue - lowerQuartileValue)*1.5
        innerFences = ((lowerQuartileValue - interquartileRange),(upperQuartileValue + interquartileRange)) 
        #Step6 find outerquartile range
        interquartileRange2 = (upperQuartileValue - lowerQuartileValue)*3.0
        outerFences = ((lowerQuartileValue - interquartileRange2),(upperQuartileValue + interquartileRange2)) 
        
        #Step7 Calculate the major and minor outliers
        majorOutliers = []
        minorOutliers = []
        #print "outerFences",outerFences
        for i in listItem:
            if i < outerFences[0] or i > outerFences[1]:
                majorOutliers.append(i)
            if (i < innerFences[0] or i > innerFences[1]) and i not in majorOutliers:
                minorOutliers.append(i) 
            return majorOutliers,minorOutliers
    
    def calculateMedian(self,listItem2):
        listItem2 = copy.deepcopy(listItem2)
        if len(listItem2) == 0:
            print "Empty array"
            return None
        if len(listItem2)%2 == 0:
            middleValue = len(listItem2)//2
            medianValue = (listItem2[middleValue] + listItem2[middleValue+1])/2.0
        else:
            medianValue = (listItem2[len(listItem2)//2])
        #print medianValue
        return medianValue

    def rangeFinder(self,listItem3,outliers=None):
        listItem3 = copy.deepcopy(listItem3)
        outliers = copy.deepcopy(outliers)
        withoutOutliersList = []
        if outliers != None and len(outliers) > 0:
            for v in listItem3:
                if v not in outliers:
                    withoutOutliersList.append(v)
            valueRange = (min(withoutOutliersList),max(withoutOutliersList))
        else:
            valueRange = (min(listItem3),max(listItem3))
            withoutOutliersList = listItem3

        #print "range",valueRange[0],valueRange[1]
        medianValue = self.calculateMedian(withoutOutliersList)
        std = np.std(withoutOutliersList)
        return valueRange,std
    
    '''This function reads files and creates a feature list which are sent to 
    writeFeatures() function which creates .csv files contaning features and
    separate files contaning related labels'''

    def processFiles(self):
        list_of_files = glob.glob(self.readPath+'*.txt')
        #list_of_files = glob.glob('./tuft_real_data/22April/*.txt')
        colorCode = -1
        colors = ["ro","bo","mo","go","yo"]
        numberOfFiles = 1
        #For median Plot
        mediansX = []
        mediansY = []
        speeds = []
        averagesX = []
        averagesY = []
        '''For the speed vs outlier analysis'''
        speedsStack = []
        outlierStackX = []
        outlierStackY = []
        for fileName in list_of_files:
            print os.path.splitext(fileName)[0].split("/")[-1]
            #data_list = open( fileName, "r" ).readlines()
        
            featureSet = []
            normalisedX = []
            normalisedY = []
            typeOfFlow = None
            speed = 0.0
            #Saving x and y coordinates separately for range analysis
            xValues = []
            yValues = []
            colorCode += 1 

            #For one file should be one origin x and y
            originX = None
            originY = None
            flagOrigin = True
            with open(fileName,'r') as f:
                for line in f:
                    if "TYPE" in line and line != "":
                        typeOfFlow = line.split()
                        typeOfFlow = typeOfFlow[1]
                        #print typeOfFlow 
                    elif "SPEED" in line and line != "":
                        speed = line.split()
                        speed = speed[1]   
                        #print speed
                    elif "VECTOR" in line and line != "":
                        coordinates = line.split()
                        #print coordinates[0]
                        #TO DO: Only fixed start coordinates can be used.
                        if flagOrigin:
                            originX = float(coordinates[1])
                            originY = float(coordinates[2])
                            flagOrigin = False
                        #normalisedX = float(coordinates[3]) - float(coordinates[1])
                        #normalisedY = float(coordinates[4]) - float(coordinates[2])
                        #xValues.append(coordinates[1])
                        xValues.append(float(coordinates[3]) - float(coordinates[1]))
                        #yValues.append(coordinates[2])
                        yValues.append(float(coordinates[4]) - float(coordinates[2]))
                        #featureSet.append(normalisedX)
                        #featureSet.append(normalisedY)
                    elif "FRAME" in line and line != "":
                        frames = line.split()
                        vectors = len(xValues) + 1
                        '''if vectors in [3660,3690,3900,4230,4470,4500,5730,9900]:
                            print frames[1]'''
                featureFileName = os.path.splitext(fileName)[0].split("/")[-1]
            
                        
            numberOfFiles += 1
            
            #Calculating the outliers
            majorOutliersX,minorOutliersX = self.outlierDetection(xValues)
            majorOutliersY,minorOutliersY = self.outlierDetection(yValues)
            outlierStackX.append(len(minorOutliersX))
            outlierStackY.append(len(minorOutliersY))
            speedsStack.append(speed)

            #Calculating Range of the vectors
            valueRangeX,medianValueX = self.rangeFinder(xValues,majorOutliersX)
            valueRangeY,medianValueY = self.rangeFinder(yValues,majorOutliersY)

            #Plotting the median and range values
            
            mediansX.append(medianValueX)
            mediansY.append(medianValueY)
            speeds.append(speed)
                        
            '''Note: The way this logic is implemented the outliers in y 
            does not come into play at all'''

            withoutOutliersListX = []
            withoutOutliersIndex = []
            for i in range(len(xValues)):
                if xValues[i] not in majorOutliersX:
                    withoutOutliersListX.append(xValues[i])
                    withoutOutliersIndex.append(i)
            withoutOutliersListY = []
            for i in range(len(yValues)):
                if i in withoutOutliersIndex:
                    withoutOutliersListY.append(yValues[i])
                    #featureSet.append(xValues[i])
                    #featureSet.append(yValues[i])   
                    normalisedX.append(xValues[i])
                    normalisedY.append(yValues[i])         
            
            #Function call for the supplying the polar coordinates
            featureSet2 = []
            featureSet2 = self.supplyStats(normalisedX,normalisedY)
            if self.flagPredict:
                self.predictResult(featureSet2)
                typeOfFlow = ['NA']
                speed = 'NA'
            self.writeFeatures(featureSet2,featureFileName,typeOfFlow,speed)

        return featureSet
    
    '''This function takes input of the xdiff, ydiff values and returns the blocks of
    Stats in place of the raw data'''
    def supplyStats(self,xDiff,yDiff):
        xDiff = copy.deepcopy(xDiff)
        yDiff = copy.deepcopy(yDiff)
        xDiffBlocks = self.cutThelengthOfdata(xDiff,self.featuresPerSeries)
        yDiffBlocks = self.cutThelengthOfdata(yDiff,self.featuresPerSeries)
        #print xDiffBlocks
        statsAsFeatures = []
        for i in range(len(xDiffBlocks)):
            #At this index the block of x values are stored, this will be processed for stats collection
            distances,angles,featureSet2 = self.supplyPolarCoordinates(xDiffBlocks[i],yDiffBlocks[i])
            statsAsFeatures.append(np.mean(distances))
            statsAsFeatures.append(np.mean(angles))
            valueRange,std = self.rangeFinder(xDiffBlocks[i])
            statsAsFeatures.append(std)
            statsAsFeatures.append(valueRange[1]-valueRange[0])
            valueRange,std = self.rangeFinder(yDiffBlocks[i])
            statsAsFeatures.append(std)
            statsAsFeatures.append(valueRange[1]-valueRange[0])

        #print "Mean Length of vector,Mean angle,median distance,range diff distance,median angles,range diff angles"
        return statsAsFeatures

    '''This function returns the blocks of the feature to be exchanged by the stats data'''
    def cutThelengthOfdata(self,dataInput,lengthOfVector):
        counter = 0
        eachLine = []
        dataOutput = []
        for i in range(len(dataInput)): #1,2,3,4,5,6,4,3,2,1
            if counter < lengthOfVector:
                eachLine.append(dataInput[i])
                counter += 1
            else:
                counter = 1                
                dataOutput.append(eachLine)
                eachLine = []
                eachLine.append(dataInput[i])
        return dataOutput

    '''Distance calculator'''
    def distanceCalculator(self,diff1,diff2):
        distance = np.sqrt(pow(diff1,2)+pow(diff2,2))
        return distance


    '''This function calculates the polar coordinates for supplied list of cartesian coordinates'''
    def supplyPolarCoordinates(self,xDiff,yDiff):
        xDiff = copy.deepcopy(xDiff)
        yDiff = copy.deepcopy(yDiff)
        polarDistances = [] 
        angles = []
        features = []
        for i in range(len(xDiff)):
            dist = self.distanceCalculator(xDiff[i],yDiff[i])
            polarDistances.append(dist)
            #theta = math.degrees(math.atan2(yDiff[i],xDiff[i]))
            theta = math.atan2(yDiff[i],xDiff[i])
            angles.append(theta)
            features.append(dist)
            features.append(theta)
        return polarDistances,angles,features

    '''This function writes the vector series data and labels in .csv format.
    The length of the series depends on the parameters in init'''
    def writeFeatures(self,fileContent,fileName,typeOfFlow,speed):
        features = copy.deepcopy(fileContent)
        #print features
        #speed = float(speed)
        if speed == 'NA' and self.learningProblem != "classification":
            print "speed not available"
            return None
        else:
            print typeOfFlow,speed


        #path = './tuft_real_data/3May/extractedFeatures/'
        path = self.writePath

        with open(path+'data/'+fileName+'.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            eachLine = []
            counter = 0
            counter2 = 0
            for i in features: 
                eachLine.append(i)
                counter += 1
                if counter%(self.numberOfFeatures) == 0:
                    writer.writerow(eachLine)
                    eachLine = []
                    counter2 += 1
            

        with open(path+'labels/'+fileName+'label.csv', 'w') as csvfile2:
            writer2 = csv.writer(csvfile2,delimiter=',')
            counter = 0
            for i in features: 
                counter += 1
                if counter%(self.numberOfFeatures) == 0:
                    if self.learningProblem == "classification":
                        writer2.writerow(typeOfFlow)
                    else:
                        #print "regression"
                        writer2.writerow([speed])

    def normalizeColumnwiseData(self):
        trainingData,desiredLabel = self.loadExperimentData()
        #Step 1: Check how many columns are there
        noOfColumns = len(trainingData[0])
        trainDataArray = np.asarray(trainingData)
        #print trainingData.shape , noOfColumns
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
        #print normalizedData.shape
        return normalizedData,desiredLabel

    def loadExperimentData(self):
        path = "./tuft_real_data/13June/extractedFeatures/"
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
        
    def svmClassifier(self):
        trainData,trainLabel = self.normalizeColumnwiseData()

        print "total available data",len(trainData)
        data_train,data_test,label_train,label_test = cross_validation.train_test_split(trainData,trainLabel,test_size=self.test_size)
        
        #self.clf = SVC(C=1.6,gamma=0.002)
        self.clf = self.clf.fit(data_train,label_train)
        print "prediction Accuracy",self.clf.score(data_test,label_test)
        print "Number of support vectors used:",len(self.clf.support_vectors_)

        '''#Use the cross_validation score
        clf2 = SVC(C=1.6,gamma=0.002)
        cv = cross_validation.ShuffleSplit(len(trainData), n_iterations=3,test_size=self.test_size, random_state=0)
        scores = cross_validation.cross_val_score(clf2, trainData, trainLabel, cv=cv)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100 / 2)'''

    def svmRegressor(self):
        trainData,trainLabel = self.loadExperimentData()

        print "total available data",len(trainData)
        data_train,data_test,label_train,label_test = cross_validation.train_test_split(trainData,trainLabel,test_size=self.test_size)
        
        #self.clf = SVC(C=1.6,gamma=0.002)
        self.clf = self.clf.fit(data_train,label_train)
        print "prediction Accuracy",self.clf.score(data_test,label_test)
        print "Number of support vectors used:",len(self.clf.support_vectors_)

        '''#Use the cross_validation score
        clf2 = SVC(C=1.6,gamma=0.002)
        cv = cross_validation.ShuffleSplit(len(trainData), n_iterations=3,test_size=self.test_size, random_state=0)
        scores = cross_validation.cross_val_score(clf2, trainData, trainLabel, cv=cv)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100 / 2)'''

    def predictResult(self,features):
        testData = copy.deepcopy(features)
        results = []
        eachLine = []
        counter = 0
        confidence = []
        for i in testData: 
            eachLine.append(i)
            counter += 1
            if counter%(self.numberOfFeatures) == 0:
                results.append(self.clf.predict(eachLine)[0])
                #confidence.append(self.clf.predict_proba(eachLine))
                eachLine = []
        #print results
        #print confidence
        self.returnFrames(results)

    def returnFrames(self,results):
        results = copy.deepcopy(results)
        lastResult = 1
        frameCounter = 0
        for r in results:
            if r != lastResult:
                print frameCounter*self.featuresPerSeries,lastResult
                lastResult = r
            frameCounter += 1

        print frameCounter*self.featuresPerSeries,lastResult
    
if __name__ == "__main__":
    predictPath = './tuft_real_data/17June/'
    readPath = './tuft_real_data/13June/'
    writePath = './tuft_real_data/13June/extractedFeatures/' 
    slidingWindow = False
    timeseriesLength = 1 #0.0625 #0.066
    test_size = 0.7
    
    #dataProcessor2.processFiles()

    #dataProcessor = predictRealData("classification",timeseriesLength,predictPath,writePath,test_size,slidingWindow,True)
    dataProcessor = predictRealData("regression",timeseriesLength,predictPath,writePath,test_size,slidingWindow,True)
    #dataProcessor.svmClassifier()
    start_time = time.time()
    dataProcessor.svmRegressor()
    print("--- %s seconds ---" % (time.time() - start_time))
    #dataProcessor.processFiles()
