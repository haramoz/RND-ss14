import copy
import csv
from sklearn import tree
import numpy as np
from sklearn import cross_validation
from sklearn.externals.six import StringIO
import pydot
import os
import matplotlib.pyplot as plt
import glob
import loadExperimentData
import time


''' Here the turbulent flow is represented by 2, separation as 3 and Laminar as 1'''
class classficationWithRealDataUsingDTree:
    def __init__(self,test_size):
        self.test_size = test_size
       
    def dtreeClassifier(self):
        path = "./tuft_real_data/24June/extractedFeatures/"
        list_of_data_files = glob.glob(path+'data/*.csv')
        #list_of_data_files = glob.glob('./tuft_real_data/13May/extractedFeatures/data/*.csv')
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
       
        print "total available data",len(trainData)
        data_train,data_test,label_train,label_test = cross_validation.train_test_split(trainData,trainLabel,test_size=self.test_size)
        print "length of traindata and testdata",len(data_train),len(data_test)
        print "length of total samples:",len(trainData)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(data_train,label_train)
        print "prediction Accuracy",clf.score(data_test,label_test)
        values = clf.tree_.value
        print "size of the tree:",len(values)
        #print "size of left values:",len(clf.tree_.children_left)
        #print "size of right values:",len(clf.tree_.children_right)
        print "feature importance",clf.feature_importances_
        #kf = cross_validation.KFold(len(trainData), n_folds=3,shuffle=True)
        
        #Use the cross_validation score
        clf2 = tree.DecisionTreeClassifier()
        cv = cross_validation.ShuffleSplit(len(trainData), n_iterations=10,test_size=self.test_size, random_state=0)
        scores = cross_validation.cross_val_score(clf2, trainData, trainLabel, cv=cv)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100 / 2)

        #For creating the pdf version of the decision tree
        """with open("classficationWithRealData.dot", 'w') as f:
            f = tree.export_graphviz(clf, out_file=f)
        os.unlink('classficationWithRealData.dot')
        classification_data = StringIO()
        tree.export_graphviz(clf, out_file=classification_data)
        graph = pydot.graph_from_dot_data(classification_data.getvalue())
        graph.write_pdf("classficationWithRealData.pdf")"""

        '''test = np.random.rand(200,6)
        test = test*200
        pred = clf.predict(test)
        for i in range(len(test)):
            print test[i],pred[i]   
            #print test'''


        '''# Parameters
        n_classes = 3
        plot_colors = "br"
        plot_step = 1.0
        target_names = [1,2]

        # Plot the decision boundary
        #plt.subplot(2, 3, pairidx + 1)
        #print data_test[:, 0].min(),data_test[:, 1].min()
        y = label_test
        #In this stage they are picking up the min max value of the features we need
        x_min, x_max = data_test[:, 0].min() - 1, data_test[:, 0].max() + 1
        y_min, y_max = data_test[:, 1].min() - 1, data_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Greens)

        plt.xlabel("X-values")
        plt.ylabel("Y-Values")
        plt.axis("tight")

        # Plot the training points
        #for i, color in zip(range(n_classes), plot_colors):
        #    idx = np.where(y == i)
        #    plt.scatter(data_test[idx,0], data_test[idx,1], c=color, label=target_names[i],
        #                cmap=plt.cm.Paired)

        plt.axis("tight")

        plt.suptitle("Decision surface of a decision tree using paired features")
        #plt.legend(handles=('L','T','S'),loc='best')
        plt.show()'''

if __name__ == "__main__":
    test_size = 0.3
    classifier = classficationWithRealDataUsingDTree(test_size) 
    start_time = time.time()
    classifier.dtreeClassifier()
    print("--- %s seconds ---" % (time.time() - start_time))
