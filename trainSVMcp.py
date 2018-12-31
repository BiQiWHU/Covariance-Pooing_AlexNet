# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 00:01:51 2018

@author: 2009b_000
an implementation of SVM classification based on conv. features and CP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:56:19 2018

@author: 2009b_000

in this code, we will:
    (1) use the trained CNN model as a feature extractor
    (2) send all these training samples into a SVM classifier, train the model 
    (3) test samples are also extracted by the trained CNN, and prediected by the SVM model
 
(with sklearn package)

"""
from networkforSVM import alex_net
from tfdata import *
import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn import preprocessing
import time
from cp import cp
from cp import spilt_avr_resize
from cp import calc_featvec
 
start = time.clock()

### input the training samples
# Dataset path
train_tfrecords = 'train.tfrecords'
test_tfrecords = 'test.tfrecords'

# load data
# we will discuss the problem of batch size later
batch_size=20

imgtrain,labeltrain=input_pipeline(train_tfrecords,batch_size,is_shuffle=False,is_train=False)
imgtest,labeltest=input_pipeline(test_tfrecords,batch_size,is_shuffle=False,is_train=False)

with tf.variable_scope('model_definition') as scope:
    train_c1,train_c2,train_c3,train_c4,train_c5,train_f7=alex_net(imgtrain,train=False)    
    scope.reuse_variables()
    test_c1,test_c2,test_c3,test_c4,test_c5,test_f7=alex_net(imgtest,train=False)
    
    ## and please do not reshape or concate tensors at here, too many problems
    ## use conv5 features
    featuretrain=train_c5
    featuretest=test_c5
    
         
saver=tf.train.Saver()

### input the pre-trained CNN, extract features
with tf.Session() as sess:
    saver.restore(sess,'checkpoint/my-model.ckpt-2000')
    
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    
    ## save feature vectors and label for training samples
    Xtemp1=[]
    Ytemp1=[]
    #global Xtrain
    #global Ytrain
    Xtrain=[]
    Ytrain=[]
    
    ## save feature vectors and label for test samples
    Xtemp2=[]
    Ytemp2=[]
    #global Xtest
    #global Ytest
    Xtest=[]
    Ytest=[]
    ### note: here u may have to change the data format like in the link
    ### https://blog.csdn.net/qq_27756361/article/details/80479278
    ### Line 140-146
    
    ## train  1260 smples 63 rounds
    for i in range(63):
        print('this is the train loop:')
        print(i)
        ## the data format that sess.run get is tensor,not ndarray
        Xtemp1a,Ytemp1a=sess.run([featuretrain,labeltrain])      
        ### here we have to concate all the Xtemp1&Ttemp1 into Xtrain&Ytrain
        ## how to concate c1,c2,...,c5 into a densefeature (named featuretrain/test)??
        ## coding here
        
        # convert np.ndarray into string so that can be fed into SVM
        # does Y need reshape？
        #print('Xtemp1a and its shape')
        #print(Xtemp1a.shape)
        
        ## here d=64,s=14
        ## note that for AlexNet,the conv5 layer got a tensor of 14*14*256
        Xtemp1=cp(Xtemp1a,64,14,batch_size)

        Ytemp1=np.array(Ytemp1a)
        
        #print('after cp, Xtemp1 and its shape:')
        #print(Xtemp1.shape)
        
        ### if else
        if i==0:
            #print('this is for the loop i=0')            
            Xtrain=Xtemp1
            Ytrain=Ytemp1
        else:           
            ## horizontal: np.hstack; vertical:np.vstack            
            #print('in the loop, the shape of Xtemp:')
            #print(Xtemp1.shape)
            Xtrain=np.vstack((Xtrain,Xtemp1))
            Ytrain=np.hstack((Ytrain,Ytemp1))
            
            #print('in the loop, the shape of Xtrain:')
            #print(Xtrain.shape)
    
     ## test  420 smples  21 rounds
    for i in range(21):
        print('this is the test loop:')
        print(i)
        Xtemp2a,Ytemp2a=sess.run([featuretest,labeltest])          
        ### here we have to concate all the Xtemp2 into Xtest
        #print('Xtemp2a and its shape')
        #print(Xtemp2a)
        #print(Xtemp2a.shape)
        
        ### use conv5 features
        Xtemp2=cp(Xtemp2a,64,14,batch_size)
        
        Ytemp2=np.array(Ytemp2a)
        
        #print('Xtemp2 and its shape')
        #print(Xtemp2)
        #print(Xtemp2.shape)
        
         ### if else
        if i==0:
            #print('this is for loop i=0')
            
            Xtest=Xtemp2
            Ytest=Ytemp2
        else:           
            ## horizontal: np.hstack; vertical:np.vstack
            Xtest=np.vstack((Xtest,Xtemp2))
            Ytest=np.hstack((Ytest,Ytemp2))
    
    ### feed them into a SVM and train SVM
    ### the input and output formula of X and Y must be np.array
    
    Xtrain=np.array(Xtrain)
    Ytrain=np.array(Ytrain)
    
    print('Xtrain and its shape')
    #print(Xtrain)
    print(Xtrain.shape)
    print('Ytrain and its shape')
    print(Ytrain)
    print(Ytrain.shape)
    
    Xtest=np.array(Xtest)
    Ytest=np.array(Ytest)
    
    print('Xtest and its shape')
    #print(Xtemp2a)
    print(Xtest.shape)
    
    
    
    ### print X and Y
    np.save('save_train',Xtrain,Ytrain)
    np.save('save_test',Xtest,Ytest)
       
    ###remove nan
    train_where_are_nan = np.isnan(Xtrain)
    train_where_are_inf = np.isinf(Xtrain)
    #Xtrain[train_where_are_nan] = -0.01
    #Xtrain[train_where_are_inf] = -10
    # test oa=82.85
    #Xtrain[train_where_are_nan] = -0.1
    #Xtrain[train_where_are_inf] = -10
    # test oa=83.095
        
    Xtrain[train_where_are_nan] = -0.1
    Xtrain[train_where_are_inf] = -10
    
    test_where_are_nan = np.isnan(Xtest)
    test_where_are_inf = np.isinf(Xtest)
    #Xtest[test_where_are_nan] = -0.01
    #Xtest[test_where_are_inf] = -10
#    
    Xtest[test_where_are_nan] = -0.1
    Xtest[test_where_are_inf] = -10
    
    #### prepossessing L2
    #Xtrain=preprocessing.normalize(Xtrain,norm='l2')
    #Xtest=preprocessing.normalize(Xtest,norm='l2')
     
    print('start train svm')

    clf = svm.LinearSVC(C=1,multi_class='ovr')
    
    #clf=svm.SVC(C=1,kernel='rbf',max_iter=-1,decision_function_shape='ovr')
    #Xtrain = preprocessing.scale(Xtrain)  #normalization
    clf.fit(Xtrain, Ytrain)
    
    print('svm train accuracy:')
    print(clf.score(Xtrain,Ytrain))
    
    ## linearSVM
    #clf = svm.LinearSVC(C=1,multi_class='ovr',max_iter=1000)
    #clf.fit(Xtrain, Ytrain)
    
    ### input testsamples and output labels
    print('svm testing accuracy:')
    print(clf.score(Xtest,Ytest))
    
    print('the predicted label is:')
    print(clf.predict(Xtest))
    
    
    ### now here add some code to print all the labels into a txt
    
    coord.request_stop()
    coord.join(threads)

end = time.clock()
print("time is :")
print(end-start)

