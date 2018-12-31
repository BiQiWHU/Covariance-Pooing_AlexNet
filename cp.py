# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:48:54 2018
this python file is an implementation of CP (covariance pooling)

input:
    (1) a series of tensors from conv. layers, the format shape is: ndarray
    (2) parameter d:the number of divison
    (3) parameter s: after resize, the tensor's size of the 1st and the 2rd dimension

output:
    (1) a high dimension feature vector after covariance pooling, the format shape is:ndarray

please pay attention that there are about 5 stages in this algorithm
besides the reading of input and the final output
stage1:according the parameter d, divide this series of conv. tensors into d sequences
stage2:for each sequences from stage1, average them to get a x*x*1 tensor (a 2-d feature image)
stage3:stack these d "images" into a tensor X, with the size of s*s*d (s is the conv height size)
stage4:calculate the covariance matrix
stage5:get the feature vector

@author: 2009b_000
"""
### packages that we need

import tensorflow as tf
import numpy as np

### division, average, and resize stage1, stage2, stage3
def spilt_avr_resize(conv,d,s):
    
    #dimconv0,dimconv1,dimconv2,dimconv3=conv.get_shape().as_list()
    dimconv0=conv.shape[0]
    dimconv1=conv.shape[1]
    dimconv2=conv.shape[2]
    dimconv3=conv.shape[3]
    
    n=dimconv3//d

    conv=np.reshape(conv,(dimconv1,dimconv2,d,n))
    #conv=np.reduce_mean(conv,axis=3)
    conv=np.mean(conv,axis=3)
    print('after func, the shape is:')
    print(conv.shape)
    return conv

### calculate the covariance matrix and get the feature vector stage4, stage5 
def calc_featvec(x):
    ###stage4:calculate the covariance matrix
    #n0,n1,n2=x.get_shape().as_list()    
    n0=x.shape[0]
    n1=x.shape[1]
    n2=x.shape[2]
    
    x=np.reshape(x,[n0*n1,64])    
    # https://blog.csdn.net/whitesilence/article/details/75071780
    avr=np.mean(x,axis=0)
    avr=avr/n2

    element = np.zeros([n2,n2])
    #ones=np.zeros([n2,n2])
    v=np.zeros([n2*n2//2])
    
#    ones.assign(tf.ones_like(element))
#     # Upper triangular matrix, including the diagonal
#    masktemp = tf.matrix_band_part(ones, 0, -1)
#    # Make a bool mask
#    mask = tf.cast(masktemp, dtype=tf.bool) 
    
    ###stage5:get the feature vector
    temp=x-np.expand_dims(avr,axis=0)
    #temp=tf.subtract(x,np.expand_dims(avr,axis=0))       
    element=np.dot(np.transpose(temp),temp)
    
    element=element/(n2-1)
    element=np.log(element) 
    
    v=element[np.triu_indices(n2)]
    print(v)
    v=np.reshape(v,[1,-1])  
    return v


#### Now,let's run it
def cp(conv,d,s,batchsize):
    ## read the input and check the tensors
   
    #print('the conv shape input is:')
    #print(conv.shape)
    
    ### get the tensors in conv. layer/per batch one by one 
    sequence=np.split(conv,indices_or_sections=batchsize, axis=0)
    
    #print('check the input tensor size:')
    #print(sequence[0].shape)
    
    vbatch=[]
    
    for i in range(batchsize): 
        print(" i=%d in this batch"%i)              
        x=spilt_avr_resize(sequence[i],d,s)
        print('the shape and size of x is:')
        print(x.shape)
        print(type(x))
#        
        v=calc_featvec(x)
        print('the shape and type of v is:')
        print(v)
        print(v.shape)
        print(type(v))
       
        if i==0:
            vbatch=v
        else:           
            ## horizontal: np.hstack; vertical:np.vstack            
            vbatch=np.vstack((vbatch,v)) 
#    
    print('the shape and type of vbatch is:')
    print(vbatch)
    print(vbatch.shape)
    print(type(v))
#    
    return vbatch
            






### division, average, and resize stage1, stage2, stage3
def tfspilt_avr_resize(conv,d,s):
    
    dimconv0,dimconv1,dimconv2,dimconv3=conv.get_shape().as_list()
    n=dimconv3//d
#
#    ## creat a new zero tesnor, with a dimension of tf.shape[1]*tf.shape[1]*d
#    ## the size of input is dtype=tf.float32, convert it above
#    x = tf.Variable(tf.zeros([dimconv1,dimconv2,d]))
#    

#
#    # the size of n: tensor dimension
#    
#    ## note: tf.split(dimension, num_split, input) 
#    ## dimension: split at that dimension, num_split:how many sequences?
#    ## link1:https://www.cnblogs.com/hellcat/p/8581804.html
#    ## link2:https://blog.csdn.net/u012223913/article/details/79069373
#    ## link3:https://blog.csdn.net/drilistbox/rss/list  tf.unstack
#    sequence=tf.split(conv,num_or_size_splits=d, axis=3)
#    ## stage2: for each sequences from stage1, average them to get a x*x*1 tensor 
#    ##         (a 2-d feature image)
#
#    for i in range(d):
#        slide=tf.unstack(sequence[i],num=n,axis=3)
#        for j in range(n):
#            x[:,:,i].assign(tf.add(x[:,:,i],tf.squeeze(slide[j])))
#            #print('finish this loop!')        
#        x[:,:,i].assign(x[:,:,i]/n)

    conv=tf.reshape(conv,(dimconv1,dimconv2,d,n))
    conv=tf.reduce_mean(conv,axis=3)
    
    return conv



### calculate the covariance matrix and get the feature vector stage4, stage5 
def tfcalc_featvec(x):
    ###stage4:calculate the covariance matrix
    n0,n1,n2=x.get_shape().as_list()    
    x=tf.reshape(x,[n0*n1,64])    
    # https://blog.csdn.net/whitesilence/article/details/75071780
    avr=tf.reduce_sum(x,reduction_indices=0)
    #avr=np.sum(x,-1)
    avr=avr/n2

    element = tf.Variable(tf.zeros([n2,n2]))
    ones=tf.Variable(tf.zeros([n2,n2]))
    v=tf.Variable(tf.zeros([n2*n2//2]))
    
    ones.assign(tf.ones_like(element))
     # Upper triangular matrix, including the diagonal
    masktemp = tf.matrix_band_part(ones, 0, -1)
    # Make a bool mask
    mask = tf.cast(masktemp, dtype=tf.bool) 
    
    ###stage5:get the feature vector
    
    temp=tf.subtract(x,tf.expand_dims(avr,axis=0))       
    element=tf.matmul(tf.transpose(temp),temp)
    
    element=tf.div(element,(n2-1))
    #element.assign(tf.div(element,(n2-1)))
    #element=np.linalg.logm(element)
    element=tf.log(element)
    #element=tf.cholesky(element, name=None)
    ### we do not need cholesky decomposition   
    # V:upper_triangular_flat
    v.assign(tf.boolean_mask(element, mask))  
    v=tf.reshape(v,[1,-1])  
    return v


#### Now,let's run it
def tfcp(conv,d,s,batchsize):
    ## read the input and check the tensors
   
    #print('the conv shape input is:')
    #print(conv.shape)
    
    ### get the tensors in conv. layer/per batch one by one 
    sequence=tf.split(conv,num_or_size_splits=batchsize, axis=0)
    
    #print('check the input tensor size:')
    #print(sequence[0].shape)
    
    vbatch=[]
    
    for i in range(batchsize): 
        print(" i=%d in this batch"%i)              
        x=spilt_avr_resize(sequence[i],d,s)
        print('the shape and size of x is:')
        print(x.shape)
        print(type(x))
#        
        v=calc_featvec(x)
        print('the shape and type of v is:')
        print(v)
        print(v.shape)
        print(type(v))
#        

#        
        if i==0:
            vbatch=v
        else:           
            ## horizontal: np.hstack; vertical:np.vstack            
            vbatch=np.hstack((vbatch,v)) 
#    
    print('the shape and type of vbatch is:')
    print(vbatch)
    print(vbatch.shape)
    print(type(v))
#    
    return vbatch
            

