# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:19:55 2018

in this code, we simply return the conv. features of different layers 
for a more detailed feature representation

@author: 2009b_000
"""


# coding: utf-8

# In[1]:


from my_ops import *


# In[2]:


def alex_net(img_batch, train=True):
    # Layer 1 (conv-relu-pool-lrn)
    conv1 = conv(img_batch, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
    norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

    # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
    conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
    norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
    if train:
        dropout6 = dropout(fc6, 0.5)
    else:
        dropout6 = fc6

    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, 4096, 4096, name='fc7')
    if train:
        dropout7 = dropout(fc7, 0.5)
    else:
        dropout7 = fc7

    # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    fc8 = fc(dropout7, 4096, 21, relu=False, name='fc8')

    return conv1,conv2,conv3,conv4,conv5,fc7

