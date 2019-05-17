# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:31:02 2019

@author: falcon1
"""

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Concatenate, Bidirectional, LSTM
from kears.layers import Dropout
sess = tf.Session()
K.set_session(sess)

HIGH = 21
WIDTH = 1000

# input layer
#inputs = tf.placeholder(tf.float32, shape=(None, HIGH*WIDTH))
inputs = Input(shape=(HIGH, WIDTH,1))

# first convolutionary layers
x1_1 = Conv2D(20,(21,1), padding="same")(inputs)
x1_2 = Conv2D(20,(20,3), padding="same")(inputs)
x1_3 = Conv2D(20,(20,5), padding="same")(inputs)
x1_4 = Conv2D(20,(20,9), padding="same")(inputs)
x1_5 = Conv2D(20,(20,15), padding="same")(inputs)
x1_6 = Conv2D(20,(20,21), padding="same")(inputs)

# Concatenate all CNN layers
x1 = Concatenate()([x1_1, x1_2, x1_3, x1_4, x1_5, x1_6])

# second CNN layer
x2 = Conv2D(128,(120,3), padding="same")(x1)

# Dropout
l_drop = Dropout(rate=0.5, seed=41)(x2)

# LSTM forward and backward layer
x3 = LSTM(256)(x2)

x4 = Bidirectional()(x3)

x5 = 


