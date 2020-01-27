# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:01:33 2019
@author: Fede
"""
import tensorflow as tf
from keras.models import Input,Model,Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,BatchNormalization
from keras.layers import *
from keras.models import load_model
from keras.layers import Activation
from keras import initializers
from keras import optimizers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import keras
from keras_radam import RAdam
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras import backend as K
num_cores = 4
GPU=False
CPU=True
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

session = tf.Session(config=config)
K.set_session(session)

def create_step_model(world_size:int,n_actions:int,learning_rate:int,neurons_fully:int,drop_rate:float,dueling=bool):

    x = Input(shape=(3,world_size, world_size))
    actions_input = Input((n_actions,),name='mask')
    
    x2 = (Conv2D(20,kernel_size=3,data_format='channels_first',padding='same',activation='relu',strides=(1,1)))(x)
    x3 = (Conv2D(20,kernel_size=3,data_format='channels_first',padding='valid',activation='relu',strides=(1,1)))(x2)
    x4 = (Conv2D(20,kernel_size=3,data_format='channels_first',padding='valid',activation='relu',strides=(1,1)))(x3)
    
    out = Flatten()(x4)
        
    if dueling==True:
        F1 = (Dense(10,activation='relu'))(out)
        F2 = (Dense(50,activation='relu'))(out)
        y4 = (Dense(1))(F1)
        y5 = (Dense(n_actions))(F2)
        y8 = keras.layers.Reshape((n_actions,1))(y5)
        y9 = keras.layers.AveragePooling1D(pool_size=n_actions)(y8)    
        output=keras.layers.Add()([y5,y4])
        output1 = keras.layers.Subtract()([output,y9])
    else:
        output1 = (Dense(5))(out)
    
    actions_input2 = keras.layers.Reshape((1,n_actions))(actions_input)
    filtered_output = keras.layers.Multiply()([actions_input2,output1])
    filtered_output = keras.layers.Reshape((n_actions,1))(filtered_output)
    
    step_model=Model(inputs=[x,actions_input],outputs=filtered_output)
    if learning_rate<1:
        opt=optimizers.SGD(lr=learning_rate, decay=0,momentum=0.9, nesterov=True)
    else:
        opt=RAdam()

    step_model.compile(optimizer=opt, loss='mean_squared_error')
    return step_model

def get_huber_loss_fn(**huber_loss_kwargs):

    def custom_huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)

    return custom_huber_loss 
  
def create_step_model2(world_size:int,n_actions:int,learning_rate:int,neurons_fully:int):

    x = Input(shape=(world_size, world_size, 3))
    actions_input = Input((n_actions,),name='mask')
    convs = []
    
    x1 = (Conv2D(15,kernel_size=3,padding='same',activation='relu',strides=(1,1)))(x)
    x2 = (Conv2D(10,kernel_size=2,padding='valid',activation='relu',strides=(1,1)))(x)
    x3 = (Conv2D(15,kernel_size=3,padding='valid',activation='relu',strides=(1,1)))(x)
    x4 = (Conv2D(20,kernel_size=4,padding='valid',activation='relu',strides=(1,1)))(x)
    
    out1 = Flatten()(x1)
    out2 = Flatten()(x2)
    out3 = Flatten()(x3)
    out4 = Flatten()(x4)
    
    convs.append(out1)
    convs.append(out2)
    convs.append(out3)
    convs.append(out4)
    out = Concatenate()(convs)
    
    if neurons_fully>1:
        out = (Dense(neurons_fully,activation='relu'))(out)
    
    y4 = (Dense(1,activation='relu',activity_regularizer=regularizers.l2(0.05)))(out)
    y5 = (Dense(5,activation='relu',activity_regularizer=regularizers.l2(0.05)))(out)
    y8 = keras.layers.Reshape((5,1))(y5)
    y9 = keras.layers.AveragePooling1D(pool_size=5)(y8)
    
    output=keras.layers.Add()([y5,y4])
    output1 = keras.layers.Subtract()([output,y9])
    actions_input2 = keras.layers.Reshape((1,5))(actions_input)

    filtered_output = keras.layers.Multiply()([actions_input2,output1])
    filtered_output = keras.layers.Reshape((5,1))(filtered_output)
    
    step_model=Model(inputs=[x,actions_input],outputs=filtered_output)
    if learning_rate<1:
        opt=optimizers.SGD(lr=learning_rate, decay=0,momentum=0.9, nesterov=True)
    else:
        opt=RAdam()
    step_model.compile(optimizer=opt, loss='mean_squared_error')
    return step_model

def create_mem_replay_with_loss():
    mem_replay_with_loss = {}
    return mem_replay_with_loss

def fill_mem_replay_with_loss(mem_replay_with_loss,fit_input,fit_output,
                              actions,target_model,counter):
    i=0
    for state in fit_input:
        act = np.ndarray((1,25))
        for j in range(25):
            if j==actions[i][j]:
                act[0,j]=1
            else:
                act[0,j]=0
        state1 = state.reshape(1,6,6,3)
        loss = np.sum(abs(fit_output[i]-target_model.predict([np.asarray(state1),act])))
        mem_replay_with_loss[counter]=loss
        i=i+1
        counter+=1
    return mem_replay_with_loss
        
def plot_mem_replay_with_loss_situation(mem_replay_with_loss):
    situation = []
    for i in mem_replay_with_loss:
        situation.append(mem_replay_with_loss[i])
    
    plt.plot(situation)
    plt.title('Loss of the memory replay with loss')
    plt.ylabel('Loss')
    plt.xlabel('States in the memory replay with loss')
    plt.legend(['Loss'], loc='upper left')
    plt.show()

