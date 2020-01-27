# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:32:59 2019

@author: Fede
"""

import gym
import gym_drones
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import copy

from NN import create_step_model
from NN import create_mem_replay_with_loss
from NN import fill_mem_replay_with_loss,plot_mem_replay_with_loss_situation

import keras
from keras.callbacks import EarlyStopping
from keras.models import Input,Model,Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,BatchNormalization
from keras.layers import *
from keras.models import load_model
from keras import initializers
from keras import optimizers
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras import backend as K

from gym_drones.classes.map import WorldMap
from gym_drones.envs.drones_env import DronesDiscreteEnv

def update_MRS(pos_MR,memory_replay,neg_MR1,neg_MR2,step_model,world_size):
    MR = [pos_MR,memory_replay,neg_MR1,neg_MR2]
    for v in MR:            
        samples = []
        for s_index in range(len(v)):
            samples.append(v[s_index])
               
        index=0
        for sample in samples:
                    sample_state = sample[0]  # Previous state                   
                    sample_action = sample[1]
                    action = np.ndarray((1,5))
                    for i in range(5):
                        action[0,i]=1
                    
                    sample_state=np.asarray(sample_state)
                    sample_state=sample_state[[0,1,3],:]
                    sample_state=sample_state.reshape(1,world_size[0],world_size[0],3)

                    sample_output = step_model.predict([np.asarray(sample_state),action])[0]

                    if np.argmax(sample_output)==sample_action:
                        if sample_action!=v[index][4]:
                            v[index][5]=3
                            v[index][4]=sample_action
                    else:
                        v[index][5]=1
                    index+=1
    return pos_MR,memory_replay,neg_MR1,neg_MR2