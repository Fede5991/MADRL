# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 08:47:52 2019
@author: Fede
"""

"""
Created on Feb 06 2019
@author: fedmason
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

from NN import create_step_model,create_step_model2
from NN import create_mem_replay_with_loss
from NN import fill_mem_replay_with_loss,plot_mem_replay_with_loss_situation

from update_MRS import update_MRS

import keras
from keras.callbacks import EarlyStopping
from keras.models import Input,Model,Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,BatchNormalization
from keras.layers import *
from keras.models import load_model
from keras import initializers
from keras import optimizers
from keras.utils.vis_utils import plot_model

import keras.backend as K
dtype='float32'
K.set_floatx(dtype)
# default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
K.set_epsilon(1e-4)

from gym_drones.classes.map import WorldMap
from gym_drones.envs.drones_env import DronesDiscreteEnv


def testing_network(world_size: [int],
                    target_num: int,
                    drone_goal: str,
                    drone_num: int,
                    extra_drone_num: int,
                    world_gain_peak_range: [float],
                    world_gain_var_range: [float],
                    world_evolution_speed: [float],

                    drone_comm: float,
                    drone_view: float,
                    drone_memory: int,
                    drone_battery: float,
                    action_step: int,
                    max_age: int,
                    lookahead_step: int,
                    malus: float,
                    final_bonus: float,
                    malus_sm:float,

                    random_episode: bool,

                    alpha: float,
                    alpha_dec: float,
                    epsilon: float,
                    epsilon_dec: float,
                    temperature: float,
                    temperature_dec: float,

                    state_MR: int,
                    limit_MR: bool,
                    three_MR: bool,
                    prioritized: bool,
                    perc_pos_MR: float,
                    perc_neg_MR1: float,
                    perc_neg_MR2: float,
                    pretrain_episode: int,
                    train_episode: int,
                    test_episode: int,
                    step_num: int,

                    testing_update: int,
                    model_update: int,
                    batch_update: int,
                    batch_size: int,
                    learning_rate: float,
                    neurons_fully:int,
                    drop_rate:float,
                    dueling:bool,

                    epochs_num: int,
                    gamma: float,
                    steps_per_epoch:int,
                    verbose: bool,
                    version:int):

    random.seed(100)

    initializers.Ones()

    env: DronesDiscreteEnv = gym.make('DronesDiscrete-v0')

    #  Generate a world-map, where a groups of target is moving

    world = WorldMap(world_size,
                     target_num,
                     world_gain_peak_range,
                     world_gain_var_range,
                     world_evolution_speed)

    # Define the log file

    output_log = "Log_files/"

    # Initialize relay memory

    memory_replay = []
    mem2=[]
    prob_MR=[]

    pos_MR=[]
    neg_MR1=[]
    neg_MR2=[]
    mem_replay_with_loss=create_mem_replay_with_loss()
    # Initialize the success data vector

    success_vec = []
    success_vec1 = []
    success_episodes=[]
    Ac0=[]
    Ac1=[]
    Ac2=[]
    Ac3=[]
    Ac4=[]
    A_0=0
    A_1=0
    A_2=0
    A_3=0
    A_4=0
    SMR=[0,0,0,0,0]
    GSMR=[]
    AM=[0,0,0]
    AAM=[]
    VARIATIONS=[]
    VAR=[]
    VARQ=0
    MINQ=[]
    MINQV=0
    MAXQ=[]
    MAXQV=0
    MEANQ=[]
    MEANQV=0
    Mappa=np.zeros((world_size[0],world_size[0]))
    LOSSES=np.zeros((4,5000))
    # Setting neural network

    step_model=create_step_model(world_size[0],5,learning_rate,neurons_fully,drop_rate,dueling)
    #step_model.load_weights('target_model23_v0.h5')
    #print (step_model.summary())
    
    target_model=keras.models.clone_model(step_model)
    target_model.set_weights(step_model.get_weights())
    #print (target_model.summary())
    
    plot_model(step_model, to_file='model_plot.pdf', show_shapes=True, show_layer_names=True)

    # Setting early stopping and history

    #early = EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')
    history = LossHistory()
    H=[]
    # ------------------------ PRE-PROCESSING PHASE  ------------------------

    # Pre-training phase:
    # Use a random policy to choose actions
    # Save the sample [state, action, reward, new_state] in the replay memory
    # No training is carried out
    if prioritized==True:
        update_MR=True
    else:
        update_MR=False
    if state_MR==0:
        pretraining_done = True
    else:
        pretraining_done = False
    if pretraining_done==True:

        #print("INITIALIZATION MEMORY-REPLAY...")

        for episode_index in tqdm(range(pretrain_episode)):
        
            counter=0
            #print("Episode index: ", episode_index)
    
            # Generate a random episode
    
            if random_episode:
    
                # Generate new map
    
                world = WorldMap(world_size,target_num,world_gain_peak_range,world_gain_var_range,world_evolution_speed);
            #print ("0")
            log_name = output_log + "env_pretrain_" + str(episode_index) + ".txt"
            log_file = open(log_name, "w+")
    
            # Configure new environment
            train=True
            env.configure_environment(world,drone_goal,drone_num,extra_drone_num,drone_comm,
                                      drone_view,drone_memory,drone_battery,action_step,max_age,
                                      lookahead_step,malus,final_bonus,log_file,verbose,train,malus_sm);
            #print ("1")
            # Get the initial state of the system
            # If needed, normalize the state as you desire
    
            state = env.get_state();
            z=0
            #print ("2")
            for step in range(step_num):
                for j in range(drone_num):
                    own_map=state[j]
                    others_map=np.zeros((world_size[0],world_size[0]))
                    for i in range(drone_num):
                        if i!=j:
                            others_map+=state[i]
                    #print ("3")
                    Mappa[int(np.argmax(state[0])/world_size[0]),np.argmax(state[0])%world_size[0]]+=1
                    Mappa[int(np.argmax(state[1])/world_size[0]),np.argmax(state[1])%world_size[0]]+=1
                    
                    number=env.get_available_targets()
                    
                    if number==0:
                        AM[0]+=1
                    elif number==1:
                        AM[1]+=1
                    else:
                        AM[2]+=1

                    model_input_state=state[[0,1,drone_num+1],:]
                    model_input_state[0]=own_map
                    model_input_state[1]=others_map
                    action = env.get_random_direction();  # Random action           
                    for i in range(drone_num):
                        if i!=j:
                            action[i]=0  
                    #print (action)
                    env.action_direction(action);  # Carry out a new action 
                    new_state = env.get_state();  # New system state
                    #print ("4")
                    model_input_newstate=new_state[[0,1,drone_num+1],:]
                    model_input_newstate[0]=new_state[j]
                    model_input_newstate[1]=others_map
                    explore_reward, exploit_reward = env.get_reward()  # Obtained reward (exploit + explore rewards)
                    reward = exploit_reward[j] # Overall reward (only exploit)
                    
                    if reward==-1:
                        SMR[0]+=1
                    if reward==-0.4:
                        SMR[1]+=1
                    if reward==1 and np.mean(exploit_reward)<1:
                        SMR[3]+=1
                    if np.mean(exploit_reward)==1:
                        SMR[4]+=1
                    if exploit_reward[0]>-0.4 and exploit_reward[0]<1 and exploit_reward[1]>-0.4 and exploit_reward[1]<1:
                        SMR[2]+=1
                    
                    sample = [model_input_state, [int(action[j])], model_input_newstate, reward]  # Sample to be saved in the memory
                    memory_replay.append(sample)
                    prob_MR.append([1,0,5])
                    state=new_state
                
            #print (counter)
            log_file.close()

        
        np.save('memory_replay',memory_replay)
        np.save('prob_MR',prob_MR)

    else:
        #print ("LOADING MEMORY REPLAY...")
        memory_replay = np.load('memory_replay.npy',allow_pickle=True)
        memory_replay=memory_replay.tolist()
        prob_MR=np.load('prob_MR.npy',allow_pickle=True)
        prob_MR=prob_MR.tolist()
    # Training phase
    # Make actions according to a epsilon greedy or softmax policy
    # Periodically train the neural network with batch taken from the replay memory

    #print("TRAINING PHASE...")
    AVG=[]
    MAX=[]

    def epsilon_func(x):
        e=(1-1/(1+np.exp(-x/100)))*0.8+0.2
        if x>400:
            e = e*((499-x)/100)
        return e
    
    e = []
    for i in range(1000):
        e.append(epsilon_func(i-500))
    

    st = []
    for ep in range(1000):
        st.append(5000/(ep+1)**(1/3))

    batch_size = 50

    #print ("TRAINING PHASE")
    # ------------------------ PROCESSING PHASE  ------------------------
    
    negative_r1=[]
    pos_r=[]
    pos_r2=[]
    negative_r2=[]
    null_r=[]
    counter1=0
    counter2=0
    counter3=0
    counter4=0
    counter5=0
    COUNTS=[0,0,0,0]
    VAR_COUNTS=[0,0,0,0] 
    for episode_index in tqdm(range(train_episode)):
        CMR0=0
        CMR1=0
        CMR2=0
        CMR3=0
        if episode_index%5==0:
            COUNTS=[0,0,0,0]
            VAR_COUNTS=[0,0,0,0]        
        AAM.append(AM)
        AM=[0,0,0]
        mem_replay_with_loss=create_mem_replay_with_loss()
        fit_input_temp=[]
        fit_output_temp=[]
        fit_actions_temp=[]
        counter_MR=0
        epsilon = epsilon_func(episode_index-500)
        avg_avg_loss=0
        avg_max_loss=0
        iter_avg=0
        worst_states=[]
        GSMR.append(SMR)
        SMR=[0,0,0,0]

        #print("\n Epsilon: ",epsilon)
        # ------------------------ TRAINING PHASE  ------------------------

        #print("Training episode ", episode_index, " with epsilon ", epsilon)

        # Generate a random episode

        if random_episode:

            # Generate new map

            world = WorldMap(world_size,
                             target_num,
                             world_gain_peak_range,
                             world_gain_var_range,
                             world_evolution_speed);

        log_name = output_log + "env_train_" + str(episode_index) + ".txt"
        log_file = open(log_name, "w")

        # Configure new environment
        train=True
        env.configure_environment(world,drone_goal,drone_num,extra_drone_num,drone_comm,
                                  drone_view,drone_memory,drone_battery,action_step,max_age,
                                  lookahead_step,malus,final_bonus,log_file,verbose,train,malus_sm);

        # Get the initial state of the system
        # If needed, normalize the state as you desire

        state = env.get_state();

        for step in range(step_num):
            for j in range(drone_num):
                model_input = state  # The input might be different than the environment state
                model_input = np.asarray(model_input)                        
                number=env.get_available_targets()
                if number==0:
                    AM[0]+=1
                elif number==1:
                    AM[1]+=1
                else:
                    AM[2]+=1
       
                others_map=np.zeros((world_size[0],world_size[0]))
                for i in range(drone_num):
                    if i!=j:
                        others_map+=state[i]
                model_input = model_input[[0,1,drone_num+1],:]
                model_input[0]=state[j]
                model_input[1]=others_map
                model_input_state=copy.deepcopy(model_input)
                #model_input=np.asarray(model_input)
                model_input=model_input.reshape(1,3,world_size[0],world_size[0])
                #print (model_input)
                action = np.ndarray((1,5))
                for i in range(5):
                    action[0,i]=1
    
                greedy_action = np.zeros(drone_num)
                greedy_action[j]=np.argmax(target_model.predict([model_input,action]))# Greedy action
                    
                random_action = env.get_random_direction();  # Random action
                for i in range(drone_num):
                    if i!=j:
                        random_action[i]=0
                action_type=0
                if np.random.uniform(0, 1) < epsilon:
                    action1 = random_action
                    action=[]
                    for i in range(drone_num):
                        action.append(int(action1[i]))
                else:
                    action = greedy_action
                    action_type=1
                env.action_direction(action);  # Carry out a new action
                new_state = env.get_state();  # New system state
                explore_reward, exploit_reward = env.get_reward();  # Obtained reward (exploit + explore rewards)
                reward = exploit_reward[j]  # Overall reward (only exploit)            
                model_input_new_state=copy.deepcopy(model_input_state)
                model_input_new_state[0]=new_state[j]
                model_input_new_state[drone_num]=new_state[drone_num+1]
                sample = [model_input_state, [int(action[j])], model_input_new_state, reward]  # Sample to be saved in the memory
                memory_replay.append(sample)
        
                prob_MR.append([1,0,5])
                state=new_state
    
                
            if (step+1) % batch_update == 0:  # Each "batch_update" steps, train the "step_model" NN

                # Choose a set of samples from the memory and insert them in the "samples" list:
                probability=np.random.rand(1)
                type_training=0
                if probability<1.1 or len(mem2)<batch_size:
                    samples_indexes = random.sample(list(range(len(memory_replay))), batch_size)
                    for i in range(len(samples_indexes)):
                        mem2.append(memory_replay[samples_indexes[i]])
                    samples = []
                    for s_index in samples_indexes:
                        samples.append(memory_replay[s_index])
                else:
                    choices=np.arange(len(memory_replay))
                    probabilities=[]
                    somma=0
                    for i in range(len(prob_MR)):
                        probabilities.append(prob_MR[i][0])
                        somma+=prob_MR[i][0]
                    probabilities=probabilities/somma
                    samples_indexes=np.random.choice(choices,batch_size,p=probabilities)
                    type_training=1
                    samples = []
                    for s_index in samples_indexes:
                        samples.append(memory_replay[s_index])


                # Deep Q-learning approach

                fit_input = []  # Input batch of the model
                fit_output = []  # Desired output batch for the input
                fit_actions = []
                fit_actions_predictions=[]
                
                for sample in samples:
                    sample_state = sample[0]  # Previous state
                    sample_action = sample[1] # Action made
                    sample_new_state = sample[2]  # Arrival state
                    sample_reward = sample[3]  # Obtained reward
                    sample_new_state=sample_new_state.reshape(1,3,world_size[0],world_size[0])
                    
                    action = np.ndarray((1,5))
                    for i in range(5):
                        action[0,i]=1
                    
                    sample_goal = sample_reward + gamma * np.max(target_model.predict([sample_new_state,action]))
                    sample_state=np.asarray(sample_state)
                    sample_state=sample_state.reshape(1,3,world_size[0],world_size[0])

                    act = np.ndarray((1,5))
                    for i in range(5):
                        if i==sample_action[0]:
                            act[0,i]=1
                        else:
                            act[0,i]=0
                            
                    sample_output = step_model.predict([np.asarray(sample_state),action])[0]
                    #print (sample_action)
                    for i in range(5):
                        if i==sample_action[0]:
                            sample_output[i,0] = (1 - alpha) * sample_output[sample_action] + alpha * sample_goal
                        else:
                            sample_output[i,0]=0
                                      
                    #print (sample_state[0])
                    #print (sample_output)
                    #print (act[0])
                    fit_input.append(sample_state[0])  # Input of the model
                    fit_input_temp.append(sample_state[0])
                    fit_output.append(sample_output)  # Output of the model
                    fit_output_temp.append(sample_output)
                    fit_actions.append(act[0])
                    fit_actions_temp.append(act[0])
                    fit_actions_predictions.append(action[0])
                    
                # Fit the model with the given batch
                step_model.fit([np.asarray(fit_input),np.asarray(fit_actions)],
                               np.asarray(fit_output),
                               batch_size=None,
                               epochs=epochs_num,
                               steps_per_epoch=steps_per_epoch,
                               callbacks=[history],
                               verbose=0)
                mean_loss=np.mean(history.losses)
                #LOSSES[MR_type,episode_index]+=mean_loss                
                H.append(history.losses)
                output=step_model.predict([np.asarray(fit_input),np.asarray(fit_actions)])
                total_output=step_model.predict([np.asarray(fit_input),np.asarray(fit_actions_predictions)])

                loss=[]
                for i in range(batch_size):
                    loss.append((output[i][np.argmax(np.asarray(fit_actions[i]))]-np.asarray(fit_output[i][np.argmax(np.asarray(fit_actions[i]))]))**2)
                for i in range(batch_size):
                    prob_MR[samples_indexes[i]][0]=loss[i][0]
                    if prob_MR[samples_indexes[i]][2]!=5:
                        if memory_replay[samples_indexes[i]][3]==-1:
                            COUNTS[0]+=1
                            if np.argmax(total_output[i])!=prob_MR[samples_indexes[i]][2]:
                                VAR_COUNTS[0]+=1
                                prob_MR[samples_indexes[i]][2]=np.argmax(total_output[i])
                        elif memory_replay[samples_indexes[i]][3]==-0.4:
                            COUNTS[1]+=1
                            if np.argmax(total_output[i])!=prob_MR[samples_indexes[i]][2]:
                                VAR_COUNTS[1]+=1
                                prob_MR[samples_indexes[i]][2]=np.argmax(total_output[i])
                        elif memory_replay[samples_indexes[i]][3]==1:
                            COUNTS[3]+=1
                            if np.argmax(total_output[i])!=prob_MR[samples_indexes[i]][2]:
                                VAR_COUNTS[3]+=1
                                prob_MR[samples_indexes[i]][2]=np.argmax(total_output[i])
                        else:
                            COUNTS[2]+=1
                            if np.argmax(total_output[i])!=prob_MR[samples_indexes[i]][2]:
                                VAR_COUNTS[2]+=1
                                prob_MR[samples_indexes[i]][2]=np.argmax(total_output[i])
                    else:
                        prob_MR[samples_indexes[i]][2]=np.argmax(total_output[i])    
                        
                            
                counter_MR+=batch_size
                      
            if (step+1) % model_update == 0:  # Each "model_update" steps, substitute the target_model with the step_model
                
                target_model.set_weights(step_model.get_weights())

                            
        log_file.close()

        # Testing phase
        # Make actions ONLY according to the NN model output (temperature is 0)
        # No training is carried out
        # The results are compared with the lookahead policy implemented in the environment

        # ------------------------ TESTING PHASE  ------------------------
        
        #if episode_index%5==0 and episode_index>0:
            #VARIATIONS.append([VAR_COUNTS[0]/COUNTS[0],VAR_COUNTS[1]/COUNTS[1],VAR_COUNTS[2]/COUNTS[2],VAR_COUNTS[3]/COUNTS[3]])
        
        if episode_index % testing_update == 0:

            #print("TESTING PHASE...")
            if  episode_index>0:
                    Ac0.append(A_0/100)
                    Ac1.append(A_1/100)
                    Ac2.append(A_2/100)
                    Ac3.append(A_3/100)
                    Ac4.append(A_4/100)
                    A_0=0
                    A_1=0
                    A_2=0
                    A_3=0
                    A_4=0
            success_vec.append([])
            success_vec1.append([])
            mean_reward=0
            PERCENTS=0
            for test_index in range(test_episode):
                REWARD=0

    
                # Generate a random episode

                if random_episode:

                    # Generate new map

                    world = WorldMap(world_size,
                                     target_num,
                                     world_gain_peak_range,
                                     world_gain_var_range,
                                     world_evolution_speed);

                log_name = output_log + "env_test_" + str(episode_index) + ".txt"
                log_file = open(log_name, "w")

                # Configure new environment

                train=True
                env.configure_environment(world,drone_goal,drone_num,extra_drone_num,drone_comm,
                                      drone_view,drone_memory,drone_battery,action_step,max_age,
                                      lookahead_step,malus,final_bonus,log_file,verbose,train,malus_sm);


                # Get the initial state of the system
                # If needed, normalize the state as you desire

                state = env.get_state()                
                for step in range(40):

                    # env.render()
                    # Choose always the greedy action using the target_model
                    for j in range(drone_num):
                        REWARD1=0
                        actions = np.ndarray((1,5))
                        for i in range(5):
                            actions[0,i]=1
                            
                        model_input = state
                        own_map=state[j]
                        model_input = np.asarray(model_input)
                        model_input=model_input[[0,1,drone_num+1],:]
                        others_map = np.zeros((world_size[0],world_size[0]))
                        for i in range(drone_num):
                            if i!=j:
                                others_map+=state[i]
                        model_input[1]=others_map
                        model_input[0]=own_map
                        model_input=model_input.reshape(1,3,world_size[0],world_size[0])
                        action = np.zeros(drone_num)
                        action[j] = np.argmax(target_model.predict([model_input,actions]))
                        env.action_direction(action);  # Carry out a new action
                        
                        if action[0]==0:
                            A_0+=1
                        elif action[0]==1:
                            A_1+=1
                        elif action[0]==2:
                            A_2+=1
                        elif action[0]==3:
                            A_3+=1
                        else:
                            A_4+=1
                        
                        new_state = env.get_state();  # New system state
                        explore_reward, exploit_reward = env.get_reward()
                        reward = np.mean(exploit_reward)
                    
                        if np.mean(exploit_reward)==1:
                            counter2+=drone_num
                        for i in range(drone_num):
                            if exploit_reward[i]==-malus_sm:
                                counter1+=1
                        for i in range(drone_num):
                            if exploit_reward[i]==-1:
                                counter3+=1
                        for i in range(drone_num):
                            if exploit_reward[i]==1 and np.mean(exploit_reward)<1:
                                counter4+=1
                        for i in range(drone_num):
                            if exploit_reward[i]<1 and exploit_reward[i]>0:
                                counter5+=1
                    
                        REWARD+=reward
                        REWARD1+=reward
                
                        state = new_state  # Substitute the previous state with the new state
                    
                #print (REWARD/30)
                success_vec[-1].append(REWARD/80)
                if REWARD1==1:
                    success_vec1[-1].append(1)
                    PERCENTS+=1
                else:
                    success_vec1[-1].append(0)
                mean_reward+=REWARD/80
                log_file.close()
                
            #print ("\n Success rate: \n")
            print (mean_reward/100)
            #print (PERCENTS)
            negative_r1.append(counter1)
            pos_r.append(counter2)
            negative_r2.append(counter3)
            pos_r2.append(counter4)
            null_r.append(counter5)
            success_episodes.append(PERCENTS)
            print (PERCENTS)
            counter1=0
            counter2=0
            counter3=0
            counter4=0
            counter5=0
            #print("\n Mean success ratio: ", np.mean(success_vec[-1]))

        # Decrease system temperatures

    log_name = output_log + "example" + str(episode_index) + ".txt"
    log_file = open(log_name, "w")
                                            
    env.close()
    log_file.close()
    target_model.save('target_model'+str(drone_num)+str(target_num)+'_v'+str(version)+'.h5')
    del step_model
    del target_model
    return success_vec,success_vec1,H,negative_r1,pos_r,negative_r2,pos_r2,null_r,Ac0,Ac1,Ac2,Ac3,Ac4,SMR,GSMR,AM,AAM,Mappa,VAR,MINQ,MAXQ,MEANQ,LOSSES,success_episodes,VARIATIONS                          

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))