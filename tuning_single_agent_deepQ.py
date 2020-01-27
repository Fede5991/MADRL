from single_agent_deepQ1 import testing_network
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plots import plot_results,plot_memory_replay,plot_qvalues

from IAHOS import IAHOS
from extraction_performances import extraction_performances
from hyperparams_initialization import hyperparams_initialization
from update_MRS import update_MRS
from plots import plot_IAHOS,plot_pretrain,plot_training

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


cycol = cycle('bgrcmk')


attempts = 4

world_size = [[6,6],[10,10],[7,7],[8,8]]  # World dimensions along the two axes
drone_num = [2,2,3]  # Number of drones
target_num = [2,3,3]  # Number of targets

world_gain_peak_range = [[5,5],[5,5],[10, 10],[10,10]]  # Maximum peak of a target
world_gain_var_range = [[1, 1],[2,2],[6,6],[4,4]]  # Maximum extension of a target
world_evolution_speed = [0, 0, 0, 0]  # Speed by which target move in the world

drone_goal = 'exploit'
extra_drone_num = 0
drone_comm = 0.5  # Range by which drones can communicate (only MultiAgent)
drone_view = 1  # Range bu which drones can see the map

drone_memory = 0  # Memory of a drone
drone_battery = 0  # Battery of a drone

action_step = 1  # Number of step for a given agent action
max_age = 10  # Max age of a map location
lookahead_step = 2  # Number of step for the lookahead
malus = 1  # Malus suffered by drones going out from the map
final_bonus = 1  # Bonus received when reaching the optimal position
malus_sm=[0.8,0.3,0.5]
reward_type=[0,1]

verbose = False  # Set True to print all the data regarding the environment evolution
random_episode = True  # Set True to make the agent learn to adapt to different scenarios

limit_MR=[False,False]
three_MR = [False,True,True]
prioritized=[False,False,True]
perc_pos_MR = [0.2,0.3,0.5]
perc_neg_MR1 = 0.1
perc_neg_MR2 = 0.1
pretrain_episode = 50 # Number of pre-training episodes
train_episode =  1001# Total number o training episodes
test_episode = 100 # Number of testing episodes
step_num = [200,250]  # Number of actions in a given episode

epsilon = .8  # Greedy policy parameter
temperature = 0.03   # Softmax policy parameter
epsilon_dec = 0
temperature_dec=0.01

# Neural network settings

epochs_num = 1
steps_per_epoch=1
batch_size = 20
learning_rate_vector = [0.01,0.1,1]
neurons_fully=[0,100,200,300]
drop_rate=[0.2,0.2,0.5]
dueling='False'
# RL settings

alpha = 1  # Learning parameter
alpha_dec = 1  # Learning parameter decreasing
gamma = [0.6,0.7,0.8]  # Discount rate
batch_update = 1# Interval between batch trainings
model_update = 25  # Interval between NN target model updates
testing_update = 100  # Interval between testing phases

# Output vector

vector_mean = []
vector_95percent = []
vector_75percent = []
vector_25percent = []
vector_5percent = []
loss_mean = []

titles=['gamma=0.6','gamma=0.7','gamma=0.8']
labels=['Mean average','Min']

vector_mean=[]
vector_mean1=[]

pre_training_RL_needed=False
pre_training_NN_needed=False

if pre_training_RL_needed==True:
    attempts=3
    variables=2
    iterations=attempts**variables
    limits = [[0.6,0.9],[0.1,1]]
    method='grid'
    rounds=2
    tgp,tgp2,ogp,ogp2,final,performances,params=IAHOS(rounds,method,limits,attempts,variables,iterations,
                                  testing_network,world_size[1],target_num,drone_goal,
                                  drone_num,extra_drone_num,world_gain_peak_range,world_gain_var_range[0],
                                  world_evolution_speed,drone_comm,drone_view,drone_memory,drone_battery,
                                  action_step,max_age,lookahead_step,malus,final_bonus,malus_sm[0],random_episode,
                                  alpha,alpha_dec,epsilon,epsilon_dec,temperature,temperature_dec,limit_MR[0],
                                  three_MR[0],prioritized[0],perc_pos_MR[0],perc_neg_MR1,perc_neg_MR2,pretrain_episode,
                                  train_episode,test_episode,step_num[0],testing_update,model_update,
                                  batch_update,batch_size,learning_rate_vector[2],neurons_fully[0],epochs_num,
                                  gamma[0],steps_per_epoch,verbose)

    model='Gamma_MALUS'
    y = ['Gamma','Malus']
    name='RL'
    plot_IAHOS(y,ogp,ogp2,tgp,tgp2,model)
    plot_pretrain(performances,tgp,params,name)

if pre_training_NN_needed:
    optimal_values=np.load('final.npy')        
    print (optimal_values)
    tgp,tgp2,ogp,ogp2,final,performances,params=IAHOS(rounds,method,limits,attempts,variables,iterations,
                                  testing_network,world_size[1],target_num,drone_goal,
                                  drone_num,extra_drone_num,world_gain_peak_range,world_gain_var_range[0],
                                  world_evolution_speed,drone_comm,drone_view,drone_memory,drone_battery,
                                  action_step,max_age,lookahead_step,malus,final_bonus,malus_sm[0],random_episode,
                                  alpha,alpha_dec,epsilon,epsilon_dec,temperature,temperature_dec,limit_MR[0],
                                  three_MR[0],perc_pos_MR[0],perc_neg_MR1,perc_neg_MR2,pretrain_episode,
                                  train_episode,test_episode,step_num[0],testing_update,model_update,
                                  batch_update,batch_size,learning_rate_vector[2],neurons_fully[0],epochs_num,
                                  gamma[0],steps_per_epoch,verbose)

    model='Filters_Reg_Dropout'
    y = ['Filters','Regularization','Dropout']
    name='NN'
    plot_IAHOS(y,ogp,ogp2,tgp,tgp2,model)
    plot_pretrain(performances,tgp,params,name)    
else:
    labels=['10x10_3targets_2drones','Min']
    loss_mean=[]
    for v in range(1):
            optimal_values=np.load('final.npy')        
            #print (optimal_values)
            output_mean = []    
            output_mean1=[]
            output,output1,H,negative_r1,pos_r,negative_r2,pos_r2,null_r,A0,A1,A2,A3,A4,SMR,GSMR,TMR,GTMR,Mappa,VAR,MIN,MAX,MEAN,LOSSES,success_episodes,VARIATIONS= testing_network(world_size=world_size[1],
                                     target_num = 4,
                                     drone_goal = drone_goal,
                                     drone_num = 2,
                                     extra_drone_num = extra_drone_num,
                                     world_gain_peak_range=world_gain_peak_range,
                                     world_gain_var_range=world_gain_var_range[0],
                                     world_evolution_speed=world_evolution_speed,
        
                                     drone_comm=drone_comm,
                                     drone_view=drone_view,
        
                                     drone_memory=drone_memory,
                                     drone_battery=drone_battery,
                                     action_step=action_step,
                                     max_age=max_age,
                                     lookahead_step=lookahead_step,
                                     malus=malus,
                                     final_bonus=final_bonus,
                                     malus_sm=0.4,
        
                                     random_episode=random_episode,
        
                                     alpha=alpha,
                                     alpha_dec=alpha_dec,
                                     epsilon=epsilon,
                                     epsilon_dec=epsilon_dec,
                                     temperature=temperature,
                                     temperature_dec=temperature_dec,
        
                                     state_MR=v,
                                     limit_MR=limit_MR[0],
                                     three_MR= three_MR[0],
                                     prioritized=prioritized[0],
                                     perc_pos_MR= perc_pos_MR[0],
                                     perc_neg_MR1 = perc_neg_MR1,
                                     perc_neg_MR2 = perc_neg_MR2,
                                     pretrain_episode=pretrain_episode,
                                     train_episode=train_episode,
                                     test_episode=test_episode,
                                     step_num=step_num[0],
        
                                     testing_update=testing_update,
                                     model_update=model_update,
                                     batch_update=batch_update,
                                     batch_size=batch_size,
                                     learning_rate=1,
                                     neurons_fully=neurons_fully[0],
                                     drop_rate=drop_rate[0],
                                     dueling=dueling,
        
                                     epochs_num=epochs_num,
                                     gamma=0.9,
                                     steps_per_epoch=steps_per_epoch,
                                     verbose=verbose,
                                     version=v)
        
            for test in output:
                output_mean.append(np.mean(test))
            for test in output1:
                output_mean1.append(np.mean(test)*100)
        
            vector_mean.append(output_mean)
            vector_mean1.append(output_mean1)
            
            H1=[]
            H2 = []
            for i in range(len(H)):
                H1.append(np.mean(H[i]))
            for i in range(len(H1)-1000):
                H2.append(np.mean(H1[i:i+1000]))
            loss_mean.append(np.asarray(H2))
            
    np.save('mean_vector',vector_mean)
    np.save('vector_mean1',vector_mean1)
    np.save('negative_r1',negative_r1)
    np.save('negative_r2',negative_r2)
    np.save('pos_r',pos_r)
    np.save('pos_r2',pos_r2)
    np.save('null_r',null_r)
    np.save('loss_mean',loss_mean)
    np.save('LOSSES',LOSSES)
    np.save('success_episodes',success_episodes)            
    plot_results(vector_mean,vector_mean1,train_episode,negative_r1,negative_r2,pos_r,pos_r2,null_r,loss_mean,labels[0],LOSSES,success_episodes)
    #plot_training(VARIATIONS)
    #plot_memory_replay(SMR,GSMR,TMR,GTMR,Mappa,labels[v])

            #plot_qvalues(VAR,MIN,MAX,MEAN,labels[v])
        
            #np.save('example',example1)
    #♦np.save('test_nn',example)
    #np.save('training',example1)
            
