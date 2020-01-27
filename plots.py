# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:40:05 2019

@author: Fede
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results(vector_mean,vector_mean1,train_episode,negative_r1,negative_r2,pos_r,pos_r2,null_r,loss_mean,title,LOSSES,success_episodes):
    bound=[]
    for i in range(int(train_episode/100)):
        bound.append(1)
    print (bound)
    episodes=[]
    for i in range(int(train_episode/10)):
        episodes.append(i*100)
    base=[]
    base1=[]
    base2=[]
    base3=[]
    base4=[]
    
    for i in range(int(len(negative_r1))):
        base.append(negative_r1[i]*100/(negative_r1[i]+negative_r2[i]+pos_r[i]+pos_r2[i]+null_r[i]))
        base1.append(negative_r2[i]*100/(negative_r1[i]+negative_r2[i]+pos_r[i]+pos_r2[i]+null_r[i]))
        base2.append(pos_r[i]*100/(negative_r1[i]+negative_r2[i]+pos_r[i]+pos_r2[i]+null_r[i]))
        base3.append(pos_r2[i]*100/(negative_r1[i]+negative_r2[i]+pos_r[i]+pos_r2[i]+null_r[i]))
        base4.append(null_r[i]*100/(negative_r1[i]+negative_r2[i]+pos_r[i]+pos_r2[i]+null_r[i]))
    
    fig = make_subplots(rows=2,cols=2,subplot_titles=("Rewards distributions","Times both drones reach their targets [%]","Mean reward per episode per step","Mean loss"))
    
    fig.add_trace(go.Bar(x=episodes, y=base,
                        marker_color='yellow',
                        name='Same position'),row=1,col=1)
    fig.add_trace(go.Bar(x=episodes, y=base2,
                        marker_color='darkgreen',
                        name='Optimal position'
                        ),row=1,col=1)
    fig.add_trace(go.Bar(x=episodes, y=base3,
                        marker_color='lightgreen',
                        name='Only 1 drone correct'
                        ),row=1,col=1)
    fig.add_trace(go.Bar(x=episodes, y=base1,
                        marker_color='crimson',
                        name='Outside map position'
                        ),row=1,col=1)
    fig.add_trace(go.Bar(x=episodes, y=base4,
                        marker_color='grey',
                        name='Null rewards'
                        ),row=1,col=1)
     
    min_v=[]
    max_v=[]
    mean_v=[]
    min_v1=[]
    max_v1=[]
    mean_v1=[]
    min_loss=[]
    max_loss=[]
    mean_loss=[]
    
    for i in range(len(vector_mean[0])):
        temp=[]
        temp1=[]
        for j in range(len(vector_mean)):
            temp.append(vector_mean[j][i])
            temp1.append(vector_mean1[j][i])
        min_v.append(min(temp))
        max_v.append(max(temp))
        mean_v.append(np.mean(temp))
        min_v1.append(np.mean(temp1)-min(temp1))
        max_v1.append(max(temp1)-np.mean(temp1))
        mean_v1.append(np.mean(temp1))
    for i in range(len(loss_mean[0])):
        temp_loss=[]
        for j in range(len(loss_mean)):
            temp_loss.append(loss_mean[j][i])
        min_loss.append(min(temp_loss))
        max_loss.append(max(temp_loss))
        mean_loss.append(np.mean(temp_loss))

    fig.add_trace(go.Bar(x=episodes, y=mean_v1,
                        marker_color='orange',error_y=dict(
                                            symmetric=False,
                                            array=max_v1,
                                            arrayminus=min_v1),
                        name='Perfect episodes [%]'
                        ),row=1,col=2) 

    fig.add_trace(go.Scatter(y=np.asarray(min_v),line=dict(width=0),mode='lines',name='Min reward'),row=2,col=1)  
    fig.add_trace(go.Scatter(y=np.asarray(mean_v),line=dict(color='rgb(31, 119, 180)'),
                             fillcolor='rgba(68, 68, 68, 0.3)',fill='tonexty',name='Mean reward'),row=2,col=1)
    fig.add_trace(go.Scatter(y=np.asarray(max_v),line=dict(width=0),fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty',name='Upper reward'),row=2,col=1)
    fig.add_trace(go.Scatter(y=bound,line=go.scatter.Line(color="crimson"),name='Bound reward'),
                  row=2,col=1)
    
    
    fig.add_trace(go.Scatter(y=np.asarray(min_loss),line=dict(width=0),mode='lines',name='Min reward'),row=2,col=2)  
    fig.add_trace(go.Scatter(y=np.asarray(mean_loss),line=dict(color='rgb(68, 0, 0)'),
                             fillcolor='rgba(68, 0, 0, 0.3)',fill='tonexty',name='Mean reward'),row=2,col=2)
    fig.add_trace(go.Scatter(y=np.asarray(max_loss),line=dict(width=0),fillcolor='rgba(68, 0, 0, 0.3)',
                             fill='tonexty',name='Upper reward'),row=2,col=2)
    
    fig.update_xaxes(tickfont=dict(size=30))
    fig.update_yaxes(tickfont=dict(size=30))
    fig.update_layout(barmode='relative',
                          title=go.layout.Title(
                          text="Results"),
                          title_font_size=30,
                          legend=go.layout.Legend(
                          font=dict(
                          family="sans-serif",
                          size=30,
                          color="black"
                          )),
                          width=1920,height=1080)
            
    fig.write_image('General_Results'+str(title)+'.pdf')
    
    fig = go.Figure(data=[
            go.Bar(x=episodes, y=base1,
                        marker_color='crimson',
                        name='Outside map position'),
            go.Bar(x=episodes, y=base,
                        marker_color='yellow',
                        name='Same position'),
            go.Bar(x=episodes, y=base4,
                        marker_color='grey',
                        name='Null rewards'
                        ),
            go.Bar(x=episodes, y=base3,
                        marker_color='lightgreen',
                        name='Only 1 drone correct'),
            go.Bar(x=episodes, y=base2,
                        marker_color='darkgreen',
                        name='Optimal position')
                   ])

    fig.update_layout(barmode='stack',
                          title=go.layout.Title(
                          text="Greedy rewards"),
                          yaxis=dict(title='Reward distributions (%)'),
                          xaxis=dict(title='Trained episodes',tick0 = 0,dtick = 50))
    fig.write_image('Greedy_rewards_'+str(title)+'.pdf')
    
    fig = go.Figure(data=[
            go.Bar(x=episodes, y=mean_v1,
                        marker_color='orange',error_y=dict(
                                            symmetric=False,
                                            array=max_v1,
                                            arrayminus=min_v1),
                        name='Perfect episodes [%]'
                        )])

    fig.update_layout(barmode='stack',
                          title=go.layout.Title(
                          text="Correct episodes"),
                          yaxis=dict(title='Correct episodes (%)'),
                          xaxis=dict(title='Trained episodes',tick0 = 0,dtick = 50))
    fig.write_image('Correct_episodes_'+str(title)+'.pdf')

    fig = go.Figure(data=[
            go.Scatter(y=np.asarray(min_loss),line=dict(width=0.5),mode='lines',name='Min reward'),
            go.Scatter(y=np.asarray(mean_loss),line=dict(color='rgb(68, 0, 0)'),
                             fillcolor='rgba(68, 0, 0, 0.3)',fill='tonexty',name='Mean reward'),
            go.Scatter(y=np.asarray(max_loss),line=dict(width=0.5),fillcolor='rgba(68, 0, 0, 0.3)',
                             fill='tonexty',name='Upper reward')])

    fig.update_layout(title=go.layout.Title(
                          text="Loss"),
                          yaxis=dict(title='MSE'),
                          xaxis=dict(title='Times NN trained',tick0 = 0,dtick = 50))
    fig.write_image('Loss_'+str(title)+'.pdf')    

    fig = go.Figure(data=[
            go.Scatter(y=np.asarray(min_v),line=dict(width=0.5),mode='lines',name='Min reward'),
            go.Scatter(y=np.asarray(mean_v),line=dict(color='rgb(31, 119, 180)'),
                             fillcolor='rgba(68, 68, 68, 0.3)',fill='tonexty',name='Mean reward'),
            go.Scatter(y=np.asarray(max_v),line=dict(width=0.5),fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty',name='Upper reward'),
            go.Scatter(y=bound,line=go.scatter.Line(color="crimson"),name='Bound reward'),
                  ])

    fig.update_layout(barmode='stack',
                          title=go.layout.Title(
                          text="Mean reward per episodes per step"),
                          yaxis=dict(title='Reward'),
                          xaxis=dict(title='Trained episodes',tick0 = 0,dtick = 50))
    fig.write_image('Mean_reward_'+str(title)+'.pdf') 
   
def plot_memory_replay(SMR,GSMR,TMR,GTMR,Map,title):
    fig = make_subplots(rows=2,cols=2,
                        subplot_titles=("Pre-training","Available gaussians in the map in the initial MR","Available gaussians in the map in the end in the MR","Map heterogeneity"),
                        specs=[[{"type": "domain"}, {"type": "domain"}],
                               [{"type": "domain"},{"type": "heatmap"}]])
    labels=["Outside map","Same position","0 reward", "1 Correct","All correct"]
    labels1=["0 available gaussians","1 available gaussians","2 available gaussians"]
    total=[0,0,0]
    for i in range(len(GTMR)):
        total[0]+=GTMR[i][0]
        total[1]+=GTMR[i][1]
        total[2]+=GTMR[i][2]
    fig.add_trace(go.Pie(values=GSMR[0],labels=labels),row=1,col=1)
    fig.add_trace(go.Pie(values=GTMR[0],labels=labels1),row=1,col=2)
    fig.add_trace(go.Pie(values=total,labels=labels1),row=2,col=1)
    maximum=max(max(Map[0]),max(Map[1]),max(Map[2]),max(Map[3]),max(Map[4]),max(Map[5]),max(Map[6]))
    fig.add_trace(go.Heatmap(z=[Map[0],Map[1],Map[2],Map[3],Map[4],Map[5],Map[6]],zmax=0,zmid=maximum/2,zmin=maximum,reversescale=False,showscale=False,colorscale='jet'),row=2,col=2)
    fig.update_xaxes(tickfont=dict(size=30))
    fig.update_yaxes(tickfont=dict(size=30))
    fig.update_layout(barmode='stack',
                          title=go.layout.Title(
                          text="Dataset"),
                          title_font_size=30,
                          legend=go.layout.Legend(
                          font=dict(
                          family="sans-serif",
                          size=30,
                          color="black"
                          )),
                          width=1920,height=1080)
            
    fig.write_image('Dataset'+str(title)+'.pdf')
    
def plot_qvalues(VAR,MIN,MAX,MEAN,title):
    fig = make_subplots(rows=2,cols=2,
                        subplot_titles=("Mean qvalues variance",
                                        "Mean qvalues minimum",
                                        "Mean qvalues maximum",
                                        "Mean qvalues mean"),
                        specs=[[{"type": "xy"}, {"type": "xy"}],
                               [{"type": "xy"},{"type": "xy"}]])
    VAR1=[]
    MEAN1=[]
    MINIMUM=[]
    MAXIMUM=[]
    for i in range(len(VAR)):
        VAR1.append(VAR[i][0])
        MEAN1.append(MEAN[i][0])
        MINIMUM.append(MIN[i][0])
        MAXIMUM.append(MAX[i][0])
    fig.add_trace(go.Scatter(y=VAR1,line=go.scatter.Line(color="red"),name='Variance'),row=1,col=1)
    fig.add_trace(go.Scatter(y=MINIMUM,line=go.scatter.Line(color="orange"),name='Minimum'),row=1,col=2)
    fig.add_trace(go.Scatter(y=MAXIMUM,line=go.scatter.Line(color="green"),name='Maximum'),row=2,col=1)
    fig.add_trace(go.Scatter(y=MEAN1,line=go.scatter.Line(color="darkgreen"),name='Mean'),row=2,col=2)
    fig.update_xaxes(tickfont=dict(size=30))
    fig.update_yaxes(tickfont=dict(size=30))
    fig.update_layout(title=go.layout.Title(
                          text="Dataset"),
                          title_font_size=30,
                          legend=go.layout.Legend(
                          font=dict(
                          family="sans-serif",
                          size=30,
                          color="black"
                          )),
                          width=1920,height=1080)
            
    fig.write_image('Qvalues'+str(title)+'.pdf')

def plot_IAHOS(y,ogp,ogp2,tgp,tgp2,model):
    """fig = make_subplots(rows=2, cols=2,subplot_titles=("Mean train. accuracy first round",
                                                   "Mean valid accuracy first round",
                                                  "Mean train accuracy last round",
                                                  "Mean valid accuracy last round"))"""

    fig = go.Figure()
    x = np.linspace(0,len(tgp[0])-1,len(tgp[0]))
    Colorscale = [[0, '#FF0000'],[0.5, '#F1C40F'], [1, '#00FF00']]
    """fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=[0,1],
                       z=ogp2, colorscale = Colorscale),row=1,col=1)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=[0,1],
                       z=ogp,colorscale=Colorscale),row=1,col=2)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=x,
                       z=tgp2, colorscale = Colorscale),row=2,col=1)"""
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=x,
                       z=tgp,colorscale=Colorscale))
    fig.update_layout(height=600, width=800,title=dict(text='IAHOS results'))
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/IAHOS_'+str(model)+'.pdf')

def plot_pretrain(performances,y,params,model):
    fig = make_subplots(rows=2, cols=2,subplot_titles=("All performances",
                                                   "Performances w.r.t. Gamma",
                                                  "Performances w.r.t. malus same position",
                                                  ""))
    indeces=[]
    for i in range(len(performances)):
        indeces.append(performances[i][-1])
    index1=np.argmax(indeces)
    index2=np.argmin(indeces)

    bound=[]
    for i in range(len(performances[0])):
        bound.append(1)      
    for i in range(len(performances)):
        if i!=index1 and i!=index2:
            fig.add_trace(go.Scatter(y=performances[i],line=go.scatter.Line(color="grey"),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(y=performances[index1],line=go.scatter.Line(color="green",width=4),name='Best performance'),row=1,col=1)
    fig.add_trace(go.Scatter(y=performances[index2],line=go.scatter.Line(color="red",width=4),name='Worst performance'),row=1,col=1)
    fig.add_trace(go.Scatter(y=bound,line=go.scatter.Line(color="green",dash='dashdot',width=4),name='Bound'),row=1,col=1)
    fig.add_trace(go.Bar(x=params[0], y=y[0]),row=1,col=2)
    fig.add_trace(go.Bar(x=params[1], y=y[1]),row=2,col=1)
    #fig.add_trace(go.Bar(x=params[2], y=y[2]),row=2,col=2)

    fig.update_layout(title=go.layout.Title(text="Results"),title_font_size=30,width=1920,height=1080)
            
    fig.write_image('Results_pretraining_'+str(model)+'.pdf')
    
def plot_training(VARIATIONS):
    VAR1=[]
    VAR2=[]
    VAR3=[]
    VAR4=[]
    for i in range(len(VARIATIONS)):
        VAR1.append(VARIATIONS[i][0]*100)
        VAR2.append(VARIATIONS[i][1]*100)
        VAR3.append(VARIATIONS[i][2]*100)
        VAR4.append(VARIATIONS[i][3]*100)
    fig = make_subplots(rows=2, cols=2,subplot_titles=("Outside map","Same position","Null rewards","+1 reward"))
    fig.add_trace(go.Scatter(y=VAR1,marker_color='crimson',name='Outside map position'),row=1,col=1)
    
    fig.add_trace(go.Scatter(y=VAR2,marker_color='orange', name='Same position'),row=1,col=2)
    fig.add_trace(go.Scatter(y=VAR3,marker_color='grey',name='Null rewards'),row=2,col=1)
    fig.add_trace(go.Scatter(y=VAR4,marker_color='lightgreen',name='+1 reward'),row=2,col=2)

    fig.update_layout(title=go.layout.Title(
                          text="Argmax q-values variations during training"),
                          yaxis=dict(title='Variation (%)'),
                          xaxis=dict(title='Trained episodes',tick0 = 0,dtick = 50)
                          ,width=1920,height=1080)
    fig.write_image('Argmax q-values variations during training.pdf')