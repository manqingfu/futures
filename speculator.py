from turtle import pos
import mesa
from mesa import Agent

import pandas as pd
import numpy  as np
import random
import math

class Speculator(Agent):
    def __init__(self, unique_id: int, model,cash,horizon):
        super().__init__(unique_id, model)
        self.deposit=0
        self.cash=cash
        self.long=[0,0] # [con hands, act hands]
        self.short=[0,0]
        self.horizon=horizon
        # print(self.horizon)
        self.Q_reward=np.zeros(((11,5,21)))
        self.Q_times=np.zeros(((11,5,21)))
        self.Q_r=np.zeros(((11,5,21)))
        self.Q_t=np.zeros(((11,5,21)))
        # self.policy=0
        self.policy=(0,0,0,0) #现金量占比，金额量占比，价格波动率，方向
        self.long_bid=[[0,0],[0,0]] # [[con_hands, con_price],[act_hands,con_position]]
        self.short_bid=[[0,0],[0,0]] # position均为正数
        self.count=0
        self.policy_list=[]
        self.deal_list=[]

    def bid_DQN(self):
        hands,price=[0,0],[0,0]
        
        return hands, price

    def bid_duQ(self):
        # 修改了Q——matrix的结构
        hands,price=[0,0],[0,0]
        t=1 if abs(self.horizon)>0.5 else 0 # 强看涨跌对远交易1，弱看涨跌对近交易0
        position=int(round(self.cash/(self.cash+self.deposit),1)*10) # 0,1,2,...,10
        if self.count/400<1 or (1<self.count/400 and random.random()<0.2):
            if self.horizon>0:# 看涨买
                self.policy=(position,random.randint(1,5),random.randint(-10,10),1)
                if t==0:
                    price[t]=round(self.model.price_list.iloc[self.model.time-1]['con_closing']*(100-self.policy[2]*abs(self.horizon))/100,1)
                elif t==1 or self.model.price_list.iloc[self.model.time-1]['date'] in self.model.settle_list:
                    price[t]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100-self.policy[2]*abs(self.horizon))/100,1)
                hands[t]=math.floor(self.cash*self.policy[1]/(2000*price[t]))
            else: #看跌卖
                self.policy=(position,random.randint(1,5),random.randint(-10,10),-1)
                if t==0:
                    price[t]=round(self.model.price_list.iloc[self.model.time-1]['con_closing']*(100+self.policy[2]*abs(self.horizon))/100,1)
                elif t==1 or self.model.price_list.iloc[self.model.time-1]['date'] in self.model.settle_list:
                    price[t]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100+self.policy[2]*abs(self.horizon))/100,1)
                hands[t]=-math.floor(self.cash*self.policy[1]/(2000*price[t]))   

        else:
            a,b=np.unravel_index(self.Q_reward[position].argmax(),self.Q_reward[position].shape)
            if self.horizon>0:# 看涨买
                self.policy=(position,a+1,b-10,1)
                if t==0:
                    price[t]=round(self.model.price_list.iloc[self.model.time-1]['con_closing']*(100-self.policy[2]*abs(self.horizon))/100,1)
                else:
                    price[t]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100-self.policy[2]*abs(self.horizon))/100,1)
                hands[t]=math.floor(self.cash*self.policy[1]/(2000*price[t]))
            else:
                self.policy=(position,a+1,b-10,-1)
                if t==0:
                    price[t]=round(self.model.price_list.iloc[self.model.time-1]['con_closing']*(100+self.policy[2]*abs(self.horizon))/100,1)
                else:
                    price[t]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100+self.policy[2]*abs(self.horizon))/100,1)
                hands[t]=-math.floor(self.cash*self.policy[1]/(2000*price[t]))     
        self.policy_list.append(self.policy)
        # print(self.unique_id,self.policy)            
        return hands,price


    def deal(self, hands, price, time):
        # print(self.unique_id,hands,price,time)
        # long&short_bid=[[con_hands, con_price],[act_hands,con_position]]
        self.cash-=20*abs(hands)
        if hands>0:
            self.long_bid[time][1]=round((self.long_bid[time][0]*self.long_bid[time][1]+hands*price)/(self.long_bid[time][0]+hands),1)
            self.long_bid[time][0]+=hands
        elif hands<0:
            self.short_bid[time][1]=round((self.short_bid[time][0]*self.short_bid[time][1]-hands*price)/(self.short_bid[time][0]-hands),1)
            self.short_bid[time][0]-=hands

    def reward(self, con_reward, act_reward):
        # change the policy according to the rewards
        # 分层的第一层
        if con_reward<0 and act_reward<0:
            if -0.1<=self.horizon<=0.1:
                self.horizon=0.1 if self.horizon<0 else -0.1
            else:
                self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1
            return
        elif con_reward+act_reward<=0 and random.random()<0.5:
            if -0.1<=self.horizon<=0.1:
                self.horizon=0.1 if self.horizon<0 else -0.1
            else:
                self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1
            return
        elif con_reward+act_reward>0 and -0.9<=self.horizon<=0.9:
            self.horizon=self.horizon+0.1 if self.horizon>0 else self.horizon-0.1

            
    def reward_Q(self):
        # 分层的第二层
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']] # con, act
        last_settle_price=[self.model.price_list.iloc[self.model.time-1]['con_settlement'],self.model.price_list.iloc[self.model.time-1]['act_settlement']]
        for i in range(len(self.policy_list)):
            if self.deal_list[i][1]>0:
                p=self.policy_list[i]
                if p[3]<0: #short
                    self.Q_r[p[0]][p[1]-1][p[2]+10]+=(last_settle_price[self.deal_list[i][0]]-settle_price[self.deal_list[i][0]])*self.deal_list[i][1]
                else:
                    self.Q_r[p[0]][p[1]-1][p[2]+10]+=(settle_price[self.deal_list[i][0]]-last_settle_price[self.deal_list[i][0]])*self.deal_list[i][1]

        if self.short_bid[0][0]+self.short_bid[1][0]>0:
            self.Q_r[self.policy[0]][self.policy[1]-1][self.policy[2]+10]=(self.Q_r[self.policy[0]][self.policy[1]-1][self.policy[2]+10]*self.Q_t[self.policy[0]][self.policy[1]-1][self.policy[2]+10]+(self.short_bid[0][1]-settle_price[0])*self.short_bid[0][0]+(self.short_bid[1][1]-settle_price[1])*self.short_bid[1][0])
        elif self.long_bid[0][0]+self.long_bid[1][0]>0:
            self.Q_r[self.policy[0]][self.policy[1]-1][self.policy[2]+10]=(self.Q_r[self.policy[0]][self.policy[1]-1][self.policy[2]+10]*self.Q_t[self.policy[0]][self.policy[1]-1][self.policy[2]+10]+(settle_price[0]-self.long_bid[0][1])*self.long_bid[0][0]+(settle_price[1]-self.long_bid[1][1])*self.long_bid[1][0])
        # else:
        #     self.Q_r[self.policy[0]][self.policy[1]-1][self.policy[2]+10]-=self.cash*self.model.inflation/365
        self.Q_t[self.policy[0]][self.policy[1]-1][self.policy[2]+10]+=1    


    def liquidation(self):
        # calculate the profit and loss everyday
        if self.long_bid[0][0]+self.long_bid[1][0]>0:
            t=0 if self.long_bid[0][0]>0 else 1
            self.deal_list.append([t,self.long_bid[0][0]+self.long_bid[1][0]])
        else:
            t=0 if self.short_bid[0][0]>0 else 1
            self.deal_list.append([t,self.short_bid[0][0]+self.short_bid[1][0]])
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']] # con, act
        con_reward=(settle_price[0]-self.long_bid[0][1])*self.long_bid[0][0]+(self.short_bid[0][1]-settle_price[0])*self.short_bid[0][0]
        act_reward=(settle_price[1]-self.long_bid[1][1])*self.long_bid[1][0]+(self.short_bid[1][1]-settle_price[1])*self.short_bid[1][0]
        
        last_settle_price=[self.model.price_list.iloc[self.model.time-1]['con_settlement'],self.model.price_list.iloc[self.model.time-1]['act_settlement']]
        con_reward+=(settle_price[0]-last_settle_price[0])*self.long[0]+(last_settle_price[0]-settle_price[0])*self.short[0]
        act_reward+=(settle_price[1]-last_settle_price[1])*self.long[1]+(last_settle_price[1]-settle_price[1])*self.short[1]

        # self.reward(con_reward,act_reward) # reward horizon policy
        self.reward_Q()

        self.long[0]+=self.long_bid[0][0]
        self.long[1]+=self.long_bid[1][0]
        self.short[0]+=self.short_bid[0][0]
        self.short[1]+=self.short_bid[1][0]
        
        return round(con_reward*1000,1),round(act_reward*1000,1)
    
    def guarding(self):
        # adjust the deposit and cash
        r1,r2=self.liquidation()
        s=self.cash+self.deposit+r1+r2
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']]
        self.deposit=(self.long[0]+self.short[0])*settle_price[0]*200+(self.long[1]+self.short[1])*settle_price[1]*200
        self.cash=s-self.deposit
        if self.cash<=0:
            # 破产后，强制平仓
            # print("bomb in ordinary",s,self.cash,self.deposit,self.long,self.short)
            self.model.bomb+=1
            self.deposit=0
            self.cash=self.model.initial_wealth
            # if -0.1<=self.horizon<=0.1:
            #     self.horizon=0.1 if self.horizon<0 else -0.1
            # else:
            #     self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1   
            self.long=[0,0] # [con position, act position]
            self.short=[0,0]
            self.count=0
            for x in range(len(self.Q_reward)):
                for y in range(len(self.Q_reward[x])):
                    for z in range(len(self.Q_reward[x][y])):
                        self.Q_reward[x][y][z]=(self.Q_reward[x][y][z]*self.Q_times[x][y][z]+self.Q_r[x][y][z]*self.Q_t[x][y][z])/(self.Q_times[x][y][z]+self.Q_t[x][y][z]) if (self.Q_times[x][y][z]+self.Q_t[x][y][z])!=0 else 0
                        self.Q_times[x][y][z]+=self.Q_t[x][y][z]
            self.Q_r=np.zeros(((11,5,21)))
            self.Q_t=np.zeros(((11,5,21)))
            self.policy_list=[]
        
            
    def delivery(self):
        self.long[0],self.long[1]=self.long[1],0
        self.short[0],self.short[1]=self.short[1],0
        s=self.cash+self.deposit
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']]
        self.deposit=(self.long[0]+self.short[0])*settle_price[0]*200
        self.cash=s-self.deposit 
        if self.cash<=0:
        # 交割日后破产
            # print("bomb after delivery",s,self.cash,self.deposit,self.long,self.short)
            self.model.bomb+=1
            self.deposit=0
            self.cash=self.model.initial_wealth
            # if -0.1<=self.horizon<=0.1:
            #     self.horizon=0.1 if self.horizon<0 else -0.1
            # else:
            #     self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1  
            # print(self.unique_id,self.horizon)          
            self.long=[0,0] # [con position, act position]
            self.short=[0,0]
            self.count=0
        for x in range(len(self.Q_reward)):
            for y in range(len(self.Q_reward[x])):
                for z in range(len(self.Q_reward[x][y])):
                    self.Q_reward[x][y][z]=(self.Q_reward[x][y][z]*self.Q_times[x][y][z]+self.Q_r[x][y][z]*self.Q_t[x][y][z])/(self.Q_times[x][y][z]+self.Q_t[x][y][z]) if (self.Q_times[x][y][z]+self.Q_t[x][y][z])!=0 else 0
                    self.Q_times[x][y][z]+=self.Q_t[x][y][z]
        self.Q_r=np.zeros(((11,5,21)))
        self.Q_t=np.zeros(((11,5,21)))
        self.policy_list=[]
        # print(self.Q_reward)


    def step(self):
        self.long_bid=[[0,0],[0,0]] # [[position, price],next_month]
        self.short_bid=[[0,0],[0,0]] # position均为正数
        position,price=self.bid_duQ() # this_month,next_month
        self.model.bid(position,price,self.unique_id)
        self.count+=1
        # print(self.Q_r)