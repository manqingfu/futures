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
        self.Q_reward=np.zeros(((2,5,21)))
        self.Q_hands=np.zeros(((2,5,21)))
        self.Q_r=np.zeros(((2,5,21)))
        self.Q_h=np.zeros(((2,5,21)))
        # self.policy=0
        self.policy=(0,0,0) #方向，金额量占比，价格波动率
        self.long_bid=[[0,0],[0,0]] # [[con_hands, con_price],[act_hands,con_position]]
        self.short_bid=[[0,0],[0,0]] # position均为正数
        self.count=0

    def bid_duQ(self):
        position,price=[0,0],[0,0]
        if abs(self.horizon)>0.8: #强看涨跌，只做远期
            if self.count/200<1 or (1<self.count/200 and random.random()<0.1): #冷启动
                if self.horizon>0: #卖远
                    self.policy=(-1,random.randint(1,5),random.randint(-10,10))
                    price[1]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100+self.policy[2])/100,1)
                    position[1]=-math.floor(self.cash*self.policy[1]/(2000*price[1]))
                else: #买远
                    self.policy=(1,random.randint(1,5),random.randint(-10,10))
                    price[1]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100-self.policy[2])/100,1)
                    position[1]=math.floor(self.cash*self.policy[1]/(2000*price[1]))                   
            else:
                if self.horizon>0: 
                    a,b=np.unravel_index(self.Q_reward[0].argmax(),self.Q_reward[0].shape)
                    self.policy=(-1,a+1,b-10)
                    price[1]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100+self.policy[2])/100,1)
                    position[1]=-math.floor(self.cash*self.policy[1]/(2000*price[1]))
                else:
                    a,b=np.unravel_index(self.Q_reward[1].argmax(),self.Q_reward[1].shape)
                    self.policy=(1,a,b-10)
                    price[1]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100-self.policy[2])/100,1)
                    position[1]=math.floor(self.cash*self.policy[1]/(2000*price[1]))                  
                
        else: # 弱看涨跌，只做近期
            if self.count/200<1 or (1<self.count/200 and random.random()<0.1):
                if self.horizon>0: #买远
                    self.policy=(1,random.randint(1,5),random.randint(-10,10))
                    price[0]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100-self.policy[2])/100,1)
                    position[0]=math.floor(self.cash*self.policy[1]/(2000*price[0]))
                else: #卖远
                    self.policy=(-1,random.randint(1,5),random.randint(-10,10))
                    price[0]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100+self.policy[2])/100,1)
                    position[0]=-math.floor(self.cash*self.policy[1]/(2000*price[0]))                   
            else:
                if self.horizon>0: 
                    a,b=np.unravel_index(self.Q_reward[1].argmax(),self.Q_reward[1].shape)
                    self.policy=(1,a+1,b-10)
                    price[0]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100-self.policy[2])/100,1)
                    position[0]=math.floor(self.cash*self.policy[1]/(2000*price[0]))
                else:
                    a,b=np.unravel_index(self.Q_reward[0].argmax(),self.Q_reward[0].shape)
                    self.policy=(-1,a,b-10)
                    price[0]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100+self.policy[2])/100,1)
                    position[0]=-math.floor(self.cash*self.policy[1]/(2000*price[0]))  
        return position,price

    def bid(self):
        position, price=[0,0],[0,0]
        if self.cash<self.deposit:
            return position,price
        if random.random()>self.count/200:
            p=round(list(np.random.normal(2.052, 3.82**0.5, 1))[0],1) # 保留一位小数
            while p<=-5 or p>=5:
                p=round(list(np.random.normal(2.052, 3.82**0.5, 1))[0],1)
        else:
            if random.random()<0.9:
                p=round((self.Q_value.index(np.max(self.Q_value))+1)/10,1)-5
            else:
                p=round(list(np.random.normal(2.052, 3.82**0.5, 1))[0],1) # 保留一位小数
                while p<=-5 or p>=5:
                    p=round(list(np.random.normal(2.052, 3.82**0.5, 1))[0],1)
        self.policy=p
        if self.horizon>0: #看涨，买近卖远
            # 买近
            price[0]=round(self.model.price_list.iloc[self.model.time-1]['con_closing']*(100-p)/100,1)
            # 卖远
            price[1]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100+p)/100,1)
            # 根据cash确定仓位
            if abs(self.horizon)>0.6:
                position=[random.randint(0,int(self.cash*0.5/(200*price[0]))),random.randint(-int(self.cash*0.5/(200*price[1])),0)]
            elif 0.6>=abs(self.horizon)>0.3:
                position=[random.randint(0,int(self.cash/(200*price[0]))),0]
        elif self.horizon<0: #看跌，卖近买远
            # 卖近
            price[0]=round(self.model.price_list.iloc[self.model.time-1]['con_closing']*(100+p)/100,1)
            # 买远
            price[1]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100-p)/100,1)
            # 根据cash确定仓位
            if abs(self.horizon)>0.6:
                position=[random.randint(-int(self.cash*0.5/(200*price[0])),0),random.randint(0,int(self.cash*0.5/(200*price[1])))]
            elif 0.6>=abs(self.horizon)>0.3:
                position=[random.randint(-int(self.cash/(200*price[0])),0),0]
        return position, price

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
        # 需要修改啊……目前的判定标准太水了
        if con_reward<0 and act_reward<0:
            if -0.1<=self.horizon<=0.1:
                self.horizon=0.1 if self.horizon<0 else -0.1
            else:
                self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1
        elif con_reward+act_reward<0:
            if random.random()<0.5:
                if -0.1<=self.horizon<=0.1:
                    self.horizon=0.1 if self.horizon<0 else -0.1
                else:
                    self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1
        elif abs(self.horizon)<1:
            if random.random()<0.5:
                self.horizon=self.horizon+0.1 if self.horizon>0 else self.horizon-0.1

            
    def reward_Q(self):
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']] # con, act
        last_settle_price=[self.model.price_list.iloc[self.model.time-1]['con_settlement'],self.model.price_list.iloc[self.model.time-1]['act_settlement']]
        for i in range(len(self.Q_r[0])):
            for j in range(len(self.Q_r[0][i])):
                if self.Q_h[0][i][j]!=0:
                    self.Q_r[0][i][j]+=(last_settle_price[0]-settle_price[0])+(last_settle_price[1]-settle_price[1]) #买空的越跌越赚
        for i in range(len(self.Q_r[1])):
            for j in range(len(self.Q_r[1][i])):
                if self.Q_h[1][i][j]!=0:
                    self.Q_r[1][i][j]+=(settle_price[0]-last_settle_price[0])+(settle_price[1]-last_settle_price[1]) #买空的越跌越赚

        # print(self.policy[0],self.policy[1]-1,self.policy[2]+10)
        if self.policy[0]==-1:# 卖
            self.Q_r[0][self.policy[1]-1][self.policy[2]+10]=(self.Q_r[0][self.policy[1]-1][self.policy[2]+10]*self.Q_h[0][self.policy[1]-1][self.policy[2]+10]+(self.short_bid[0][1]-settle_price[0])*self.short_bid[0][0]+(self.short_bid[1][1]-settle_price[1])*self.short_bid[1][0])/(self.Q_h[0][self.policy[1]-1][self.policy[2]+10]+self.short_bid[0][0]+self.short_bid[1][0]) if (self.Q_h[0][self.policy[1]-1][self.policy[2]+10]+self.short_bid[0][0]+self.short_bid[1][0])!=0 else 0 # (Q_r+con_short_price_r+act_short_price_r)/(Q_hands+con_short_hands+act_short_hands)
            self.Q_h[0][self.policy[1]-1][self.policy[2]+10]+=self.short_bid[0][0]+self.short_bid[1][0]
        else:
            self.Q_r[1][self.policy[1]-1][self.policy[2]+10]=(self.Q_r[1][self.policy[1]-1][self.policy[2]+10]*self.Q_h[1][self.policy[1]-1][self.policy[2]+10]+(settle_price[0]-self.long_bid[0][1])*self.long_bid[0][0]+(settle_price[1]-self.long_bid[1][1])*self.long_bid[1][0])/(self.Q_h[0][self.policy[1]-1][self.policy[2]+10]+self.long_bid[0][0]+self.long_bid[1][0]) if (self.Q_h[0][self.policy[1]-1][self.policy[2]+10]+self.long_bid[0][0]+self.long_bid[1][0])!=0 else 0 # (Q_r+con_short_price_r+act_short_price_r)/(Q_hands+con_short_hands+act_short_hands)
            self.Q_h[1][self.policy[1]-1][self.policy[2]+10]+=self.long_bid[0][0]+self.long_bid[1][0]    
        # print(self.Q_r[0][self.policy[1]-1][self.policy[2]+10],self.Q_h[0][self.policy[1]-1][self.policy[2]+10],self.Q_r[1][self.policy[1]-1][self.policy[2]+10],self.Q_h[1][self.policy[1]-1][self.policy[2]+10])


    def liquidation(self):
        # calculate the profit and loss everyday
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']] # con, act
        con_reward=(settle_price[0]-self.long_bid[0][1])*self.long_bid[0][0]+(self.short_bid[0][1]-settle_price[0])*self.short_bid[0][0]
        act_reward=(settle_price[1]-self.long_bid[1][1])*self.long_bid[1][0]+(self.short_bid[1][1]-settle_price[1])*self.short_bid[1][0]
        
        last_settle_price=[self.model.price_list.iloc[self.model.time-1]['con_settlement'],self.model.price_list.iloc[self.model.time-1]['act_settlement']]
        con_reward+=(settle_price[0]-last_settle_price[0])*self.long[0]+(last_settle_price[0]-settle_price[0])*self.short[0]
        act_reward+=(settle_price[1]-last_settle_price[1])*self.long[1]+(last_settle_price[1]-settle_price[1])*self.short[1]

        self.reward(con_reward,act_reward) # reward horizon policy
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
            print("bomb in ordinary",s,self.cash,self.deposit,self.long,self.short)
            self.model.bomb+=1
            self.deposit=0
            self.cash=1000000
            if -0.1<=self.horizon<=0.1:
                self.horizon=0.1 if self.horizon<0 else -0.1
            else:
                self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1   
            self.long=[0,0] # [con position, act position]
            self.short=[0,0]
            self.count=0
            for x in range(len(self.Q_reward)):
                for y in range(len(self.Q_reward[x])):
                    for z in range(len(self.Q_reward[x][y])):
                        self.Q_reward[x][y][z]=(self.Q_reward[x][y][z]*self.Q_hands[x][y][z]+self.Q_r[x][y][z]*self.Q_h[x][y][z])/(self.Q_hands[x][y][z]+self.Q_h[x][y][z]) if (self.Q_hands[x][y][z]+self.Q_h[x][y][z]) !=0 else 0
                        self.Q_hands[x][y][z]+=self.Q_h[x][y][z]
            self.Q_r=np.zeros(((2,5,21)))
            self.Q_h=np.zeros(((2,5,21)))
        
            
    def delivery(self):
        self.long[0],self.long[1]=self.long[1],0
        self.short[0],self.short[1]=self.short[1],0
        s=self.cash+self.deposit
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']]
        self.deposit=(self.long[0]+self.short[0])*settle_price[0]*200
        self.cash=s-self.deposit 
        if self.cash<=0:
        # 交割日后破产
            print("bomb after delivery",s,self.cash,self.deposit,self.long,self.short)
            self.model.bomb+=1
            self.deposit=0
            self.cash=1000000
            if -0.1<=self.horizon<=0.1:
                self.horizon=0.1 if self.horizon<0 else -0.1
            else:
                self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1  
            # print(self.unique_id,self.horizon)          
            self.long=[0,0] # [con position, act position]
            self.short=[0,0]
            self.count=0
        for x in range(len(self.Q_reward)):
            for y in range(len(self.Q_reward[x])):
                for z in range(len(self.Q_reward[x][y])):
                    self.Q_reward[x][y][z]=(self.Q_reward[x][y][z]*self.Q_hands[x][y][z]+self.Q_r[x][y][z]*self.Q_h[x][y][z])/(self.Q_hands[x][y][z]+self.Q_h[x][y][z]) if (self.Q_hands[x][y][z]+self.Q_h[x][y][z])!=0 else 0
                    self.Q_hands[x][y][z]+=self.Q_h[x][y][z]
        self.Q_r=np.zeros(((2,5,21)))
        self.Q_h=np.zeros(((2,5,21)))
        # print(self.Q_reward)


    def step(self):
        self.long_bid=[[0,0],[0,0]] # [[position, price],next_month]
        self.short_bid=[[0,0],[0,0]] # position均为正数
        position,price=self.bid_duQ() # this_month,next_month
        self.model.bid(position,price,self.unique_id)
        self.count+=1
        # print(self.Q_r)