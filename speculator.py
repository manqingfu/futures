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
        self.long=[0,0] # [this_month position, next_month position]
        self.short=[0,0]
        self.horizon=horizon
        self.Q_value=[0]*100
        self.Q_count=[0]*100
        self.policy=0
        self.long_bid=[[0,0],[0,0]] # [[position, price],next_month]
        self.short_bid=[[0,0],[0,0]] # position均为正数
        self.count=0

    def bid(self):
        position, price=[0,0],[0,0]
        if self.cash<self.deposit:
            return position,price
        if random.random()>self.count>200:
            p=round(list(np.random.normal(2.052, 3.82**0.5, 1))[0],1) # 保留一位小数
            while p<=-5 or p>=5:
                p=round(list(np.random.normal(2.052, 3.82**0.5, 1))[0],1)
        else:
            if random.random()<0.8:
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
                position=[math.floor(self.cash*0.25/(200*price[0])),-math.floor(self.cash*0.25/(200*price[1]))]
            elif 0.6>=abs(self.horizon)>0.3:
                position=[math.floor(self.cash*0.5/(200*price[0])),0]
        elif self.horizon<0: #看跌，卖近买远
            # 卖近
            price[0]=round(self.model.price_list.iloc[self.model.time-1]['con_closing']*(100+p)/100,1)
            # 买远
            price[1]=round(self.model.price_list.iloc[self.model.time-1]['act_closing']*(100-p)/100,1)
            # 根据cash确定仓位
            if abs(self.horizon)>0.6:
                position=[-math.floor(self.cash*0.25/(200*price[0])),math.floor(self.cash*0.25/(200*price[1]))]
            elif 0.6>=abs(self.horizon)>0.3:
                position=[-math.floor(self.cash*0.5/(200*price[0])),0]
        return position, price

    def deal(self, hands, price, time):
        # print(self.unique_id,hands,price,time)
        self.cash-=20*hands
        if hands>0:
            self.long_bid[time][1]=round((self.long_bid[time][0]*self.long_bid[time][1]+hands*price)/(self.long_bid[time][0]+hands),1)
            self.long_bid[time][0]+=hands
        elif hands<0:
            self.short_bid[time][1]=round((self.short_bid[time][0]*self.short_bid[time][1]-hands*price)/(self.short_bid[time][0]-hands),1)
            self.short_bid[time][0]-=hands

    def reward(self, this_reward, next_reward):
        # change the policy according to the rewards
        if this_reward<0 and next_reward<0:
            if -0.1<=self.horizon<=0.1:
                self.horizon=0.1 if self.horizon<0 else -0.1
            else:
                self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1
        elif this_reward+next_reward<0:
            if random.random()<0.5:
                if -0.1<=self.horizon<=0.1:
                    self.horizon=0.1 if self.horizon<0 else -0.1
                else:
                    self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1
        elif abs(self.horizon)<1:
            self.horizon=self.horizon+0.1 if self.horizon>0 else self.horizon-0.1
        self.Q_value[int((self.policy+5)*10-1)]=(self.Q_value[int((self.policy+5)*10-1)]*self.Q_count[int((self.policy+5)*10-1)]+this_reward+next_reward)/(self.Q_count[int((self.policy+5)*10-1)]+1)
        self.Q_count[int((self.policy+5)*10-1)]+=1
        # print(self.unique_id,self.horizon)

    def liquidation(self):
        # calculate the profit and loss everyday
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']]
        last_settle_price=[self.model.price_list.iloc[self.model.time-1]['con_settlement'],self.model.price_list.iloc[self.model.time-1]['act_settlement']]
        
        this_reward=(settle_price[0]-self.long_bid[0][1])*self.long_bid[0][0]+(self.short_bid[0][1]-settle_price[0])*self.short_bid[0][0]
        this_reward+=(settle_price[0]-last_settle_price[0])*self.long[0]+(last_settle_price[0]-settle_price[0])*self.short[0]
        next_reward=(settle_price[1]-self.long_bid[1][1])*self.long_bid[1][0]+(self.short_bid[1][1]-settle_price[1])*self.short_bid[1][0]
        next_reward+=(settle_price[1]-last_settle_price[1])*self.long[1]+(last_settle_price[1]-settle_price[1])*self.short[1]
        self.reward(this_reward,next_reward)

        self.long[0]+=self.long_bid[0][0]
        self.long[1]+=self.long_bid[1][0]
        self.short[0]+=self.short_bid[0][0]
        self.short[1]+=self.short_bid[1][0]
        
        return round(this_reward*1000,1),round(next_reward*1000,1)
    
    def guarding(self):
        # adjust the deposit and cash
        r1,r2=self.liquidation()
        s=self.cash+self.deposit+r1+r2
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']]
        self.deposit=(self.long[0]+self.short[0])*settle_price[0]*200+(self.long[1]+self.short[1])*settle_price[1]*100
        self.cash=s-self.deposit
        if self.cash<=0:
            # 破产后，强制平仓
            self.deposit=0
            self.cash=1000000
            if -0.1<=self.horizon<=0.1:
                self.horizon=0.1 if self.horizon<0 else -0.1
            else:
                self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1  
            # print(self.unique_id,self.horizon)          
            self.long=[0,0] # [this_month position, next_month position]
            self.short=[0,0]
            self.Q_value=[0]*100
            self.Q_count=[0]*100
            self.count=0
        
    def delivery(self):
        self.long[0],self.long[1]=self.long[1],0
        self.short[0],self.short[1]=self.short[1],0
        s=self.cash+self.deposit
        settle_price=[self.model.price_list.iloc[self.model.time]['con_settlement'],self.model.price_list.iloc[self.model.time]['act_settlement']]
        self.deposit=(self.long[0]+self.short[0])*settle_price[0]*200
        self.cash=s-self.deposit 
        if self.cash<=0:
        # 交割日后破产
            print("bomb after delivery",s,self.cash,self.deposit)
            self.deposit=0
            self.cash=1000000
            if -0.1<=self.horizon<=0.1:
                self.horizon=0.1 if self.horizon<0 else -0.1
            else:
                self.horizon=self.horizon+0.1 if self.horizon<0 else self.horizon-0.1  
            # print(self.unique_id,self.horizon)          
            self.long=[0,0] # [this_month position, next_month position]
            self.short=[0,0]
            self.Q_value=[0]*100
            self.Q_count=[0]*100
            self.count=0


    def step(self):
        self.long_bid=[[0,0],[0,0]] # [[position, price],next_month]
        self.short_bid=[[0,0],[0,0]] # position均为正数
        position,price=self.bid() # this_month,next_month
        self.model.bid(position,price,self.unique_id)
        self.count+=1