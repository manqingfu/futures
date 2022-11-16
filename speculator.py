from turtle import pos
import mesa
from mesa import Agent

import pandas as pd
import numpy  as np
import random

class Speculator(Agent):
    def __init__(self, unique_id: int, model,cash,horizon):
        super().__init__(unique_id, model)
        self.deposit=0
        self.cash=cash
        self.long=[0,0] # [this_month position, next_month position]
        self.short=[0,0]
        self.horizon=horizon
        # print(self.horizon)
        self.long_bid=[[0,0],[0,0]] # [[position, price],next_month]
        self.short_bid=[[0,0],[0,0]] # position均为正数

    def bid(self):
        ## position, price都要改
        position, price=[0,0],[0,0]
        if self.model.price_list.iloc[self.model.time]['act_settlement']==self.model.price_list.iloc[self.model.time]['con_settlement']: #操作主连
            if self.horizon>0: #看涨
                p=(round(list(np.random.normal(-2, 10**0.5, 1))[0],0)+100)/100
                price=[round(self.model.price_list.iloc[self.model.time-1]['con_closing']*p,1),0]
                position=[round(self.cash/(0.2*1000*price[0]),0),0] #买近
            elif self.horizon<0: #看跌
                p=(round(list(np.random.normal(2, 10**0.5, 1))[0],0)+100)/100
                price=[round(self.model.price_list.iloc[self.model.time-1]['con_closing']*p,1),0]
                position=[-round(self.cash/(0.2*1000*price[0]),0),0] #卖近    
        else: #操作next month
            if self.horizon>0: #看涨
                p=(round(list(np.random.normal(2, 10**0.5, 1))[0],0)+100)/100
                print(self.unique_id, self.horizon,p)
                price=[0,round(self.model.price_list.iloc[self.model.time-1]['act_closing']*p,1)]
                position=[0,-round(self.cash/(0.1*1000*price[1]),0)] #卖远
            elif self.horizon<0: #看跌
                p=(round(list(np.random.normal(-2, 10**0.5, 1))[0],0)+100)/100
                print(self.unique_id, self.horizon,p)
                price=[0,round(self.model.price_list.iloc[self.model.time-1]['act_closing']*p,1)]
                position=[0,round(self.cash/(0.1*1000*price[1]),0)] #买远              

        return position, price
        # position=[random.randint(-10,10),random.randint(-10,10)]
        # price=[0,0]
        # for i in range(len(price)):
        #     if position[i]>0:
        #         price[i]=(round(list(np.random.normal(-2, 10**0.5, 1))[0],0)+10)
        #     elif position[i]<0:
        #         price[i]=(round(list(np.random.normal(2, 10**0.5, 1))[0],0)+10)
        # return position, price

    def deal(self, hands, price, time):
        print(self.unique_id,hands,price,time)
        self.cash-=20*hands
        if hands>0:
            self.long_bid[time][1]=round((self.long_bid[time][0]*self.long_bid[time][1]+hands*price)/(self.long_bid[time][0]+hands),1)
            self.long_bid[time][0]+=hands
        elif hands<0:
            self.short_bid[time][1]=round((self.short_bid[time][0]*self.short_bid[time][1]-hands*price)/(self.short_bid[time][0]-hands),1)
            self.short_bid[time][0]-=hands

    def reward(self):
        pass

    def liquidation(self):
        # calculate the profit and loss
        self.long[0]+=self.long_bid[0][0]
        self.long[1]+=self.long_bid[1][0]
        self.short[0]+=self.short_bid[0][0]
        self.short[1]+=self.short_bid[1][0]
        # print("unique_id:",self.unique_id)
        # print(self.long_bid,self.short_bid)
        # print(self.long,self.short)
        this_reward=(self.model.predict[0][-1]-self.long_bid[0][1])*self.long_bid[0][0]+(self.short_bid[0][1]-self.model.predict[0][-1])*self.short_bid[0][0]
        next_reward=(self.model.predict[1][-1]-self.long_bid[1][1])*self.long_bid[1][0]+(self.short_bid[1][1]-self.model.predict[1][-1])*self.short_bid[1][0]
        # print(round(this_reward*1000,1),round(next_reward*1000,1))
        return round(this_reward*1000,1),round(next_reward*1000,1)
    
    def guarding(self):
        # adjust the deposit and cash
        r1,r2=self.liquidation()
        s=self.cash+self.deposit+r1+r2
        self.deposit=(self.long[0]+self.short[0])*self.model.predict[0][-1]*200+(self.long[1]+self.short[1])*self.model.predict[1][-1]*100
        self.cash=s-self.deposit
        # print("unique_id:",self.unique_id)
        # print(self.deposit,self.cash)
        # print()

    def step(self):
        position,price=self.bid() # this_month,next_month
        self.model.bid(position,price,self.unique_id)