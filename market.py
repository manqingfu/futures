from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

import pandas as pd
import numpy as np
import random
import copy
import pickle

from speculator import *

class Market(Model):
    def __init__(self, N, data, settle_date):
        # super().__init__(*args, **kwargs)

        self.price_list=data
        self.settle_list=settle_date

        self.time=1 #系统日期
        self.settlement=True if self.price_list.iloc[self.time]['date'] in self.settle_list else False
        
        self.schedule=RandomActivation(self)
        self.agent_list=[]
        # y=list(np.random.normal(2.052, 3.82**0.5, 2*N))
        # y=[abs(round(i,1)) for i in y if 0<=i<=10]
        for i in range(N):
            #### important
            horizon=random.choice([-1,1])
            # print(horizon,y[i])
            agent=Speculator(i,self,cash=1000000,horizon=horizon)
            self.schedule.add(agent=agent)
            self.agent_list.append(agent)

        long=pd.DataFrame({'long':[],'price':[],'agent':[]})
        short=pd.DataFrame({'short':[],'price':[],'agent':[]})
        self.this_month_table=[copy.deepcopy(long),copy.deepcopy(short)]
        self.next_month_table=[copy.deepcopy(long),copy.deepcopy(short)]
        self.predict=[[self.price_list.iloc[self.time-1]['con_settlement']],[self.price_list.iloc[self.time-1]['act_settlement']]]

    def step(self):
        for i in range(1,len(self.price_list)):
            print(i)
            if self.settlement:
                self.delivery()
            long=pd.DataFrame({'long':[],'price':[],'agent':[]})
            short=pd.DataFrame({'short':[],'price':[],'agent':[]})
            self.this_month_table=[copy.deepcopy(long),copy.deepcopy(short)]
            self.next_month_table=[copy.deepcopy(long),copy.deepcopy(short)]
            self.schedule.step()
            # print(self.this_month_table)
            # print(self.next_month_table)
            self.pricing()
            self.guarding()
            self.time+=1
        for a in self.agent_list:
            print(a.unique_id,a.cash+a.deposit, a.horizon)

    def bid(self,position,price, agentid):
        ## 之后想想办法把这个函数写简单一点
        if position[0]>0: #long this month
            temp=self.this_month_table[0][self.this_month_table[0]['price']>=price[0]].index.tolist()
            p=temp[-1]+1 if len(temp)!=0 else 0
            df = pd.DataFrame(np.insert(self.this_month_table[0].values, p, values=[position[0],price[0],agentid], axis=0))
            df.columns=self.this_month_table[0].columns
            self.this_month_table[0]=copy.deepcopy(df)
        elif position[0]<0: # short this month
            temp=self.this_month_table[1][self.this_month_table[1]['price']>=price[0]].index.tolist()
            p=temp[-1]+1 if len(temp)!=0 else 0
            df = pd.DataFrame(np.insert(self.this_month_table[1].values, p, values=[position[0],price[0],agentid], axis=0))
            df.columns=self.this_month_table[1].columns
            self.this_month_table[1]=copy.deepcopy(df)
        if position[1]>0: #long this month
            temp=self.next_month_table[0][self.next_month_table[0]['price']>=price[1]].index.tolist()
            p=temp[-1]+1 if len(temp)!=0 else 0
            df = pd.DataFrame(np.insert(self.next_month_table[0].values, p, values=[position[1],price[1],agentid], axis=0))
            df.columns=self.next_month_table[0].columns
            self.next_month_table[0]=copy.deepcopy(df)
        elif position[1]<0: # short this month
            temp=self.next_month_table[1][self.next_month_table[1]['price']>=price[1]].index.tolist()
            p=temp[-1]+1 if len(temp)!=0 else 0
            df = pd.DataFrame(np.insert(self.next_month_table[1].values, p, values=[position[1],price[1],agentid], axis=0))
            df.columns=self.next_month_table[1].columns
            self.next_month_table[1]=copy.deepcopy(df)

    def pricing(self):
        # 买价高于卖价，以买价成交
        this_money=0
        this_hands=0
        p,q=0,0 #long, short
        # print(type(self.this_month_table[0].iloc[p]['agent'].item()))
        while p<len(self.this_month_table[0]) and q<len(self.this_month_table[1]):
            # print(int(self.this_month_table[0].iloc[p]['agent']),int(self.this_month_table[1].iloc[q]['agent']))
            if self.this_month_table[0].iloc[p]['price']<self.this_month_table[1].iloc[q]['price']:
                q+=1
                continue
            else:
                if self.this_month_table[0].iloc[p]['long']>=abs(self.this_month_table[1].iloc[q]['short']):
                    self.agent_list[int(self.this_month_table[0].iloc[p]['agent'])].deal(-self.this_month_table[1].iloc[q]['short'],self.this_month_table[0].iloc[p]['price'],0)
                    self.agent_list[int(self.this_month_table[1].iloc[q]['agent'])].deal(self.this_month_table[1].iloc[q]['short'],self.this_month_table[0].iloc[p]['price'],0)
                    this_money+=abs(self.this_month_table[1].iloc[q]['short'])*self.this_month_table[0].iloc[p]['price']
                    this_hands+=abs(self.this_month_table[1].iloc[q]['short'])
                    self.this_month_table[0].iloc[p]['long']+=self.this_month_table[1].iloc[q]['short']
                    self.this_month_table[1].iloc[q]['short']=0
                    q+=1
                else:
                    self.agent_list[int(self.this_month_table[0].iloc[p]['agent'])].deal(self.this_month_table[0].iloc[p]['long'],self.this_month_table[0].iloc[p]['price'],0)
                    self.agent_list[int(self.this_month_table[1].iloc[q]['agent'])].deal(-self.this_month_table[0].iloc[p]['long'],self.this_month_table[0].iloc[p]['price'],0)
                    this_money+=self.this_month_table[0].iloc[p]['long']*self.this_month_table[0].iloc[p]['price']
                    this_hands+=self.this_month_table[0].iloc[p]['long']
                    self.this_month_table[1].iloc[q]['short']+=self.this_month_table[0].iloc[p]['long']
                    self.this_month_table[0].iloc[p]['long']=0
                    p+=1      
        self.predict[0].append(round(this_money/this_hands,1) if this_hands!=0 else self.predict[0][-1])

        next_money=0
        next_hands=0
        p,q=0,0 #long, short
        while p<len(self.next_month_table[0]) and q<len(self.next_month_table[1]):
            # print(int(self.next_month_table[0].iloc[p]['agent']),int(self.next_month_table[1].iloc[q]['agent']))
            if self.next_month_table[0].iloc[p]['price']<self.next_month_table[1].iloc[q]['price']:
                q+=1
                continue
            else:
                if self.next_month_table[0].iloc[p]['long']>=abs(self.next_month_table[1].iloc[q]['short']):
                    self.agent_list[int(self.next_month_table[0].iloc[p]['agent'])].deal(abs(self.next_month_table[1].iloc[q]['short']),self.next_month_table[0].iloc[p]['price'],1)
                    self.agent_list[int(self.next_month_table[1].iloc[q]['agent'])].deal(abs(self.next_month_table[1].iloc[q]['short']),self.next_month_table[0].iloc[p]['price'],1)
                    next_money+=abs(self.next_month_table[1].iloc[q]['short'])*self.next_month_table[0].iloc[p]['price']
                    next_hands+=abs(self.next_month_table[1].iloc[q]['short'])
                    self.next_month_table[0].iloc[p]['long']+=self.next_month_table[1].iloc[q]['short']
                    self.next_month_table[1].iloc[q]['short']=0
                    q+=1
                else:
                    self.agent_list[int(self.next_month_table[0].iloc[p]['agent'])].deal(self.next_month_table[0].iloc[p]['long'],self.next_month_table[0].iloc[p]['price'],1)
                    self.agent_list[int(self.next_month_table[1].iloc[q]['agent'])].deal(self.next_month_table[0].iloc[p]['long'],self.next_month_table[0].iloc[p]['price'],1)
                    next_money+=self.next_month_table[0].iloc[p]['long']*self.next_month_table[0].iloc[p]['price']
                    next_hands+=self.next_month_table[0].iloc[p]['long']
                    self.next_month_table[1].iloc[q]['short']+=self.next_month_table[0].iloc[p]['long']
                    self.next_month_table[0].iloc[p]['long']=0
                    p+=1      
        self.predict[1].append(round(next_money/next_hands,1) if next_hands!=0 else self.predict[1][-1])
    
    def guarding(self):
        for agent in self.agent_list:
            agent.guarding()

    def delivery(self):
        for agent in self.agent_list:
            agent.delivery()        


if __name__=="__main__":
    data=pd.read_csv(r'data.csv')

    fr=open("settle_date.txt","rb")
    settle_date=pickle.load(fr)

    market=Market(500,data=data,settle_date=settle_date)
    market.step()
    print(market.predict)
    file=open("predict.txt","wb")
    pickle.dump(market.predict,file)