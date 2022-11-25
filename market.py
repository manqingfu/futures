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
        self.price_list=data
        self.settle_list=settle_date

        self.time=1 #系统日期
        self.settlement=True if self.price_list.iloc[self.time]['date'] in self.settle_list else False
        self.datetime=self.price_list.iloc[self.time]['date']
        self.con_deal=0
        self.act_deal=0
        self.inflation=0.029
        
        self.schedule=RandomActivation(self)
        self.agent_list=[]
        self.bomb=0
        self.act_deal=0
        self.con_deal=0
        for i in range(N):
            horizon=random.choice([-1,1])
            agent=Speculator(i,self,cash=1000000,horizon=horizon)
            self.schedule.add(agent=agent)
            self.agent_list.append(agent)

        long=pd.DataFrame({'long':[],'price':[],'agent':[]})
        short=pd.DataFrame({'short':[],'price':[],'agent':[]})
        self.con_table=[copy.deepcopy(long),copy.deepcopy(short)]
        self.act_table=[copy.deepcopy(long),copy.deepcopy(short)]
        self.predict=[[self.price_list.iloc[self.time-1]['con_settlement']],[self.price_list.iloc[self.time-1]['act_settlement']]]

        self.datacollector=DataCollector(
            model_reporters={
                'bomb':"bomb",
                'datetime':"datetime",
                'act_deal':"act_deal",
                'con_deal':"con_deal",
                'act_table':"act_table",
                'con_table':"con_table",
                # 'predict':"predict"
            },
            agent_reporters={
                'unique_id':'unique_id',
                'deposit':"deposit",
                'cash':"cash",
                'policy':"policy",
                'horizon':"horizon",
                # 'Q_reward':"Q_reward",
                # 'Q_hands':"Q_hands"
            }
        )

    def step(self):
        self.datetime=self.price_list.iloc[self.time]['date']
        print(self.price_list.iloc[self.time]['date'])
        self.settlement=True if self.price_list.iloc[self.time]['date'] in self.settle_list else False
        long=pd.DataFrame({'long':[],'price':[],'agent':[]})
        short=pd.DataFrame({'short':[],'price':[],'agent':[]})
        self.con_table=[copy.deepcopy(long),copy.deepcopy(short)]
        self.act_table=[copy.deepcopy(long),copy.deepcopy(short)]
        self.schedule.step()
        # print(self.con_table)
        # print(self.act_table)
        self.pricing()
        # print(self.predict[0][-1],self.predict[1][-1])
        self.guarding()
        if self.settlement:
            self.delivery()
        print(self.bomb)
        self.datacollector.collect(self)
        self.bomb=0
        self.time+=1      


    def bid(self,position,price, agentid):
        ## 之后想想办法把这个函数写简单一点
        if position[0]>0: #long this month
            temp=self.con_table[0][self.con_table[0]['price']>=price[0]].index.tolist()
            p=temp[-1]+1 if len(temp)!=0 else 0
            df = pd.DataFrame(np.insert(self.con_table[0].values, p, values=[position[0],price[0],agentid], axis=0))
            df.columns=self.con_table[0].columns
            self.con_table[0]=copy.deepcopy(df)
        elif position[0]<0: # short this month
            temp=self.con_table[1][self.con_table[1]['price']>=price[0]].index.tolist()
            p=temp[-1]+1 if len(temp)!=0 else 0
            df = pd.DataFrame(np.insert(self.con_table[1].values, p, values=[position[0],price[0],agentid], axis=0))
            df.columns=self.con_table[1].columns
            self.con_table[1]=copy.deepcopy(df)
        if position[1]>0: #long this month
            temp=self.act_table[0][self.act_table[0]['price']>=price[1]].index.tolist()
            p=temp[-1]+1 if len(temp)!=0 else 0
            df = pd.DataFrame(np.insert(self.act_table[0].values, p, values=[position[1],price[1],agentid], axis=0))
            df.columns=self.act_table[0].columns
            self.act_table[0]=copy.deepcopy(df)
        elif position[1]<0: # short this month
            temp=self.act_table[1][self.act_table[1]['price']>=price[1]].index.tolist()
            p=temp[-1]+1 if len(temp)!=0 else 0
            df = pd.DataFrame(np.insert(self.act_table[1].values, p, values=[position[1],price[1],agentid], axis=0))
            df.columns=self.act_table[1].columns
            self.act_table[1]=copy.deepcopy(df)

    def pricing(self):
        # 买价高于卖价，以买价成交
        this_money=0
        this_hands=0
        p,q=0,0 #long, short
        # print(type(self.con_table[0].iloc[p]['agent'].item()))
        while p<len(self.con_table[0]) and q<len(self.con_table[1]):
            # print(int(self.con_table[0].iloc[p]['agent']),int(self.con_table[1].iloc[q]['agent']))
            if self.con_table[0].iloc[p]['price']<self.con_table[1].iloc[q]['price']:
                q+=1
                continue
            else:
                if self.con_table[0].iloc[p]['long']>=abs(self.con_table[1].iloc[q]['short']):
                    self.agent_list[int(self.con_table[0].iloc[p]['agent'])].deal(-self.con_table[1].iloc[q]['short'],self.con_table[0].iloc[p]['price'],0)
                    self.agent_list[int(self.con_table[1].iloc[q]['agent'])].deal(self.con_table[1].iloc[q]['short'],self.con_table[0].iloc[p]['price'],0)
                    this_money+=abs(self.con_table[1].iloc[q]['short'])*self.con_table[0].iloc[p]['price']
                    this_hands+=abs(self.con_table[1].iloc[q]['short'])
                    self.con_table[0].iloc[p]['long']+=self.con_table[1].iloc[q]['short']
                    self.con_table[1].iloc[q]['short']=0
                    q+=1
                else:
                    self.agent_list[int(self.con_table[0].iloc[p]['agent'])].deal(self.con_table[0].iloc[p]['long'],self.con_table[0].iloc[p]['price'],0)
                    self.agent_list[int(self.con_table[1].iloc[q]['agent'])].deal(-self.con_table[0].iloc[p]['long'],self.con_table[0].iloc[p]['price'],0)
                    this_money+=self.con_table[0].iloc[p]['long']*self.con_table[0].iloc[p]['price']
                    this_hands+=self.con_table[0].iloc[p]['long']
                    self.con_table[1].iloc[q]['short']+=self.con_table[0].iloc[p]['long']
                    self.con_table[0].iloc[p]['long']=0
                    p+=1      
        self.predict[0].append(round(this_money/this_hands,1) if this_hands!=0 else self.price_list.iloc[self.time-1]['con_closing'])

        next_money=0
        next_hands=0
        p,q=0,0 #long, short
        while p<len(self.act_table[0]) and q<len(self.act_table[1]):
            # print(int(self.act_table[0].iloc[p]['agent']),int(self.act_table[1].iloc[q]['agent']))
            if self.act_table[0].iloc[p]['price']<self.act_table[1].iloc[q]['price']:
                q+=1
                continue
            else:
                if self.act_table[0].iloc[p]['long']>=abs(self.act_table[1].iloc[q]['short']):
                    self.agent_list[int(self.act_table[0].iloc[p]['agent'])].deal(abs(self.act_table[1].iloc[q]['short']),self.act_table[0].iloc[p]['price'],1)
                    self.agent_list[int(self.act_table[1].iloc[q]['agent'])].deal(abs(self.act_table[1].iloc[q]['short']),self.act_table[0].iloc[p]['price'],1)
                    next_money+=abs(self.act_table[1].iloc[q]['short'])*self.act_table[0].iloc[p]['price']
                    next_hands+=abs(self.act_table[1].iloc[q]['short'])
                    self.act_table[0].iloc[p]['long']+=self.act_table[1].iloc[q]['short']
                    self.act_table[1].iloc[q]['short']=0
                    q+=1
                else:
                    self.agent_list[int(self.act_table[0].iloc[p]['agent'])].deal(self.act_table[0].iloc[p]['long'],self.act_table[0].iloc[p]['price'],1)
                    self.agent_list[int(self.act_table[1].iloc[q]['agent'])].deal(self.act_table[0].iloc[p]['long'],self.act_table[0].iloc[p]['price'],1)
                    next_money+=self.act_table[0].iloc[p]['long']*self.act_table[0].iloc[p]['price']
                    next_hands+=self.act_table[0].iloc[p]['long']
                    self.act_table[1].iloc[q]['short']+=self.act_table[0].iloc[p]['long']
                    self.act_table[0].iloc[p]['long']=0
                    p+=1      
        self.predict[1].append(round(next_money/next_hands,1) if next_hands!=0 else self.price_list.iloc[self.time-1]['act_closing'])
        self.con_deal,self.act_deal=this_hands,next_hands
        print(this_hands,next_hands)

    def guarding(self):
        for agent in self.agent_list:
            agent.guarding()

    def delivery(self):
        for agent in self.agent_list:
            agent.delivery()        


if __name__=="__main__":
    data=pd.read_csv(r'E:\CCDA-PC\code\futures\futures\data.csv')

    fr=open("E:\CCDA-PC\code\\futures\\futures\settle_date.txt","rb")
    settle_date=pickle.load(fr)

    market=Market(500,data=data,settle_date=settle_date)
    # market.step()
    for _ in range(len(data)-1):
    # for _ in range(300):
        market.step()
    for a in market.agent_list:
        # print(a.Q_reward,a.Q_hands)
        print(a.unique_id,a.cash+a.deposit, a.horizon)
        # for i in range(len(a.Q_reward)):
        #     print(np.unravel_index(a.Q_reward[i].argmax(),a.Q_reward[i].shape))
    print(market.predict)
    file=open("E:\CCDA-PC\code\\futures\\futures\predict1125.txt","wb")
    pickle.dump(market.predict,file)

    agent_data=market.datacollector.get_agent_vars_dataframe()
    agent_data.to_csv("E:\CCDA-PC\code\\futures\\futures\\agent_data1125.csv",index=False)
    model_data=market.datacollector.get_model_vars_dataframe()
    model_data.to_csv("E:\CCDA-PC\code\\futures\\futures\model_data1125.csv",index=False)