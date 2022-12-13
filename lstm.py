import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import copy


def generate_df_affect_by_n_days(series, n, index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = pd.DataFrame()
    for i in range(n):
        df['c%d' % i] = series.tolist()[i:-(n - i)]
    df['y'] = series.tolist()[n:]
    if index:
        df.index = series.index[n:]
    return df


def readData(df, column='ratio', n=30, all_too=True, index=False, train_end=-30):
    # df = pd.read_csv(r"E:\CCDA-PC\code\futures\various\cotton\cotton.csv")

    df.index=df['date'].values.tolist()
    # df.index = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), df.index))
    df_column = df[column].copy()
    df_column_train, df_column_test = df_column[:train_end], df_column[train_end - n:]
    df_generate_from_df_column_train = generate_df_affect_by_n_days(df_column_train, n, index=index)
    if all_too:
        return df_generate_from_df_column_train, df_column, df.index.tolist()
    return df_generate_from_df_column_train


class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data):
        # 定义好 image 的路径
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


n = 30
LR = 0.0001
EPOCH = 100
train_end = -30
# 数据集建立
data=pd.read_csv(r"E:\CCDA-PC\code\futures\server_various\various\silver\silver.csv")
temp=[(data.iloc[i]['con_settlement']-data.iloc[i-1]['con_settlement'])*100/data.iloc[i-1]['con_settlement'] for i in range(1,len(data))]
temp.insert(0,0)
data['ratio']=temp
print(data['ratio'])
df, df_all, df_index = readData(data[1:],'ratio', n=n, train_end=train_end)

df_all = np.array(df_all.tolist())
plt.plot(df_index, df_all, label='real-data')

df_numpy = np.array(df)

df_numpy_mean = np.mean(df_numpy)
df_numpy_std = np.std(df_numpy)

df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std
df_tensor = torch.Tensor(df_numpy)

trainset = TrainSet(df_tensor)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

# rnn = torch.load('rnn.pkl')

rnn = RNN(n)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

for step in range(EPOCH):
    for tx, ty in trainloader:
        output = rnn(torch.unsqueeze(tx, dim=0))
        loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()
    print(step, loss)
    if step % 10:
        torch.save(rnn, 'rnn.pkl')
torch.save(rnn, 'rnn.pkl')
#
generate_data_train = []
generate_data_test = []

test_index = len(df_all) + train_end

df_all_normal = (df_all - df_numpy_mean) / df_numpy_std
df_all_normal_tensor = torch.Tensor(df_all_normal)
for i in range(n, len(df_all)):
    x = df_all_normal_tensor[i - n:i]
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
    y = rnn(x)
    if i < test_index:
        generate_data_train.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
    else:
        generate_data_test.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
print(generate_data_test,type(generate_data_test),len(generate_data_test))

# plt.plot(df_index[n:train_end], generate_data_train, label='generate_train')
# plt.plot(df_index[train_end:], generate_data_test, label='generate_test')
# plt.legend()
# plt.show()
# plt.cla()
# plt.plot(df_index[train_end:-30], df_all[train_end:-30], label='real-data')
# plt.plot(df_index[train_end:-30], generate_data_test[:-30], label='generate_test')
# plt.legend()
# plt.show()

data=data.iloc[-30:,:]
generate_data_test=generate_data_test[-30:]
con=data['con_settlement'].values.tolist()
lstm_pre=[]
for i in range(len(generate_data_test)):
    lstm_pre.append(con[-30+i]*(generate_data_test[i]+100)/100)
print(con)
print(lstm_pre)
mae,mape,smape,mse,rmse,msle=0,0,0,0,0,0
for i in range(30):
    mae+=abs(lstm_pre[i]-con[-30+i])
    mape+=abs(con[-30+i]-lstm_pre[i])*100/con[-30+i]
    smape+=abs(lstm_pre[i]-con[-30+i])/((lstm_pre[i]+con[-30+i])/2)
    mse+=(lstm_pre[i]-con[-30+i])**2
    msle+=(np.log(1+con[-30+i])-np.log(1+lstm_pre[i]))**2
print(mae/30,mape/30,smape/30,mse/30,(mse/30)**0.5,msle/30)