# -*- coding: utf-8 -*-

import torch
from torch import nn

class ACRNN(nn.Module):
    def channel_wise(self):
        self.r = int(self.C / self.ratio)
        self.fc = nn.Sequential(
            nn.Linear(self.C, self.r),
            nn.Tanh(),
            nn.Linear(self.r, self.C)
        )
        self.softmax1 = nn.Softmax(dim=-1)
    
    def CNN(self):
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.k, (self.kernel_height, self.kernel_width), self.kernel_stride),
            nn.BatchNorm2d(self.k),
            nn.ELU(),
            nn.MaxPool2d((1, self.pooling_width), self.pooling_stride),
            nn.Dropout(p=0.5)
        )
    
    def LSTM(self):
        self.lstm = nn.LSTM(
            input_size=self.k * 28,
            hidden_size=self.hidden,
            num_layers=2,
            batch_first=True
        )
    
    def inside(self):
        self.W1 = nn.Parameter(torch.rand(size=(self.hidden, self.hidden_attention)))
        self.W2 = nn.Parameter(torch.rand(size=(self.hidden, self.hidden_attention)))
        self.b = nn.Parameter(torch.zeros(self.hidden_attention))
        self.activation = nn.Softmax(dim=-1)
        self.vector = nn.Linear(self.hidden, self.hidden)
    
    def Self_attention(self):
        self.self_attention = nn.Sequential(
            
            nn.Linear(self.hidden_attention, self.hidden),
            nn.ELU()
        )
        self.softmax2 = nn.Softmax(dim=2)
        self.dropout2 = nn.Dropout(p=0.5)
    
    def __init__(self, reduce, k):
        super(ACRNN, self).__init__()
        self.C = 32
        self.W = 384
        self.ratio = reduce
        self.k = k
        self.kernel_height = self.C
        self.kernel_width = 40
        self.kernel_stride = 1
        self.pooling_width = 75
        self.pooling_stride = 10
        self.hidden = 64
        self.hidden_attention = 512
        self.num_labels = 2
        
        self.channel_wise()
        self.CNN()
        self.LSTM()
        self.inside()
        self.Self_attention()
        
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden, self.num_labels),
            nn.Softmax(dim=1)
        )
        
        self.mean_pool = nn.AdaptiveAvgPool1d(1)  # [n, C, 384] â†?[n, C, 1]

    def forward(self, x):
        # x: [n, C, W] = [n, 32, 384]
        x1 = self.mean_pool(x)  # [n, C, 1]
        x1 = x1.view(x.size(0), -1)  # [n, C]
        
        feature_pre = self.fc(x1)   # [n, C]
        v = self.softmax1(feature_pre)  # [n, C]
        
        vr = v.unsqueeze(-1).repeat(1, 1, self.W)  # [n, C, W]
        x = x * vr  # channel-wise attention

        x = x.unsqueeze(1)  # [n, 1, C, W]
        x = self.conv(x)    # [n, 40, 1, 28]  
        
        x = x.reshape(x.size(0), 1, -1)  # [n, 1, k*T'] = [n, 1, 40*28]

        h, _ = self.lstm(x)  # [n, 1, hidden]

        y = self.vector(h)  # [n, 1, hidden]
        
        y = self.activation(torch.matmul(h, self.W1) + torch.matmul(y, self.W2) + self.b)

        z = self.self_attention(y)
        p = z * h
        p = self.softmax2(p)
        
        A = p * h
        A = A.reshape(-1, self.hidden)
        A = self.dropout2(A)
        
        x = self.softmax(A)
        return x


