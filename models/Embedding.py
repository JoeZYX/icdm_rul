import torch
import torch.nn as nn
import math
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

class PositionalEmbedding(nn.Module):
    """
    input shape should be (batch, seq_length, feature_channel)
    
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # -------------------- TODO -----------------  ????? pe.requires_grad = False
        pe = pe.unsqueeze(0)# [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] # select the the length same as input

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embedd_kernel_size=3):
        super(TokenEmbedding, self).__init__()
 
        self.embedd_kernel_size = embedd_kernel_size
        # --------------------- TODO 这里过早融合了多时间序列之间的信息 ---------------------
        self.tokenConv = nn.Conv1d(in_channels=c_in, 
                                   out_channels=d_model, 
                                   kernel_size=self.embedd_kernel_size)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # 因为一维卷积是在最后维度上扫的  需要转换两次
        
        embedd_padding_size   = int(self.embedd_kernel_size/2)
        paddding_x            = nn.functional.pad(x.permute(0, 2, 1), 
                                                  pad=(embedd_padding_size, embedd_padding_size),
                                                  mode='replicate')
        x = self.tokenConv(paddding_x).permute(0, 2, 1)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, 
                 c_in, 
                 d_model, 
                 embedd_kernel_size, 
                 dropout=0.1):
        """
        c_in = input channel number 
        d_model = embedding dimension
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, 
                                              d_model=d_model,
                                              embedd_kernel_size=embedd_kernel_size)
        
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x) 
        # 这里是否要把requires_grad设立为False？
        # -------------------- concat ------------------------------????
        # --------------------- TODO check the difference -------------------------
        # x =  self.value_embedding(x) + Variable(self.position_embedding(x),requires_grad = False) 
        return self.dropout(x)

def check_the_posencoding(c_in = 1, d_model = 100, embedd_kernel_size =3,dropout=0, length =1000):
    embedding =  DataEmbedding(c_in, 
                               d_model,
                               embedd_kernel_size,
                               dropout).double()
    
    input_ = torch.tensor(np.zeros((1,length,c_in))) #batch length channel
    embedding_out = embedding(input_)
    plt.figure(figsize=(15,5))
    sns.heatmap(embedding_out.detach().numpy()[0], linewidth=0)
