import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Attention import AttentionLayer, MaskAttention, ProbAttention
from models.Embedding import DataEmbedding
from models.Encoder import Encoder,EncoderLayer

class TStransformer(nn.Module):
    def __init__(self, 
                 enc_in, 
                 input_length, 
                 c_out, 
                 d_model=512, 
                 attention_layer_types=["Triangular"],
                 
                 embedd_kernel_size = 3, 
                 forward_kernel_size =1,
                 value_kernel_size = 1,
                 causal_kernel_size=3, 
                 
                 d_ff =None,
                 n_heads=8, 
                 e_layers=3,  
                 dropout=0.1, 
                 norm = "batch", 
                 se_block = False,
                 activation='relu', 
                 output_attention = True):
        
        """
        enc_in : 输入给encoder的channel数，也就是最开始的channel数, 这个通过dataloader获得
        input_length: 数据的原始长度，最后预测的时候要用，也要从dataloader获得
        c_out ： 最后的输出层，这里应该等于原始输入长度
        --------------------------- TODO 是加还是concant？？---------------------
        d_model：每一层encoding的数量 ，这个数基本不变，因为在transofomer 中的相加 residual， d_model 就不变化
        attention_layer_types 一个list 包含那些attention的类型       
        n_heads 总共attention多少个头，目前大概是三的倍数
        e_layers： 多少层encoder
        
        
        """
        super(TStransformer, self).__init__()

        self.enc_in             = enc_in
        self.d_model            = d_model
        self.embedd_kernel_size = embedd_kernel_size
        self.dropout            = dropout
        
        self.attention_layer_types = attention_layer_types
        self.n_heads               = n_heads
        self.d_ff                  = d_ff
        self.activation            = activation
        self.forward_kernel_size   = forward_kernel_size
        self.value_kernel_size     = value_kernel_size
        self.causal_kernel_size    = causal_kernel_size
        self.norm                  = norm
        self.output_attention      = output_attention
        self.se_block              = se_block
        self.e_layers              = e_layers
        
        self.input_length          = input_length
        self.c_out                 = c_out
        # Encoding
        
        self.enc_embedding = DataEmbedding(c_in = enc_in, 
                                           d_model = d_model,
                                           embedd_kernel_size=embedd_kernel_size,
                                           dropout=dropout).double()

        
        
        # Encoder        
        self.encoder = Encoder([EncoderLayer(attention_layer_types = self.attention_layer_types,
                                             d_model               = self.d_model,
                                             n_heads               = self.n_heads,
                                             d_ff                  = self.d_ff,
                                             dropout               = self.dropout,
                                             activation            = self.activation,
                                             forward_kernel_size   = self.forward_kernel_size,
                                             value_kernel_size     = self.value_kernel_size,
                                             causal_kernel_size    = self.causal_kernel_size,
                                             output_attention      = self.output_attention,
                                             norm                  = self.norm,
                                             se_block              = self.se_block) for l in range(self.e_layers)]
                               ).double()
        # 这里的输出是 （B， L, d_model） 
        # self.decoder = ??????????????


        self.donwconv1 = nn.Conv1d(in_channels = d_model, 
                                   out_channels = int(d_model/2), 
                                   kernel_size  = 3).double() # 这里是因为最后输出是每一个时间步骤为1
        self.activation1 = F.relu
        self.norm1       = nn.BatchNorm1d(int(d_model/2))
        

        self.donwconv2 = nn.Conv1d(in_channels = int(d_model/2), 
                                   out_channels = 1, 
                                   kernel_size  = 3).double() # 这里是因为最后输出是每一个时间步骤为1    
        self.activation2 = F.relu
        self.norm2       = nn.BatchNorm1d(1)

        self.predict = nn.Linear(in_features=self.input_length, 
                                 out_features=self.c_out).double() #这里可以换成kernel比较大的conv1d
								 
        #self.predict = nn.Conv1d(in_channels = 1,
        #                         out_channels = 1,
        #                         kernel_size  = 5).double()




        
        


        
    def forward(self, x):
        
        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out)
        #print(enc_out.shape)
        paddding_enc_out  = nn.functional.pad(enc_out.permute(0, 2, 1), 
                                              pad=(1, 1),
                                              mode='replicate')
        
        enc_out = self.norm1(self.activation1(self.donwconv1(paddding_enc_out)))
        
        paddding_enc_out  = nn.functional.pad(enc_out,
                                              pad=(1, 1),
                                              mode='replicate')       
        

        enc_out = self.norm2(self.activation2(self.donwconv2(paddding_enc_out))).permute(0,2,1).squeeze()#.permute(0,2,1).squeeze()
        # Conv 1d prediction       
        #paddding_enc_out  = nn.functional.pad(enc_out,
        #                                      pad=(2, 2),
        #                                      mode='replicate')   
											  
        #enc_out = self.predict(paddding_enc_out).permute(0,2,1).squeeze()

        # fully layer prediction 											  
        enc_out = self.predict(enc_out) 

        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out # [B, L, D]
