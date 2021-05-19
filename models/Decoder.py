import sys
sys.path.append("..")
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from models.Attention import AttentionLayer,MaskAttention,ProbAttention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 self_attention_layer_types, 
                 cross_attention_layer_types, 
                 n_heads,                # 和上面一样，所以需要是 类型个数的倍数
                 d_ff=None,              # 因为attention不改变维度，在最后forward的时候，中间过程的维度
                 dropout=0.1,  
                 activation="relu", 
                 forward_kernel_size = 1, 
                 value_kernel_size=1,
                 causal_kernel_size = 3,
                 output_attention=True, 
                 norm ="layer", 
                 se_block =False):

        super(DecoderLayer, self).__init__()
		
		
        nr_heads_type = len(self_attention_layer_types)
        heads_each_type = int(n_heads/nr_heads_type)
        d_model_each_type = int(d_model/nr_heads_type)
        self.norm = norm
        self.se_block = se_block	
		
        # 第一部分，进行attention
        self_attention_layer_list = []
        for type_attn in self_attention_layer_types:
            if type_attn=="ProbMask":
                self_attention_layer_list.append(AttentionLayer(attention = ProbAttention(mask_flag=False, 
                                                                                          factor=5, 
                                                                                          scale=None, 
                                                                                          attention_dropout=dropout,
                                                                                          output_attention=output_attention),
                                                                input_dim = d_model,
                                                                output_dim = d_model_each_type, # number * (d_model/heads)
                                                                d_model = d_model_each_type,
                                                                n_heads = heads_each_type, # part heads
                                                                causal_kernel_size= causal_kernel_size,
                                                                value_kernel_size = value_kernel_size,
                                                                resid_pdrop=dropout)) # 这个思考一下？？？？？？？
            else:
                self_attention_layer_list.append(AttentionLayer(attention = MaskAttention(mask_typ = type_attn, 
                                                                                          attention_dropout=dropout,
                                                                                          output_attention=output_attention),
                                                                input_dim = d_model,
                                                                output_dim = d_model_each_type, # number * (d_model/heads)
                                                                d_model = d_model_each_type,
                                                                n_heads = heads_each_type, # part heads
                                                                causal_kernel_size= causal_kernel_size,
                                                                value_kernel_size = value_kernel_size,
                                                                resid_pdrop=dropout))
		
        self.self_attention = self_attention
        self.cross_attention = cross_attention		
		


	
		

        d_ff = d_ff or 2*d_model

		
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x