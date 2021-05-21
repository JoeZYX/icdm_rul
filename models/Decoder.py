import sys
sys.path.append("..")
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from models.Attention import AttentionLayer,MaskAttention,ProbAttention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 self_attention_layer_types,    # 这里其实self和cross attention的混合类型都是一样的，但是从第二层开始，就不再做self attention了
                 cross_attention_layer_types, 
                 d_model,
                 n_heads,                # 和上面一样，所以需要是 类型个数的倍数
                 d_ff=None,              # 因为attention不改变维度，在最后forward的时候，中间过程的维度
                 dropout=0.1,  
                 activation="relu", 
                 forward_kernel_size = 1, 
                 value_kernel_size=1,
                 causal_kernel_size = 3,
                 output_attention=True):
                 #norm ="layer", 
                 #se_block =False):

        super(DecoderLayer, self).__init__()

        nr_heads_type = len(cross_attention_layer_types)
        heads_each_type = int(n_heads/nr_heads_type)
        d_model_each_type = int(d_model/nr_heads_type)
        
        #self.norm = norm
        #self.se_block = se_block
        self.self_attention_layer_types = self_attention_layer_types
        self.cross_attention_layer_types = cross_attention_layer_types
        
        
        # 第一部分，进行attention
        # ----------- self_attention_layer_types ---------------
        if len(self.self_attention_layer_types)>0:
            self_attention_layer_list = []
            for type_attn in self_attention_layer_types:
                if type_attn=="ProbMask":
                    self_attention_layer_list.append(AttentionLayer(attention = ProbAttention(mask_flag=False, 
                                                                                              factor=5, 
                                                                                              scale=None, 
                                                                                              attention_dropout=dropout,
                                                                                              output_attention=output_attention),
                                                                    input_dim = d_model,
                                                                    output_dim = d_model_each_type, 
                                                                    d_model = d_model_each_type,
                                                                    n_heads = heads_each_type, 
                                                                    causal_kernel_size= causal_kernel_size,
                                                                    value_kernel_size = value_kernel_size,
                                                                    resid_pdrop=dropout)) # 这个思考一下？？？？？？？
                else:
                    self_attention_layer_list.append(AttentionLayer(attention = MaskAttention(mask_typ = type_attn, 
                                                                                              attention_dropout=dropout,
                                                                                              output_attention=output_attention),
                                                                    input_dim = d_model,
                                                                    output_dim = d_model_each_type, 
                                                                    d_model = d_model_each_type,
                                                                    n_heads = heads_each_type, 
                                                                    causal_kernel_size= causal_kernel_size,
                                                                    value_kernel_size = value_kernel_size,
                                                                    resid_pdrop=dropout))

            self.self_attention_layer_list = nn.ModuleList(self_attention_layer_list)
            self.norm1 = nn.BatchNorm1d(d_model)
        else:
            self.self_attention_layer_list = None
            self.norm1 = None         

        # ----------- cross_attention_layer_types ---------------
        cross_attention_layer_list = []
        for type_attn in cross_attention_layer_types:
            if type_attn=="ProbMask":
                cross_attention_layer_list.append(AttentionLayer(attention = ProbAttention(mask_flag=False, 
                                                                                           factor=5, 
                                                                                           scale=None, 
                                                                                           attention_dropout=dropout,
                                                                                           output_attention=output_attention),
                                                                 input_dim = d_model,
                                                                 output_dim = d_model_each_type, 
                                                                 d_model = d_model_each_type,
                                                                 n_heads = heads_each_type, 
                                                                 causal_kernel_size= causal_kernel_size,
                                                                 value_kernel_size = value_kernel_size,
                                                                 resid_pdrop=dropout)) # 这个思考一下？？？？？？？
            else:
                cross_attention_layer_list.append(AttentionLayer(attention = MaskAttention(mask_typ = type_attn, 
                                                                                           attention_dropout=dropout,
                                                                                           output_attention=output_attention),
                                                                 input_dim = d_model,
                                                                 output_dim = d_model_each_type,
                                                                 d_model = d_model_each_type,
                                                                 n_heads = heads_each_type, 
                                                                 causal_kernel_size= causal_kernel_size,
                                                                 value_kernel_size = value_kernel_size,
                                                                 resid_pdrop=dropout))


        self.cross_attention_layer_list = nn.ModuleList(cross_attention_layer_list)
        self.norm2 = nn.BatchNorm1d(d_model)
        
        
        d_ff = d_ff or 2*d_model
        self.forward_kernel_size = forward_kernel_size
        
        
        self.conv1 = nn.Conv1d(in_channels =d_model, 
                               out_channels=d_ff, 
                               kernel_size=self.forward_kernel_size)
        
        self.activation1 = F.relu if activation == "relu" else F.gelu
        
        
        self.conv2 = nn.Conv1d(in_channels = d_ff, 
                               out_channels= d_model, 
                               kernel_size=self.forward_kernel_size)
        
        self.activation2 = F.relu if activation == "relu" else F.gelu

        self.norm3 = nn.BatchNorm1d(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross):
        # self attention
        if len(self.self_attention_layer_types)>0:
            attns = []
            outs = []
            for attn in self.self_attention_layer_list:
                out_value, out_attn = attn(x,x,x)
                attns.append(out_attn)
                outs.append(out_value) 
            new_x =torch.cat(outs,dim=-1) 
            attn = torch.cat(attns,dim=1)
            # 这里是self attention的结果
            # 以下是 第一次 residual add
            x = x + self.dropout(new_x) # B L C  
            x = x.permute(0, 2, 1)
            x = self.norm1(x)
            x = x.permute(0, 2, 1)
            
        # cross attention            
        attns = []
        outs = []
        for attn in self.cross_attention_layer_list:
            out_value, out_attn = attn(x,cross,cross)
            attns.append(out_attn)
            outs.append(out_value) 
        new_x =torch.cat(outs,dim=-1) # gen ju C die jia
        # attention 输出的新value 肯定是 B L C的形状
        attn = torch.cat(attns,dim=1)
        # 这里是cross attention 的结果，也保留的 x 
        x = x + self.dropout(new_x) # B L C  

            
        y = x = self.norm2(x.permute(0, 2, 1))

        forward_padding_size = int(self.forward_kernel_size/2) 
        paddding_y  = nn.functional.pad(y, 
                                        pad=(forward_padding_size, forward_padding_size),
                                        mode='replicate') 
        y = self.dropout(self.activation1(self.conv1(paddding_y)))    


        paddding_y  = nn.functional.pad(y, 
                                        pad=(forward_padding_size, forward_padding_size),
                                        mode='replicate')           
        y = self.dropout(self.activation2(self.conv2(paddding_y)))

        y = self.norm3(x+y).permute(0, 2, 1)  # B L  C 

        return y, attn
    


class Decoder(nn.Module):
    def __init__(self, decoder_layers):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList(decoder_layers)


    def forward(self, x, cross):
        
        for layer in self.decoder_layers:
            x, _ = layer(x, cross)

        return x