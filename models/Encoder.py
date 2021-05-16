import sys
sys.path.append("..")
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from models.Attention import AttentionLayer,MaskAttention,ProbAttention

###################### SE Block #############################################
class SELayer(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear_reduction =  nn.Linear(in_channel, 
                                           in_channel // reduction, 
                                           bias=False)
        self.activation1 = F.relu
        self.linear_backtransform =  nn.Linear(in_channel // reduction, 
                                               in_channel,
                                               bias=False)
        self.activation2 = torch.sigmoid
      

    def forward(self, x):
        # x size = Batch Length Channel
        b, l, c, = x.size()
        x = x.permute(0,2,1) # b c l
        y = self.avg_pool(x).view(b, c)
        y = self.linear_reduction(y)
        y = self.activation1(y)
        y = self.linear_backtransform(y)
        y =self.activation2(y).view(b, c, 1).expand_as(x)
        y =x * y
        return y.permute(0,2,1)

###################### Transformer Encoder ################################ 
class EncoderLayer(nn.Module):
    def __init__(self, 
                 attention_layer_types,  # attention的种类list 
                 d_model,                # 最终的维度，如果attention类型有多个，那么要平均分给每个种类
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
        
        super(EncoderLayer, self).__init__()
        """
        attention_layer_list : 一个list的AttentionLayer的实例， 
        AttentionLayer做的工作就是 返回新的value， 尺寸的和 应该和 d_model 一致
        
        d_model 是整个transfommer 输出  输入的 尺寸  因为要经历两次residual 的相加
        
        d_ff 类似bottleneck的感觉  可大可小， 如果没有被定义的话，那么就是dmodel的两倍
        """
        
        # --------------- TODO  ---------------- 是否需要add redidual
        # 当 input 和 output的尺寸不一样的时候 就不需要加

        # 这里根据注意力种类的个数，计算每个的outdim
        # d_model / nr_heads_type / heads_each_type = 实际每个head的dimension
        # d_model一定要是 总head 的倍数
        nr_heads_type = len(attention_layer_types)
        heads_each_type = int(n_heads/nr_heads_type)
        d_model_each_type = int(d_model/nr_heads_type)
        
        self.norm = norm
        self.se_block = se_block
        
        # 第一部分，进行attention
        attention_layer_list = []
        for type_attn in attention_layer_types:
            if type_attn=="ProbMask":
                attention_layer_list.append(AttentionLayer(attention = ProbAttention(mask_flag=False, 
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
                attention_layer_list.append(AttentionLayer(attention = MaskAttention(mask_typ = type_attn, 
                                                                                     attention_dropout=dropout,
                                                                                     output_attention=output_attention),
                                                           input_dim = d_model,
                                                           output_dim = d_model_each_type, # number * (d_model/heads)
                                                           d_model = d_model_each_type,
                                                           n_heads = heads_each_type, # part heads
                                                           causal_kernel_size= causal_kernel_size,
                                                           value_kernel_size = value_kernel_size,
                                                           resid_pdrop=dropout))
 


        self.attention_layer_list = nn.ModuleList(attention_layer_list)

        # ----------------------- 这里是自己加的 se_block --------------- 是为了平衡每个head之间的关系
        #if self.se_block:
        #    self.se = SELayer(d_model)
            
            
        # 这里的输出 为 B L d_model
        # 在加上residual之后 要norm一次
        # ------------------这里是我自己加的 batchnorm ---------------------
        if self.norm == "layer":
            self.norm1 = nn.LayerNorm(d_model)
        else:
            self.norm1 = nn.BatchNorm1d(d_model)
            
            
            
            
        # 第二部分，进行         
        d_ff = d_ff or d_model*2
        self.forward_kernel_size = forward_kernel_size
        
        self.conv1 = nn.Conv1d(in_channels=d_model, 
                               out_channels=d_ff, 
                               kernel_size=self.forward_kernel_size)
        
        self.activation1 = F.relu if activation == "relu" else F.gelu
        
        
        self.conv2 = nn.Conv1d(in_channels=d_ff, 
                               out_channels=d_model, 
                               kernel_size=self.forward_kernel_size)
        self.activation2 = F.relu if activation == "relu" else F.gelu
        
        
        if self.norm == "layer":
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm2 = nn.BatchNorm1d(d_model)

        
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : [B,L,Channel]
        """
        # 第一部分 attetion + residual + layer norm
        # -------------   需要这个residual吗？我觉得不需要   -----------------
        attns = []
        outs = []
        for attn in self.attention_layer_list:
            out_value, out_attn = attn(x,x,x)
            attns.append(out_attn)
            outs.append(out_value) 
        new_x =torch.cat(outs,dim=-1) # gen ju C die jia
        # attention 输出的新value 肯定是 B L C的形状
        attn = torch.cat(attns,dim=1)
        #if self.se_block:
        #    new_x = self.se(new_x)

        x = x + new_x # B L C
        
        forward_padding_size = int(self.forward_kernel_size/2)
        
        if self.norm == "layer": 
            # layer normalization informer 
            y = x = self.norm1(x) # B L C
            
            paddding_y  = nn.functional.pad(y.permute(0, 2, 1), 
                                            pad=(forward_padding_size, forward_padding_size),
                                            mode='replicate')
            y = self.dropout(self.activation1(self.conv1(y)))
            
            paddding_y  = nn.functional.pad(y, 
                                            pad=(forward_padding_size, forward_padding_size),
                                            mode='replicate')
            
            y = self.dropout(self.activation2(self.conv2(y).permute(0, 2, 1)))

            y = self.norm2(x+y)
        else:
            # batch normalizaton 输入Shape：（N, C）或者(N, C, L)
            y = x = self.norm1(x.permute(0, 2, 1)) # B C L zhe yang shi bu hui bian de 
            
            paddding_y  = nn.functional.pad(y, 
                                            pad=(forward_padding_size, forward_padding_size),
                                            mode='replicate') 
            y = self.dropout(self.activation1(self.conv1(y)))    
            
            
            paddding_y  = nn.functional.pad(y, 
                                            pad=(forward_padding_size, forward_padding_size),
                                            mode='replicate')           
            y = self.dropout(self.activation2(self.conv2(y)))
            
            y = self.norm2(x+y).permute(0, 2, 1)  # B L  C 
            
        return y, attn
    
    
################################# Model Backbone #############################
class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(encoder_layers)

        #self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        attns = []

        for encoder_layer in self.encoder_layers:
            x, attn = encoder_layer(x)
            attns.append(attn)


        return x, attns
