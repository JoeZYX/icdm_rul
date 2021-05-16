import torch
import torch.nn as nn
import math
from math import sqrt
import numpy as np
import torch.nn.functional as F
import random


####################### Mask #########################################

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            mask = torch.ones((L, L)).to(device)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)  
            
    @property   
    def mask(self):
        return self._mask

class LocalMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            window_size = math.ceil(1.2*np.log2(L))   
            # 这里如何设置local的大小？？？？      
            mask = torch.ones((L, L)).to(device)
            mask = torch.triu(mask,-window_size).T
            mask = torch.triu(mask,0).T        
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device) 
    @property    
    def mask(self):
        return self._mask

class LocalLogSymmetryMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            mask = torch.zeros((L, L), dtype=torch.float).to(device)
            for i in range(L):
                mask[i] = self.row_mask(i, L)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)

            
    def row_mask(self,index, L):
        local_window_size = math.ceil(np.log2(L)/2) # 1/2 window size
        # 对当前行进行初始化
        mask = torch.zeros((L), dtype=torch.float)

        if((index - local_window_size + 1) < 0):
            mask[:index] = 1 # Local attention
        else:
            mask[index - local_window_size + 1:(index + 1)] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index - local_window_size + 1 - int(1.5**i)
                if new_index >= 0:
                    mask[new_index] = 1
                else:
                    break
                    
        if ((index + local_window_size-1 )>=L):
            mask[index:] = 1 
        else:
            mask[index:index+local_window_size] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index + local_window_size-1 +int(1.5**i)
                if new_index < L:
                    mask[new_index] = 1
                else:
                    break
        return mask               

    @property          
    def mask(self):
        return self._mask


class LocalSymmetryMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            window_size = math.ceil(1.2*np.log2(L)/2)  #halb
            mask = torch.ones((L, L)).to(device)
            mask = torch.triu(mask,-window_size).T
            mask = torch.triu(mask,-window_size)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)  
    @property            
    def mask(self):
        return self._mask

class LocalLogMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            mask = torch.zeros((L, L), dtype=torch.float).to(device)
            for i in range(L):
                mask[i] = self.row_mask(i, L)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)
            
    def row_mask(self,index, L):
        local_window_size = math.ceil(np.log2(L)) # window size
        # 对当前行进行初始化
        mask = torch.zeros((L), dtype=torch.float)

        if((index - local_window_size + 1) < 0):
            mask[:index] = 1 # Local attention
        else:
            mask[index - local_window_size + 1:(index + 1)] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index - local_window_size + 1 - int(1.5**i)
                if new_index >= 0:
                    mask[new_index] = 1
                else:
                    break

        return mask   

    @property    
    def mask(self):
        return self._mask


Mask_dict = {"Triangular"     :TriangularCausalMask,
             "LocalLog"       :LocalLogMask,
             "LocalSymmetry"  :LocalSymmetryMask,
             "Full"           :FullMask,
             "Local"          :LocalMask,
             "LocLogSymmetry" :LocalLogSymmetryMask}




###################### Mask Attention ####################
class MaskAttention(nn.Module):
    def __init__(self, mask_flag=True, 
                 mask_typ = "Triangular",
                 attention_dropout=0.1, 
                 output_attention=False):
        """
        任务就是通过 获知使用哪种mask，进行不同部位的attention
        dropout  是对分数的dropout
        
        """
        super(MaskAttention, self).__init__()
        self.mask_typ = mask_typ
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        """
        queries : [Batch, Length, Heads, E]
        keys    : [Batch, Length, Heads, E]
        values  : [Batch, Length, Heads, D]


        返回的是两个东西
        1.  新的value  格式依旧是 [Batch, Length, Heads, D]
        2.  attention 的map
        """
        B, L, H, E = queries.shape
        _, _, _, D = values.shape
        #print(".............................",queries.device)
        scale =  1./math.sqrt(E) #每个head的dimension

        queries = queries.permute(0, 2, 1, 3)  # [batch, heads, length, chanell]
        keys = keys.permute(0, 2, 3, 1)  # [batch, heads, chanell, length]
        scores = torch.matmul(queries, keys)
        # scores1 = torch.einsum("blhe,bshe->bhls", queries, keys) 和这个是一样的
        scores = scale * scores
        
        if self.mask_flag:
            #print(self.mask_typ)           
            attn_mask = Mask_dict[self.mask_typ](B, L, device=queries.device)
            #print("....................",attn_mask.mask.device)
            scores.masked_fill_(attn_mask.mask, -np.inf) #其实就是把不想要的地方设为True，然后再在这些地方加上 -inf

        pre_att = self.dropout(torch.softmax(scores , dim=-1))
        
        values = values.permute(0, 2, 1, 3)# [batch, heads, length, chanell]
        attn_values = torch.matmul(pre_att, values).permute(0,2,1,3) #[batch, length, heads, chanell]
        # 这里和torch.einsum("bhls,bshd->blhd", att, values) 结果也是一样的
        
        if self.output_attention:
            return (attn_values.contiguous(), pre_att)
        else:
            return (attn_values.contiguous(), None)



#################### Prob Attention ###############################
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        # L_K 是肯定等于 L_Q的
        # sample_k 是为了算M 要采样的K的个数
        # n_top 选多少个Q
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        
        
        #index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        sample_list = []
        for i in range(L_Q):
            index_list = list(np.arange(L_Q))
            #shuffle(index_list)
            sample_list.append(random.sample(index_list,sample_k))
        sample_array = np.array(sample_list)
        index_sample = torch.tensor(sample_array,dtype=torch.long)        
        
        
        
        
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        #print(M_top)

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k
        #print(Q_K)

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        #print("................................",V.device)

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V)
        if self.output_attention:
            #这里和context_in是一样的
            attns = (torch.ones([B, H, L_V, L_V])/L_V).double().to(attn.device)
            attns[torch.arange(B)[:, None, None], 
                  torch.arange(H)[None, :, None], 
                  index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        # L_Q 和 L_K 是肯定相等的
        assert L_Q==L_K

        #queries = queries.view(B, H, L_Q, -1)
        #keys = keys.view(B, H, L_K, -1)
        #values = values.view(B, H, L_K, -1)

        keys = keys.permute(0, 2, 1, 3) 
        queries = queries.permute(0, 2, 1, 3) 
        values = values.permute(0, 2, 1, 3) 
        # U_part 也是肯定等于 u的 这里需要检查 U_part不能大于L_K
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        if U_part>L_K:
            #print(".")
            U_part = L_K
            u = L_K
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)
        context = context.permute(0,2,1,3)
        
        return context.contiguous(), attn


###################  AttentionLayer #########################

class AttentionLayer(nn.Module):
    def __init__(self, attention, 
                 input_dim, 
                 output_dim,
                 d_model, 
                 n_heads, 
                 d_keys=None,
                 d_values=None,
                 causal_kernel_size=3, 
                 value_kernel_size = 1,
                 resid_pdrop = 0.1):
        # d_model 在这里仅仅代表 隐藏层的总的维度， 它会被分为head份
        """
        这个就是transformer encoder中的中间部分，虽然我想做的是混合，但是混合不在这里，这里只进行一种类型的attention
        这个类的输入是x,然后基于x计算Q K V,然后再基于KQV计算attention
        输入的尺寸是 【batch， length， channel】
        输出就是新的 value
        理论上 input_dim = output_dim 因为在attention完事之后，是需要加上residual的  如果尺寸不一样  怎么加？
        attention参数 是告诉采取哪一种注意力
        
        TODO 思考
        causal convolutions to produce queries and keys in the self attention layer
        
        
        """
        super(AttentionLayer, self).__init__()
        # 每个heads的维度
        self.d_keys = d_keys or (d_model//n_heads)
        # 每个heads value的维度
        self.d_values = d_values or (d_model//n_heads)
        # 多少个头？
        self.n_heads = n_heads
        # 因为是时间序列，这里采取的causal attention
        self.causal_kernel_size = causal_kernel_size
        self.value_kernel_size  = value_kernel_size
        
        self.query_projection = nn.Conv1d(in_channels = input_dim, 
                                          out_channels = self.d_keys*self.n_heads, 
                                          kernel_size  = self.causal_kernel_size)
        
        self.key_projection = nn.Conv1d(in_channels = input_dim, 
                                       out_channels = self.d_keys*self.n_heads, 
                                       kernel_size  = self.causal_kernel_size)
        
        self.value_projection = nn.Conv1d(in_channels=input_dim, 
                                          out_channels=self.d_values * self.n_heads, 
                                          kernel_size = self.value_kernel_size) 
        # value 是只基于当前的t计算的 所以kernel=1
        
        self.inner_attention = attention
        
 
        # 这里需要改变kernel size
        self.out_projection = nn.Conv1d(in_channels=self.d_values * self.n_heads, 
                                        out_channels=output_dim, 
                                        kernel_size = self.value_kernel_size) 
        self.activation = F.relu
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, queries, keys, values):
        """
        input x : [batch, length, in_channel ]  in_channel=input_dim
        return y : [batch, length, output_dim]
        """
        B, L_Q, I_Q = queries.shape
        _, L_K, I_K = keys.shape
        _, L_V, I_V = values.shape
        H = self.n_heads        
        #value projection
        value_padding_size   = int(self.value_kernel_size/2)
        paddding_values      = nn.functional.pad(values.permute(0, 2, 1), 
                                                 pad=(value_padding_size, value_padding_size),
                                                 mode='replicate')
        values               = self.value_projection(paddding_values).permute(0, 2, 1)  # B L C
        
        # query projection
        queries_padding_size = int(self.causal_kernel_size/2)
        paddding_queries     = nn.functional.pad(queries.permute(0, 2, 1), 
                                                 pad=(queries_padding_size, queries_padding_size),
                                                 mode='replicate')
        queries              = self.query_projection(paddding_queries).permute(0, 2, 1) # B L C

        
        paddding_keys        = nn.functional.pad(keys.permute(0, 2, 1), 
                                                 pad=(queries_padding_size, queries_padding_size),
                                                 mode='replicate')
        keys                 = self.key_projection(paddding_keys).permute(0, 2, 1) # B L C  
        
        # 以上 B L C 中的C是包含了所有Head的特征
        # 这里希望Q K V都保持一样的形状， 也就是 （B， L，H， -1）
        query = queries.view(B, L_Q, H, -1)
        key = keys.view(B, L_K, H, -1)
        values = values.view(B, L_V, H, -1)

        # out 是 新算出来的值  尺寸应该是 【 B，L，headsxchannel】但是其实是 B L H C
        # attn 是 attn 的 map
        out, attn = self.inner_attention(query,
                                         key,
                                         values)
        
        # attention 的输出固定格式为 B L H C ,所以要合并
        out = out.view(B, L_Q, -1)
        paddding_out   = nn.functional.pad(out.permute(0, 2, 1), 
                                           pad=(value_padding_size, value_padding_size),
                                           mode='replicate')
        
        
        out = self.activation(self.out_projection(paddding_out)).permute(0, 2, 1)
        
        out = self.resid_dropout(out)

        return out, attn
