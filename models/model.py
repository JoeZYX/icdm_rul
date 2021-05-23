import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Attention import AttentionLayer, MaskAttention, ProbAttention
from models.Embedding import DataEmbedding
from models.Encoder import Encoder,EncoderLayer
from models.Decoder import DecoderLayer,Decoder
from models.Predictor import FullPredictor, LinearPredictor, ConvPredictor

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
                 output_attention = True,
				 

                 predictor_type = "linear",
				 
                 d_layers = 0
                 add_raw  = False):
        
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
        self.predictor_type        = predictor_type
		
        self.d_layers              = d_layers
        self.add_raw               = add_raw
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

        if self.predictor_type == "full":
            self.predictor = FullPredictor(d_model, input_length).double()
        if self.predictor_type == "linear":
            self.predictor = LinearPredictor(d_model).double()
        if self.predictor_type == "conv":
            self.predictor = ConvPredictor(d_model = d_model, pred_kernel = 3).double()
			
        # Decoder 			
        if self.d_layers > 0 :
            if self.add_raw:
                dec_in_dimension = enc_in+1
            else:
                dec_in_dimension = 1
            self.dec_embedding = DataEmbedding(c_in = dec_in_dimension, 
                                               d_model = d_model,
                                               embedd_kernel_size=embedd_kernel_size,
                                               dropout=dropout).double()
											   
            temp_attention_layer_types = attention_layer_types
            decoder_list = []
            for l in range(self.d_layers):
                if l > 0:
                    temp_attention_layer_types = []
                decoder_list.append(DecoderLayer(self_attention_layer_types   = temp_attention_layer_types,
                                                 cross_attention_layer_types  = self.attention_layer_types ,
                                                 d_model                      = self.d_model,
                                                 n_heads                      = self.n_heads,
                                                 d_ff                         = self.d_ff,
                                                 dropout                      = self.dropout,
                                                 activation                   = self.activation,
                                                 forward_kernel_size          = self.forward_kernel_size,
                                                 value_kernel_size            = self.value_kernel_size,
                                                 causal_kernel_size           = self.causal_kernel_size,
                                                 output_attention             = self.output_attention))
            self.decoder = Decoder(decoder_list).double()
            #self.final_predictor = LinearPredictor(d_model).double()
            self.final_predictor = ConvPredictor(d_model).double()


        
    def forward(self, x):
        # x shape 是 batch， L， Enc_in
        
        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out)

        enc_pred = self.predictor(enc_out) # 这里的形状是 【B,L】
        if len(enc_pred.shape)==1:
            enc_pred = torch.unsqueeze(enc_pred, 0)
		
        if self.d_layers > 0:
            dec_in = torch.div(enc_pred,120) #除以最大maxlife
            dec_in = torch.unsqueeze(dec_in, 2)
            if self.add_raw
                dec_in  = torch.cat([x,dec_in],dim=-1)
            dec_embed  = self.dec_embedding(dec_in) 
            dec_out    = self.decoder(dec_embed, enc_out)
            final_pred = self.final_predictor(dec_out)
            if self.output_attention:
                return [enc_pred, final_pred], attns
            else:
                return [enc_pred, final_pred] # [B, L]		
		
        else :
            if self.output_attention:
                return [enc_pred], attns
            else:
                return [enc_pred] # [B, L]
