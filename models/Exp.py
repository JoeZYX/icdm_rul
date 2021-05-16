import sys
sys.path.append("..")
import os 
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import scipy.stats as stats
import math
from models.dataloader import CMAPSSData
from models.model import TStransformer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from time import time

class Weighted_MSE_Loss(nn.Module):
    
    def __init__(self, seq_length=40, sigma_faktor=1.5, device = "cuda"):
        super(Weighted_MSE_Loss, self).__init__()
        self.seq_length = seq_length
        self.sigma      = seq_length/sigma_faktor
        
        x = np.linspace(1, seq_length, seq_length)
        #mu = self.seq_length/2
        mu = 0
        y = stats.norm.pdf(x, mu, self.sigma)
        y = 2*np.max(y)-y
        y = y/np.sum(y)*seq_length
        plt.plot(x, y)
        plt.show()
        with torch.no_grad():
            self.weights = torch.Tensor(y).double().to(device)  
        #self.weights = torch.Tensor([1]*seq_length).double().to(device)  
        
    def forward(self, pred, target):
        se  = (pred-target)**2
        out = se * self.weights.expand_as(target)
        loss = out.mean() 
        return loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
  
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("new best score!!!!")
            self.best_score = score
            self.save_checkpoint(val_loss, model,path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'{}_checkpoint.pth'.format(str(val_loss)))
        self.val_loss_min = val_loss
        
        
class adjust_learning_rate_class:
    def __init__(self, args, verbose):
        self.patience = args.learning_rate_patience
        self.factor   = args.learning_rate_factor
        self.learning_rate = args.learning_rate
        self.args = args
        self.verbose = verbose
        self.val_loss_min = np.Inf
        self.counter = 0
        self.best_score = None
    def __call__(self, optimizer, val_loss):
        # val_loss 是正值，越小越好
        # 但是这里加了负值，score愈大越好
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.counter += 1
        elif score <= self.best_score :
            self.counter += 1
            if self.verbose:
                print(f'Learning rate adjusting counter: {self.counter} out of {self.patience}')
        else:
            if self.verbose:
                print("new best score!!!!")
            self.best_score = score
            self.counter = 0
            
        if self.counter == self.patience:
            self.learning_rate = self.learning_rate * self.factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                if self.verbose:
                    print('Updating learning rate to {}'.format(self.learning_rate))
            self.counter = 0


class Exp_TStransformer(object):
    def __init__(self, args):
        self.args = args
        # args.use_gpu
        # args.gpu

        self.device = self._acquire_device()
        
        if self.args.flag == "train":
        
            self.train_data, self.train_loader = self._get_data(flag = 'train')
            self.vali_data , self.vali_loader  = self._get_data(flag = "val")
            self.input_dimension = self.train_data.data_channel
            
        if self.args.flag == "test":
            
            self.test_data, self.test_loader = self._get_data(flag = 'test')   
            self.input_dimension = self.test_data.data_channel
        
        

        self.model = self._build_model().to(self.device)
        
        self.optimizer_dict = {"Adam":optim.Adam}
        self.criterion_dict = {"MSE":nn.MSELoss,"CrossEntropy":nn.CrossEntropyLoss,"WeightMSE":Weighted_MSE_Loss}



    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    
    def _build_model(self):
        if not self.args.late:
            print("build not late")
            model = TStransformer(enc_in                = self.input_dimension,
                                  input_length          = self.args.sequence_length,
                                  c_out                 = self.args.sequence_length,
                                  d_model               = self.args.d_model,
                                  attention_layer_types = self.args.attention_layer_types,
                                  embedd_kernel_size    = self.args.embedd_kernel_size,
                                  forward_kernel_size   = self.args.forward_kernel_size,
                                  value_kernel_size     = self.args.value_kernel_size,
                                  causal_kernel_size    = self.args.causal_kernel_size,
                                  d_ff                  = self.args.d_ff,
                                  n_heads               = self.args.n_heads,
                                  e_layers              = self.args.e_layers,
                                  dropout               = self.args.dropout,
                                  norm                  = self.args.norm,
                                  se_block              = self.args.se_block,
                                  activation            = self.args.activation,
                                  output_attention      = self.args.output_attention)
        else:
            raise NotImplementedError("late transformer") 

        return model.double()
    
    
    def _select_optimizer(self):
        # 这两个在train里面被调用
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError
            
        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), 
                                                               lr=self.args.learning_rate)
        return model_optim
    
    
    def _select_criterion(self):
        # 这两个在train里面被调用
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError
            
        criterion = self.criterion_dict[self.args.criterion]()
        return criterion

    def _get_data(self, flag="train"):
        args = self.args
        
        if flag == 'train':
            # 只有train需要被shuffle
            shuffle_flag = True
        else:
            shuffle_flag = False 
            
        data_set = CMAPSSData(data_path        = args.data_path, 
                              Data_id          = args.Data_id, 
                              sequence_length  = args.sequence_length,
                              MAXLIFE          = args.MAXLIFE,
                              flag             = flag,
                              difference       = args.difference,
                              normalization    = args.normalization,
                              validation       = args.validation)


        data_loader = DataLoader(data_set, 
                                 batch_size  =args.batch_size,
                                 shuffle     =shuffle_flag,
                                 num_workers =0,
                                 drop_last   =False)

        return data_set, data_loader
    
    
    def train(self, save_path):

        # 中间过程存储地址
        path = './logs/'+save_path
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 根据batch ssize 看一个epoch里面有多少训练步骤 以及validation的步骤
        train_steps = len(self.train_loader)
        print("train_steps: ",train_steps)
        print("test_steps: ",len(self.vali_loader))
        
        # 初始化 早停止
        early_stopping = EarlyStopping(patience=self.args.early_stop_patience, 
                                       verbose=True)
        # 初始化 学习率
        learning_rate_adapter = adjust_learning_rate_class(self.args,
                                                           True)
        # 选择优化器
        model_optim = self._select_optimizer()
        
        # 选择优化的loss function
        criterion =  self._select_criterion()
        
        
        print("start training")
        for epoch in range(self.args.train_epochs):    
            start_time = time()		
            
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)

                # model prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x)[0]
                else:
                    outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
    
                loss.backward()
                model_optim.step()

            end_time = time()	
            epoch_time = end_time - start_time
            train_loss = np.average(train_loss) # 这个整个epoch中的平均loss
            
            vali_loss  = self.validation(self.vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format(
                epoch + 1, train_steps, train_loss, vali_loss, epoch_time))
            
            # 在每个epoch 结束的时候 进行查看需要停止和调整学习率
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break


            learning_rate_adapter(model_optim,vali_loss)
        
        last_model_path = path+'/'+'last_checkpoint.pth'
        torch.save(self.model.state_dict(), last_model_path)
        

    def validation(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        for i, (batch_x,batch_y) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)
            
            # prediction
            if self.args.output_attention:
                outputs = self.model(batch_x)[0]
            else:
                outputs = self.model(batch_x)

            pred = outputs.detach()#.cpu()
            true = batch_y.detach()#.cpu()

            loss = criterion(pred, true) 

            total_loss.append(loss.item())
          
        average_vali_loss = np.average(total_loss)


        self.model.train()
        return average_vali_loss

#     def test(self, save_path):
#         test_data, test_loader = self._get_data(flag='test')
#         self.model.eval()
        
#         preds = []
#         trues = []
        
#         for i, (batch_x,batch_y) in enumerate(test_loader):
#             batch_x = batch_x.double().to(self.device)
#             batch_y = batch_y.long().to(self.device)

#             # prediction
#             if self.args.output_attention:
#                 outputs = self.model(batch_x)[0]
#             else:
#                 outputs = self.model(batch_x)

            
#             pred = list(np.argmax(outputs.detach().cpu().numpy(),axis=1))
#             true = list(batch_y.detach().cpu().numpy())
            
#             preds.extend(pred)
#             trues.extend(true)

#         preds = np.array(preds)
#         trues = np.array(trues)


#         # result save
#         path = './logs/'+save_path+'/'
#         if not os.path.exists(path):
#             os.makedirs(path)
			
#         df_metrics = calculate_metrics(trues, preds)
#         with open(path+"test_result.pickle", "wb") as f:
#             pickle.dump(df_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         return df_metrics
    
#     def calculate_metrics(self, y_true, y_pred):
#         res = pd.DataFrame(data=np.zeros((1, 3), dtype=np.float), index=[0],
#                            columns=['precision', 'accuracy', 'recall'])
#         res['precision'] = precision_score(y_true, y_pred, average='macro')
#         res['accuracy'] = accuracy_score(y_true, y_pred)
#         res['recall'] = recall_score(y_true, y_pred, average='macro')

#         return res


