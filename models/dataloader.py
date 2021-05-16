import sys
sys.path.append("..")
import os
import pandas as pd
import numpy as np
from tsfresh.feature_selection.significance_tests import target_real_feature_real_test, target_real_feature_binary_test
from sklearn import preprocessing
from torch.utils.data import Dataset


def identify_and_remove_unique_columns(Dataframe):
    Dataframe = Dataframe.copy()
    del Dataframe["engine_id"]
    del Dataframe["cycle"]
    
 
    unique_counts = Dataframe.nunique()
    record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
    unique_to_drop = list(record_single_unique['feature'])
    Dataframe = Dataframe.drop(columns = unique_to_drop)
    

    unique_counts = Dataframe.nunique()
    record_single_unique = pd.DataFrame(unique_counts).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
    record_single_unique["type"] = record_single_unique["nunique"].apply(lambda x:"real" if x>2 else "binary")
    for i in range(record_single_unique.shape[0]):
        col = record_single_unique.loc[i,"feature"]
        _type = record_single_unique.loc[i,"type"]
        if _type == "real":
            p_value = target_real_feature_real_test(Dataframe[col], Dataframe["RUL"])
        else:
            le = preprocessing.LabelEncoder()
            p_value = target_real_feature_binary_test(pd.Series(le.fit_transform(Dataframe[col])), Dataframe["RUL"])
        if p_value>0.05:
            unique_to_drop.append(col)
    
    return  unique_to_drop



def Cmapss_train_vali_batch_generator(training_data, sequence_length=15):
    """
    data generate for turbofan dataset
    Generator function for creating random batches of training-data
    """
    
    engine_ids = list(training_data["engine_id"].unique())
    #print(engine_ids)
    temp = training_data.copy()
    for id_ in engine_ids:
        indexes = temp[temp["engine_id"] == id_].index
        traj_data = temp.loc[indexes]
        cutoff_cycle = max(traj_data['cycle']) - sequence_length  + 1
        
        if cutoff_cycle<=0:
            drop_range = indexes
            print("sequence_length + window_size is too large")
        else:
            cutoff_cycle_index = traj_data['cycle'][traj_data['cycle'] == cutoff_cycle+1].index
            drop_range = list(range(cutoff_cycle_index[0], indexes[-1] + 1))
            
        temp.drop(drop_range, inplace=True)
    indexes = list(temp.index)
    del temp
    
    feature_number = training_data.shape[1]-3

    x_shape = (len(indexes), sequence_length, feature_number)
    x_batch = np.zeros(shape=x_shape, dtype=np.float32)
    y_shape = (len(indexes), sequence_length)
    y_batch = np.zeros(shape=y_shape, dtype=np.float32)


    for batch_index, index in enumerate(indexes):
        y_batch[batch_index] = training_data.iloc[index:index+sequence_length,-1]
        x_batch[batch_index] = training_data.iloc[index:index+sequence_length, 2:-1].values



    
    return x_batch, y_batch   


def Cmapss_test_batch_generator(test_data, sequence_length=5):

    engine_ids = list(test_data["engine_id"].unique())

    feature_number = test_data.shape[1]-3 
    
    x_batch = []
    y_batch = []
    
    for _id in set(test_data['engine_id']):
        test_of_one_id =  test_data[test_data['engine_id'] == _id]
        
        if test_of_one_id.shape[0]>=sequence_length:
            x_batch.append(test_of_one_id.iloc[-sequence_length:,2:-1].values)
            y_batch.append(test_of_one_id.iloc[-sequence_length:,-1].values)
        

    return np.array(x_batch), np.array(y_batch)

def cmapss_data_loader(data_path, Data_id, sequence_length = 40,
                       MAXLIFE=120, flag="train",difference=False,
                       normalization="znorm",
                       validation=0.1):


    # --------------- read the train data, test data and labels for test ---------------

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

    train_FD = pd.read_table("{}/train_{}.txt".format(data_path,Data_id), header=None, delim_whitespace=True)
    train_FD.columns = column_name


    test_FD = pd.read_table("{}/test_{}.txt".format(data_path,Data_id), header=None, delim_whitespace=True)
    test_FD.columns = column_name

    RUL_FD = pd.read_table("{}/RUL_{}.txt".format(data_path,Data_id), header=None, delim_whitespace=True)


    # --------------- define the label for train and test ---------------
    # piecewise linear RUL  for Training data
    id='engine_id'
    rul = [] 
    for _id in set(train_FD[id]):
        trainFD_of_one_id =  train_FD[train_FD[id] == _id]
        cycle_list = trainFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list)

        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        for i in range(0, len(cycle_list)):
            # 
            if i < knee_point:
                kink_RUL.append(MAXLIFE)
            else:
                tmp = max_cycle-i-1
                kink_RUL.append(tmp)
        rul.extend(kink_RUL)

    train_FD["RUL"] = rul

    # piecewise linear RUL  for Test data
    id='engine_id'
    rul = []
    for _id_test in set(test_FD[id]):
        true_rul = int(RUL_FD.iloc[_id_test - 1])
        testFD_of_one_id =  test_FD[test_FD[id] == _id_test]
        cycle_list = testFD_of_one_id['cycle'].tolist()
        max_cycle = max(cycle_list) + true_rul
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        for i in range(0, len(cycle_list)):
            if i < knee_point:
                kink_RUL.append(MAXLIFE)
            else:
                tmp = max_cycle-i-1
                kink_RUL.append(tmp)    

        rul.extend(kink_RUL)

    test_FD["RUL"] = rul

    # --------------- acoording to the labels of training dataset, delete redundant input sensors ---------------

    col_to_drop = identify_and_remove_unique_columns(train_FD)
    train_FD = train_FD.drop(col_to_drop,axis = 1)
    test_FD = test_FD.drop(col_to_drop,axis = 1)
    #print(train_FD.shape)
    #print(test_FD.shape)


    # ---------------- difference ------------------------??????????????????????????????????????????


    # ---------------- Normalization --------------------------------

    if normalization == "znorm":
        mean = train_FD.iloc[:, 2:-1].mean()
        std = train_FD.iloc[:, 2:-1].std()
        std.replace(0, 1, inplace=True)


        # training dataset
        train_FD.iloc[:, 2:-1] = (train_FD.iloc[:, 2:-1] - mean) / std

        # Testing dataset
        test_FD.iloc[:, 2:-1] = (test_FD.iloc[:, 2:-1] - mean) / std

    # ------------------- batch generator -------------------------------
    number_fo_id = len(train_FD["engine_id"].unique())
    train_engine_id = train_FD["engine_id"].unique()[:int((number_fo_id)*(1-validation))]
    valid_engine_id = train_FD["engine_id"].unique()[int((number_fo_id)*(1-validation)):]

    if flag == "train":
        data_df = pd.DataFrame()
        for idx in train_engine_id:
            temp = train_FD[train_FD["engine_id"]==idx]
            data_df = pd.concat([data_df,temp])
        data_df.reset_index(inplace=True,drop=True)    
        data_x , data_y = Cmapss_train_vali_batch_generator(data_df,sequence_length)

    elif flag == "val":
        data_df = pd.DataFrame()
        for idx in valid_engine_id:
            temp = train_FD[train_FD["engine_id"]==idx]
            data_df = pd.concat([data_df,temp])
        data_df.reset_index(inplace=True,drop=True)    
        data_x , data_y = Cmapss_train_vali_batch_generator(data_df,sequence_length)

    else:
        data_x, data_y = Cmapss_test_batch_generator(test_FD, sequence_length)
    
    return data_x, data_y



class CMAPSSData(Dataset):

    def __init__(self, 
                 data_path, 
                 Data_id, 
                 sequence_length=40,
                 MAXLIFE=120,
                 flag='train',
                 difference=False,
                 normalization='znorm',
                 validation=0.1):
        
        
        self.data_path       = data_path
        self.Data_id         = Data_id
        self.sequence_length = sequence_length
        self.MAXLIFE         = MAXLIFE
        self.difference      = difference
        self.flag            = flag
        self.validation      = validation
        self.normalization   = normalization
        
        
        # check flag 

        self.__read_data__()

    def __read_data__(self):
        print("load the data ", self.data_path," ",self.Data_id)
        data_x, data_y = cmapss_data_loader(data_path       = self.data_path, 
                                            Data_id         = self.Data_id,
                                            sequence_length = self.sequence_length,
                                            MAXLIFE         = self.MAXLIFE,
                                            flag            = self.flag,
                                            difference      = self.difference,
                                            normalization   = self.normalization,
                                            validation      = self.validation)
        # data_x (n_samples, length, channel)  
        # data_y (n_samples, length)
        print(self.flag, ": the shape of data_X is : ", data_x.shape)


        self.data_x = data_x
        self.data_y = data_y
        self.data_channel = data_x.shape[2]
    
    def __getitem__(self, index):


        sample_x = self.data_x[index]
        #sample_y = self.data_y_original[index]
        sample_y = self.data_y[index]


        return sample_x,sample_y
    
    def __len__(self):
        return len(self.data_x)

