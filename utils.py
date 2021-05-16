import warnings 
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tsfresh.feature_selection.significance_tests import target_real_feature_real_test, target_real_feature_binary_test
from sklearn import preprocessing


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
	
	
def batch_generator(training_data, sequence_length=15):
    """
    data generate for turbofan dataset
    Generator function for creating random batches of training-data
    """
    engine_ids = list(training_data["engine_id"].unique())
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