import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time

#for one-hot-encoding
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix = column)], axis=1)
        data = data.drop(column, axis=1)
    return data

# main function : this function make combination of numerical data scaling and categorical data encoding.
# output : six type scaling & encoding dataset
def scale_encode_combination(dataset, numerical_feature_list, categorical_feature_list):
    start = time.time() 

    #scaler
    scaler_stndard = preprocessing.StandardScaler()
    scaler_MM = preprocessing.MinMaxScaler()
    scaler_robust = preprocessing.RobustScaler()
    scaler_maxabs = preprocessing.MaxAbsScaler()
    scaler_normalize = preprocessing.Normalizer()
    scalers = [scaler_stndard, scaler_MM, scaler_robust, scaler_maxabs, scaler_normalize]
    scalers_name = ["standard", "minmax", "robust", "maxabs", "normalize"]

    #encoder
    encoder_ordinal = preprocessing.OrdinalEncoder()
    #one hot encoding => using pd.get_dummies() (not used preprocessing.OneHotEncoder())
    encoders_name = ["ordinal", "onehot"]

    result = []
    result_dict = {}
    i = 0

    for scaler in scalers:
        #===== scalers + ordinal encoding
        result.append(dataset.copy())

        #scaling
        result[i][numerical_feature_list] = scaler.fit_transform(dataset[numerical_feature_list])
        result[i][categorical_feature_list] = encoder_ordinal.fit_transform(dataset[categorical_feature_list])

        #save in dictionary
        dataset_type = scalers_name[int(i/2)] + "_" + encoders_name[i%2]
        result_dict[dataset_type] = result[i]
        i = i + 1


        #===== scalers + OneHot encoding
        result.append(dataset.copy())

        #encoding
        result[i][numerical_feature_list] = scaler.fit_transform(dataset[numerical_feature_list])
        result[i] = dummy_data(result[i], categorical_feature_list)

        #save in dictionary
        dataset_type = scalers_name[int(i/2)] + "_" + encoders_name[i%2]
        result_dict[dataset_type] = result[i]
        i = i + 1


    print("The time that this function finish :", time.time() - start)
    return result_dict
