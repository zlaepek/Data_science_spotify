import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pprint import pprint as pp
import time

#one-hot-encoding을 위한 함수
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix = column)], axis=1)
        data = data.drop(column, axis=1)
    return data


def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

#출처 : https://programmers.co.kr/learn/courses/21/lessons/11044#


# this function make combination of numerical data scaling and categorical data encoding.
# scaling -> 
# encoding ->
# 6개의 dataset이 리턴됩니당!

#인자로 Dataset, 그리고 encoder 할 칼럼 이름이 담긴 리스트?
def scale_encode_combination(dataset, numerical_feature_list, categorical_feature_list):
    start = time.time() 

    #scaler
    scaler_stndard = preprocessing.StandardScaler()
    scaler_MM = preprocessing.MinMaxScaler()
    scaler_robust = preprocessing.RobustScaler()
    scalers = [scaler_stndard, scaler_MM, scaler_robust]

    #encoder
    encoder_ordinal = preprocessing.OrdinalEncoder()
    #one hot encoding is using pd.get_dummies()

    #s+od, M+od, R+od, s+one, M+one, R+one
    result = []

    i = 0

    for scaler in scalers:
        result.append(dataset.copy())
        result[i][numerical_feature_list] = scaler.fit_transform(dataset[numerical_feature_list])
        result[i][categorical_feature_list] = encoder_ordinal.fit_transform(dataset[categorical_feature_list])
        i = i + 1

        result.append(dataset.copy())
        result[i][numerical_feature_list] = scaler.fit_transform(dataset[numerical_feature_list])
        result[i] = dummy_data(result[i], categorical_feature_list)
        i = i + 1


    dataset_type = ["standard_ordinal", "standard_onehot", "minmax_ordinal", "minmax_onehot", "robust_ordinal", "robust_onehot"]
    result_dict = {}

    for i, data in enumerate(result):
        result_dict[dataset_type[i]] = data

    print("The time that this function finish :", time.time() - start)
    return result_dict



##사용방법
dataset = pd.read_csv("tracks.csv")

numerical_feature_list = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
categorical_feature_list = ['time_signature', 'key']


#데이터셋, numerical column name, categorical column name을 넘겨주면 6개의 변형된 데이터 셋이 담긴 dictionary가 리턴됨
combination_dataset = scale_encode_combination(dataset, numerical_feature_list, categorical_feature_list)
pp(combination_dataset)